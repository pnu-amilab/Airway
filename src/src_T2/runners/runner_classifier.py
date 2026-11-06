import torch
import torch.nn.functional as F

from common.losses.loss import DSC, PRE, TDR, BDR
from common.runners.runner_base import Runner_Base
from src_T2.utils.loss import BranchCCE, BranchConsistency, ForegroundAccuracy, ForegroundCCE, ForegroundHierarchicalCCE, HierarchicalDistanceLoss, MultiClassDice
from src_T2.utils.utils_dataloader import Dataset_AW_T2
from src_T2.utils.utils_model import load_model_T2


class Runner_Classifier_T2(Runner_Base):

    def __init__(self, hp):
        super().__init__(hp)
        self.loss_keys = ['CCE', 'HCE', 'HDL', 'BrCE', 'BrCon']
        self.rate_keys = ['DSC', 'PRE', 'TDR', 'BDR', 'mDSC', 'ACC', 'BrACC', 'Score']
        self.CCE_func = ForegroundCCE(ignore_index=hp.background_index)
        self.HCE_func = ForegroundHierarchicalCCE(ignore_index=hp.background_index)
        self.HDL_func = HierarchicalDistanceLoss(num_classes=hp.num_classes-1, ignore_index=hp.background_index).to(hp.device)
        self.BrCE_func = BranchCCE(ignore_index=hp.background_index, min_voxels=hp.branch_min_voxels)
        self.BrCon_func = BranchConsistency(ignore_index=hp.background_index, min_voxels=hp.branch_min_voxels)
        self.mDSC_func = MultiClassDice(num_classes=hp.num_classes, ignore_index=hp.background_index)
        self.ACC_func = ForegroundAccuracy(ignore_index=hp.background_index)
        self.DSC_func = DSC()
        self.PRE_func = PRE()
        self.TDR_func = TDR()
        self.BDR_func = BDR()

    def build_scheduler(self, optimizer, total_steps):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hp.scheduler_step, gamma=self.hp.scheduler_gamma)

    def run_epoch(self, phase, loader, model, optimizer, scaler, scheduler):
        return super().run_epoch(phase, loader, model, optimizer, scaler, scheduler=None)

    def _clip_gradients(self, model):
        pass

    def build_model(self):
        return load_model_T2(self.hp, load_init=self.hp.epoch_load==0).to(self.hp.device)

    def build_dataset(self):
        train_set = Dataset_AW_T2(self.hp, phase='train')
        valid_set = Dataset_AW_T2(self.hp, phase='valid')
        self.CCE_func = ForegroundCCE(ignore_index=self.hp.background_index, weight=train_set.class_weight.to(self.hp.device))

        return train_set, valid_set

    def get_batch(self, data):
        return data[0].to(self.hp.device), data[1].to(self.hp.device), data[2].to(self.hp.device)

    def forward(self, data, model):
        X, y, CK = data
        c1_logits, c2_logits, emb = model(X)
        T1_mask = X[:, 2] > 0.5
        CK_train = torch.where(T1_mask, CK, torch.zeros_like(CK))

        return {'X':X, 'y':y, 'CK':CK, 'CK_train':CK_train, 'c1_logits':c1_logits, 'c2_logits':c2_logits, 'c_rate':c2_logits, 'emb':emb}

    def _branch_accuracy(self, logits, target, branch_map):
        values = []
        for batch_idx in range(logits.shape[0]):
            valid = (target[batch_idx] > 0) & (branch_map[batch_idx] > 0)
            if not torch.any(valid): continue
            branch = branch_map[batch_idx][valid].long()
            target_fg = target[batch_idx][valid].long()-1
            probability = torch.softmax(logits[batch_idx], dim=0).permute(1, 2, 3, 0)[valid]
            _, inverse = torch.unique(branch, sorted=False, return_inverse=True)
            n = int(inverse.max().item())+1
            count = torch.zeros(n, device=logits.device)
            count.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
            p_branch = torch.zeros(n, logits.shape[1], device=logits.device)
            p_branch.index_add_(0, inverse, probability.float())
            t_branch = torch.zeros_like(p_branch)
            t_branch.index_add_(0, inverse, F.one_hot(target_fg, num_classes=logits.shape[1]).float())
            keep = count >= self.hp.branch_min_voxels
            if torch.any(keep): values.append((p_branch[keep].argmax(dim=1) == t_branch[keep].argmax(dim=1)).float().mean())
        if len(values) == 0: return logits.sum()*0.0

        return torch.stack(values).mean()

    def forward_rate(self, res):
        X, y, CK = res['X'], res['y'], res['CK']
        geometry = X[:, [2]] > 0.5
        class_prediction = torch.argmax(res['c_rate'], dim=1)+1
        class_prediction = torch.where(geometry[:, 0], class_prediction, torch.zeros_like(class_prediction))
        rate_dict = {
            'DSC':self.DSC_func(geometry, y[:, [0]]),
            'PRE':self.PRE_func(geometry, y[:, [0]]),
            'TDR':self.TDR_func(geometry, y[:, [1]]),
            'BDR':self.BDR_func(geometry, y[:, [1]], y[:, [2]]),
            'mDSC':self.mDSC_func(class_prediction, CK),
            'ACC':self.ACC_func(class_prediction, CK),
            'BrACC':self._branch_accuracy(res['c_rate'], res['CK_train'], y[:, 2])}
        rate_dict['Score'] = (rate_dict['mDSC']+rate_dict['ACC']+rate_dict['BrACC'])/3

        return rate_dict

    def forward_loss(self, res):
        c1_logits, c2_logits = res['c1_logits'], res['c2_logits']
        CK = res['CK_train']
        branch_map = res['y'][:, 2]
        weight_E = self.hp.deep_supervision_weight
        weight_D = 1-weight_E
        loss_E = {
            'CCE':self.CCE_func(c1_logits, CK),
            'HCE':self.HCE_func(c1_logits, CK),
            'HDL':self.HDL_func(c1_logits, CK),
            'BrCE':self.BrCE_func(c1_logits, CK, branch_map),
            'BrCon':self.BrCon_func(c1_logits, CK, branch_map)}
        loss_D = {
            'CCE':self.CCE_func(c2_logits, CK),
            'HCE':self.HCE_func(c2_logits, CK),
            'HDL':self.HDL_func(c2_logits, CK),
            'BrCE':self.BrCE_func(c2_logits, CK, branch_map),
            'BrCon':self.BrCon_func(c2_logits, CK, branch_map)}
        loss_final = sum((weight_E*loss_E[key]+weight_D*loss_D[key])*self.hp.loss_weight[key] for key in self.loss_keys)

        return loss_D, loss_final
