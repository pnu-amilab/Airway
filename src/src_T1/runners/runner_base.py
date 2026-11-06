import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset.utils_dataloader import Dataset_AW
from common.losses.loss import GUL, BCE, CTL, BGR, DSC, PRE, TDR, BDR
from common.runners.runner_base import Runner_Base as Runner_Common
from src_T1.utils.utils_model import load_model


class Runner_Base(Runner_Common):

    def __init__(self, hp):
        super().__init__(hp)
        self.loss_keys = ['GUL', 'BCE'] if hp.name_group == 'Others' else ['GUL', 'BCE', 'CTL', 'BGR']
        self.rate_keys = ['DSC', 'PRE', 'TDR', 'BDR']
        self.set_epoch(0, hp.epochs)
        self.BCE_func = BCE()
        self.CTL_func = CTL()
        self.BGR_func = BGR(alpha=hp.bgr_alpha)
        self.DSC_func = DSC()
        self.PRE_func = PRE()
        self.TDR_func = TDR()
        self.BDR_func = BDR()

    def set_epoch(self, epoch, epoch_e):
        progress = (epoch+1)/epoch_e
        GUL_alpha = self.hp.gul_alpha_start+(self.hp.gul_alpha_end-self.hp.gul_alpha_start)*progress
        self.GUL_func = GUL(alpha=GUL_alpha, gamma=self.hp.gul_gamma)
        if hasattr(self.hp, 'loss_weight_start') and hasattr(self.hp, 'loss_weight_end'):
            warmup = self.hp.loss_schedule_warmup
            weight_progress = float(np.clip((progress-warmup)/(1-warmup), 0, 1))
            self.hp.loss_weight = {key:self.hp.loss_weight_start[key]+(self.hp.loss_weight_end[key]-self.hp.loss_weight_start[key])*weight_progress for key in self.hp.loss_weight_start}

    def build_scheduler(self, optimizer, total_steps):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hp.scheduler_step, gamma=self.hp.scheduler_gamma)

    def run_epoch(self, phase, loader, model, optimizer, scaler, scheduler):
        return super().run_epoch(phase, loader, model, optimizer, scaler, scheduler=None)

    def _clip_gradients(self, model):
        pass

    def build_model(self):
        return load_model(self.hp).to(self.hp.device)

    def build_dataset(self):
        return Dataset_AW(self.hp, phase='train'), Dataset_AW(self.hp, phase='valid')

    def build_dataloader(self):
        train_set, valid_set = self.build_dataset()
        kwargs = {'batch_size':self.hp.batch_size, 'num_workers':self.hp.num_workers, 'pin_memory':torch.cuda.is_available()}
        if self.hp.num_workers > 0:
            kwargs['prefetch_factor'] = 2
            kwargs['persistent_workers'] = True
        train_loader = DataLoader(dataset=train_set, shuffle=True, **kwargs)
        valid_loader = DataLoader(dataset=valid_set, shuffle=True, **kwargs)

        return [train_loader, valid_loader]

    def forward_rate(self, res):
        p = res['p_rate']
        y = res['y']
        r = p > 0.5

        return {'DSC':self.DSC_func(r, y[:, [0]]),
                'PRE':self.PRE_func(r, y[:, [0]]),
                'TDR':self.TDR_func(r, y[:, [1]]),
                'BDR':self.BDR_func(r, y[:, [1]], y[:, [2]])}
