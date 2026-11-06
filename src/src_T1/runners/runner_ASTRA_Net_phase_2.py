import random

import torch

from common.utils.checkpoint_compat import load_options
import torch.nn as nn
import torch.nn.functional as F

from src_T1.utils.utils_model import load_model
from src_T1.runners.runner_ASTRA_Net_phase_1 import Runner_ASTRA_Net_phase_1


class Runner_ASTRA_Net_phase_2(Runner_ASTRA_Net_phase_1):

    def __init__(self, hp):
        super().__init__(hp)
        self.loss_keys = ['GUL', 'BCE', 'KDL', 'IDL']
        self.MSE_func = nn.MSELoss()

    def build_model(self):
        hp_old = load_options(f'{self.hp.path_Res}/{self.hp.name_teacher}/options.pt')
        self.model_T = load_model(hp_old, epoch_load=hp_old.epochs).to(self.hp.device)
        model_S = load_model(hp_old, epoch_load=hp_old.epochs).to(self.hp.device)
        self.model_T.eval()
        for p in self.model_T.parameters(): p.requires_grad = False

        return model_S

    def forward(self, data, model):
        X, y = data
        scale = [random.choice(sublist) for sublist in [[0.5], [1], [1]]]
        scale_inv = [1/i for i in scale]
        self.model_T.eval()
        with torch.no_grad():
            S = F.interpolate(X, scale_factor=scale, mode='trilinear', align_corners=True)
            p1, p2, emb_p = self.model_T(X)
            p1 = (p1 > 0.2).to(torch.float32)
            p2 = (p2 > 0.2).to(torch.float32)

        s1, s2, emb_s = model(S)
        i1, i2, emb_i = model(X)
        s1 = F.interpolate(s1, scale_factor=scale_inv, mode='trilinear', align_corners=True)
        s2 = F.interpolate(s2, scale_factor=scale_inv, mode='trilinear', align_corners=True)
        emb_s = F.interpolate(emb_s, scale_factor=scale_inv, mode='trilinear', align_corners=True)

        return {'p1':p1, 'p2':p2, 'emb_p':emb_p, 's1':s1, 's2':s2, 'emb_s':emb_s, 'i1':i1, 'i2':i2, 'emb_i':emb_i, 'p_rate':s2, 'y':y}

    def forward_loss(self, res):
        p2 = res['p2']
        GUL_res_E = self.GUL_func(res['s1'], p2)
        GUL_res_D = self.GUL_func(res['s2'], p2)
        BCE_res_E = self.BCE_func(res['s1'], p2)
        BCE_res_D = self.BCE_func(res['s2'], p2)
        KDL_res_s = self.MSE_func(res['emb_p'], res['emb_s'])
        KDL_res = KDL_res_s
        IDL_GUL_E = self.GUL_func(res['i1'], p2)
        IDL_GUL_D = self.GUL_func(res['i2'], p2)
        IDL_BCE_E = self.BCE_func(res['i1'], p2)
        IDL_BCE_D = self.BCE_func(res['i2'], p2)
        IDL_res = IDL_GUL_E + IDL_GUL_D + IDL_BCE_E + IDL_BCE_D
        loss_dict = {}
        loss_dict['GUL'] = GUL_res_D
        loss_dict['BCE'] = BCE_res_D
        loss_dict['KDL'] = KDL_res
        loss_dict['IDL'] = IDL_res
        loss_GUL = (GUL_res_E + GUL_res_D) / 2
        loss_BCE = (BCE_res_E + BCE_res_D) / 2
        loss_final = (loss_GUL * self.hp.loss_weight['GUL'] +
                      loss_BCE * self.hp.loss_weight['BCE'] +
                      KDL_res * self.hp.loss_weight['KDL'] +
                      IDL_res * self.hp.loss_weight['IDL'])

        return loss_dict, loss_final
