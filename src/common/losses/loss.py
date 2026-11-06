import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, p, y, w=None):
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if w is not None: w = w.reshape(w.shape[0], -1)
        else: w = 1
        hat = torch.sum(w*(p*y*2))
        cup = torch.sum(w*(p+y))+self.smooth

        return 1-hat/cup


class GUL(nn.Module):

    def __init__(self, alpha=0.1, gamma=0.7, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma

    def forward(self, p, y, w=None):
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if w is not None: w = w.reshape(w.shape[0], -1)
        else: w = 1
        hat = torch.sum(w*torch.pow(p+1e-3, self.gamma)*y)
        cup = torch.sum(w*(self.alpha*p+self.beta*y))+self.smooth

        return 1-hat/cup


class CTL(nn.Module):

    def __init__(self, gamma=0.7, smooth=1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, p, y, w=None):
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if w is not None: w = w.reshape(w.shape[0], -1)
        else: w = 1
        hat = torch.sum(w*torch.pow(p+1e-3, self.gamma)*y)
        cup = torch.sum(w*y)+self.smooth

        return 1-hat/cup


class BCE(nn.Module):

    def __init__(self, alpha=1, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, p, y, weight=None):
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        bce = -(y*torch.log(p+self.smooth)+(1-y)*torch.log(1-p+self.smooth))
        if weight is not None: bce = bce*weight.reshape(p.shape[0], -1)

        return torch.mean(bce)*self.alpha


class BGR(nn.Module):

    def __init__(self, scale_factor=8, alpha=1e-5, mode='trilinear', align_corners=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, emb, y):
        y = F.interpolate(y, size=emb.shape[-3:], mode=self.mode, align_corners=self.align_corners)>0
        background = (~y).to(emb.dtype)

        return ((emb*background)**2).mean()*self.alpha


class DSC(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, p, y):
        p = p.flatten()
        y = y.flatten()
        hat = p*y*2
        cup = p+y+self.smooth

        return hat.sum()/cup.sum()


class PRE(nn.Module):

    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, p, y):
        p = p.flatten()
        y = y.flatten()

        return (torch.sum(p*y)+self.smooth)/(torch.sum(p)+self.smooth)


class TDR(nn.Module):

    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, p, s):
        p = p.flatten()
        s = s.flatten()

        return (torch.sum(p*s)+self.smooth)/(torch.sum(s)+self.smooth)


class BDR(nn.Module):

    def __init__(self):
        super().__init__()
        self.T = 0.8

    def forward(self, p, s, t):
        p = p.flatten().to(torch.int16)
        s = s.flatten().to(torch.int16)
        t = t.flatten().to(torch.int16)
        b_true = s*t
        b_true_count = torch.bincount(b_true)[1:]
        num_branch_true = b_true_count.shape[0]
        if num_branch_true == 0: return p.sum()*0.0+1.0
        b_pred = b_true*p
        b_pred_count = torch.bincount(b_pred)[1:]
        num_branch_pred = b_pred_count.shape[0]
        if num_branch_true != num_branch_pred:
            lack_num = num_branch_true-num_branch_pred
            b_pred_count = torch.concat([b_pred_count, torch.zeros(lack_num, device=p.device, dtype=b_pred_count.dtype)])
        bdr = b_pred_count/b_true_count
        bdr = torch.nan_to_num(bdr, nan=1)

        return torch.mean((bdr>=self.T).to(torch.float32))
