import torch
import torch.nn as nn
import torch.nn.functional as F


def printt(x, t=False):
    if t: print(x)


def sigmoid(x, alpha=10000):
    return 1/(1+torch.exp(-x*alpha))


class EGA_module(nn.Module):

    def __init__(self, dim_in=1, alpha=-1e+2, learnable=True, recalib=True):
        super().__init__()
        # Edge Detection
        self.conv = nn.Conv3d(dim_in, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sig = nn.Sigmoid()
        kernel = torch.ones([1, 1, 3, 3, 3])
        kernel[0, 0, 1, 1, 1] = alpha
        self.conv.weight.data = kernel
        if not learnable: self.conv.weight.requires_grad = False

        # DL-based Recalibration
        self.recalib = recalib
        if recalib:
            self.conv_recalib = nn.Sequential(
                nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(1),
                nn.Sigmoid())

    def forward(self, x):
        x = torch.mean(x, axis=1, keepdim=True)
        x = self.conv(x)
        x = self.sig(x)
        x = torch.clip(x, 0.5, 1)
        x = (x-0.5)/(1-0.5)

        if self.recalib: x = self.conv_recalib(x)

        return x


class SE_block(nn.Module):

    def __init__(self, dim_in, dim_out, norm='batch', dilation=1, act=nn.SiLU()):
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.act = act

        if norm=='batch': self.norm = nn.BatchNorm3d(dim_out)
        if norm=='group': self.norm = nn.GroupNorm(dim_out//8, dim_out)
        if norm=='instance': self.norm = nn.InstanceNorm3d(dim_out)

        self.encoder = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False),
            self.norm,
            self.act)

        self.spatial_att = nn.Sequential(
            nn.Conv3d(dim_out, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.take = nn.Sequential(
            nn.Conv3d(dim_out, 2, kernel_size=1, bias=False))

    def forward(self, x):
        # Base Encoding
        y = self.encoder(x)
        y = y*self.spatial_att(y)
        t = self.take(y)

        return y, t


class SSE_block(nn.Module):

    def __init__(self, dim_in, dim_out, norm='batch', dilation=1, act=nn.SiLU()):
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.act = act

        if norm=='batch': self.norm = nn.BatchNorm3d(dim_out)
        if norm=='group': self.norm = nn.GroupNorm(dim_out//8, dim_out)
        if norm=='instance': self.norm = nn.InstanceNorm3d(dim_out)

        self.encoder = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False),
            self.norm,
            self.act)

        self.spatial_att_1 = nn.Sequential(
            nn.Conv3d(dim_out, 1, kernel_size=1, bias=False),
            nn.Sigmoid())
        self.spatial_att_2 = nn.Sequential(
            nn.Conv3d(dim_out, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.take = nn.Sequential(
            nn.Conv3d(dim_out, 2, kernel_size=1, bias=False))

    def forward(self, x):
        # Base Encoding
        y = self.encoder(x)
        y = y*self.spatial_att_1(y)
        y = y*self.spatial_att_2(y)
        t = self.take(y)

        return y, t


class Droplayer(nn.Module):

    def __init__(self, channel_num=1, thr=0.3):
        super().__init__()
        self.channel_num = channel_num
        self.threshold = thr

    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0], self.channel_num, 1, 1, 1, device=x.device, dtype=x.dtype)
            r = (r>=self.threshold).to(x.dtype)
            r = r*self.channel_num/(r.sum()+0.01)

            return x*r
        else:

            return x


class CorrModule(nn.Module):

    def __init__(self, window=2, pool=True, down=True):
        super().__init__()
        self.window = window*2
        self.pad = window
        self.pool = pool
        self.down = down

        self.expand_2 = nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=False)
        self.expand_4 = nn.Upsample(scale_factor=4.0, mode='trilinear', align_corners=False)
        self.reduce_2 = nn.Upsample(scale_factor=1/2, mode='trilinear', align_corners=False)
        self.reduce_4 = nn.Upsample(scale_factor=1/4, mode='trilinear', align_corners=False)

    def forward(self, x, y=None):
        '''
        x: Fmap
        y: Deep Fmap
        '''
        with torch.no_grad():
            if y==None: y=x
            x_dtype = x.dtype
            if self.down:
                x = self.reduce_2(x)
                y = self.reduce_2(y)

            x, y = x.to(torch.float16), y.to(torch.float16)
            r = int(x.shape[-1]/y.shape[-1])
            if r==2: y = self.expand_2(y)
            if r==4: y = self.expand_4(y)

            c = int(y.shape[1]/x.shape[1])
            if c!=1: y = y.reshape(y.shape[0], y.shape[1]//c, c, y.shape[2], y.shape[3], y.shape[4]).mean(dim=2)

            s = x.shape[-1]
            y = F.pad(y, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
            y = y.unfold(2, s, 1).unfold(3, s, 1).unfold(4, s, 1)
            y = y.reshape(y.shape[0], y.shape[1], -1, s, s, s)
            y = torch.swapaxes(y, 1, 2)
            x = x.unsqueeze(1)

            norm_x = torch.sqrt(torch.sum(x*x, dim=2))
            norm_y = torch.sqrt(torch.sum(y*y, dim=2))
            res = torch.sum(x*y, axis=2) / (norm_x*norm_y + 1e-5)

            if (self.pool) & (r==2): res = self.reduce_2(res)
            if (self.pool) & (r==4): res = self.reduce_4(res)
            if self.down: res = self.expand_2(res)

        return res.to(x_dtype)
