'''
WingsNet
"Alleviating Class-wise Gradient Imbalance for Pulmonary Airway Segmentation"
IEEE transactions on medical imaging, 40(9), 2452-2462.
'''
import torch
import torch.nn as nn

from src_T1.networks.modules import *


class SE_block_origin(nn.Module):

    def __init__(self, dim_in, dim_out, norm='batch', dilation=1, act=nn.SiLU(), size=128):
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
            nn.Conv3d(dim_out, 2, kernel_size=1, bias=False),
            nn.Upsample(size=[size, size, size], mode='trilinear', align_corners=True))

    def forward(self, x):
        # Base Encoding
        y = self.encoder(x)
        y = y*self.spatial_att(y)
        t = self.take(y)

        return y, t


class SSE_block_origin(nn.Module):

    def __init__(self, dim_in, dim_out, norm='batch', dilation=1, act=nn.SiLU(), size=128):
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
            nn.Conv3d(dim_out, 2, kernel_size=1, bias=False),
            nn.Upsample(size=[size, size, size], mode='trilinear', align_corners=True))

    def forward(self, x):
        # Base Encoding
        y = self.encoder(x)
        y = y*self.spatial_att_1(y)
        y = y*self.spatial_att_2(y)
        t = self.take(y)

        return y, t


class WingsNet(nn.Module):

    def __init__(self, size=64, dim_in=1, dim_out=1, dim_base=8):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_base = dim_base

        self.e_conv_1 = SE_block_origin(dim_in, dim_base*1, size=size)
        self.e_conv_2 = SE_block_origin(dim_base*1, dim_base*2, size=size)
        self.e_conv_3 = SE_block_origin(dim_base*2, dim_base*4, dilation=2, size=size)

        self.e_conv_4 = SSE_block_origin(dim_base*4, dim_base*4, size=size)
        self.e_conv_5 = SSE_block_origin(dim_base*4, dim_base*4, dilation=2, size=size)
        self.e_conv_6 = SSE_block_origin(dim_base*4, dim_base*8, dilation=2, size=size)

        self.e_conv_7 = SSE_block_origin(dim_base*8, dim_base*8, size=size)
        self.e_conv_8 = SSE_block_origin(dim_base*8, dim_base*8, dilation=2, size=size)
        self.e_conv_9 = SSE_block_origin(dim_base*8, dim_base*8, dilation=2, size=size)

        self.b_conv_1 = SSE_block_origin(dim_base*8, dim_base*8, size=size)
        self.b_conv_2 = SSE_block_origin(dim_base*8, dim_base*8, size=size)
        self.b_conv_3 = SSE_block_origin(dim_base*8, dim_base*8, size=size)

        self.d_conv_9 = SSE_block_origin(dim_base*8*2, dim_base*8, size=size)
        self.d_conv_8 = SSE_block_origin(dim_base*8, dim_base*8, size=size)

        self.d_conv_6 = SSE_block_origin(dim_base*8*2, dim_base*8, size=size)
        self.d_conv_5 = SSE_block_origin(dim_base*8, dim_base*4, size=size)

        self.d_conv_3 = SE_block_origin(dim_base*4*2, dim_base*4, size=size)
        self.d_conv_2 = SE_block_origin(dim_base*4, dim_base*2, size=size)

        self.expand = nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        self.reduce = nn.MaxPool3d(2, 2)
        self.last_1 = nn.Sequential(
            Droplayer(24, 0.2),
            nn.Conv3d(24, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())
        self.last_2 = nn.Sequential(
            Droplayer(12, 0.2),
            nn.Conv3d(12, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        e1, et1 = self.e_conv_1(x)
        e2, et2 = self.e_conv_2(e1)
        e3, et3 = self.e_conv_3(e2)

        e4, et4 = self.e_conv_4(self.reduce(e3))
        e5, et5 = self.e_conv_5(e4)
        e6, et6 = self.e_conv_6(e5)

        e7, et7 = self.e_conv_7(self.reduce(e6))
        e8, et8 = self.e_conv_8(e7)
        e9, et9 = self.e_conv_9(e8)

        b1, bt1 = self.b_conv_1(self.reduce(e9))
        b2, bt2 = self.b_conv_2(b1)
        b3, bt3 = self.b_conv_3(b2)

        d9, dt9 = self.d_conv_9(torch.concat([self.expand(b3), e9], axis=1))
        d8, dt8 = self.d_conv_8(d9)

        d6, dt6 = self.d_conv_6(torch.concat([self.expand(d8), e6], axis=1))
        d5, dt5 = self.d_conv_5(d6)

        d3, dt3 = self.d_conv_3(torch.concat([self.expand(d5), e3], axis=1))
        d2, dt2 = self.d_conv_2(d3)

        last1 = torch.concat([et1, et2, et3, et4, et5, et6, et7, et8, et9, bt1, bt2, bt3], axis=1)
        last2 = torch.concat([dt9, dt8, dt6, dt5, dt3, dt2], axis=1)

        last1 = self.last_1(last1)
        last2 = self.last_2(last2)

        return last1, last2, b3
