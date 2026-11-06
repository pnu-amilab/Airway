'''
ASTRA-Net
"Discovery of Peripheral Airway Beyond Incomplete CT Annotations for Navigational Bronchoscopy"
IEEE Transactions on Medical Imaging (2026).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .modules import *
else:
    from modules import *


class ASTRA_Net(nn.Module):

    def __init__(self, dim_in=1, dim_out=1, dim_base=8, corr_vol=False,
                 ega=True, ega_learnable=True, ega_recalibration=False):
        super().__init__()
        self.corr_vol = corr_vol
        self.ega = ega
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_base = dim_base

        self.e_conv_1 = SE_block(dim_in, dim_base*1)
        self.e_conv_2 = SE_block(dim_base*1, dim_base*2)
        self.e_conv_3 = SE_block(dim_base*2, dim_base*4, dilation=2)

        self.e_conv_4 = SSE_block(dim_base*4, dim_base*4)
        self.e_conv_5 = SSE_block(dim_base*4, dim_base*4, dilation=2)
        self.e_conv_6 = SSE_block(dim_base*4, dim_base*8, dilation=2)

        self.e_conv_7 = SSE_block(dim_base*8, dim_base*8)
        self.e_conv_8 = SSE_block(dim_base*8, dim_base*8, dilation=2)
        self.e_conv_9 = SSE_block(dim_base*8, dim_base*8, dilation=2)

        self.b_conv_1 = SSE_block(dim_base*8, dim_base*8)
        self.b_conv_2 = SSE_block(dim_base*8, dim_base*8)
        self.b_conv_3 = SSE_block(dim_base*8, dim_base*8)

        q = 125 if corr_vol else 0
        self.d_conv_9 = SSE_block(dim_base*8*2+q, dim_base*8)
        self.d_conv_8 = SSE_block(dim_base*8, dim_base*8)

        self.d_conv_6 = SSE_block(dim_base*8*2+q, dim_base*8)
        self.d_conv_5 = SSE_block(dim_base*8, dim_base*4)

        self.d_conv_3 = SE_block(dim_base*4*2+q, dim_base*4)
        self.d_conv_2 = SE_block(dim_base*4, dim_base*2)

        self.last_1 = nn.Sequential(
            Droplayer(18+6, 0.2),
            nn.Conv3d(18+6, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())
        self.last_2 = nn.Sequential(
            Droplayer(12+6, 0.2),
            nn.Conv3d(12+6, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

        self.expand = nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        self.reduce = nn.MaxPool3d(2, 2)
        if self.corr_vol: self.corr_module = CorrModule()
        if self.ega: self.ega_module = EGA_module(learnable=ega_learnable, recalib=ega_recalibration)

    def forward(self, x):
        s1, s2, s3 = x.shape[-3], x.shape[-2], x.shape[-1]
        e1, et1 = self.e_conv_1(x)
        e2, et2 = self.e_conv_2(e1)
        e3, et3 = self.e_conv_3(e2) #

        e4, et4 = self.e_conv_4(self.reduce(e3))
        e5, et5 = self.e_conv_5(e4)
        e6, et6 = self.e_conv_6(e5) #

        e7, et7 = self.e_conv_7(self.reduce(e6))
        e8, et8 = self.e_conv_8(e7)
        e9, et9 = self.e_conv_9(e8) #

        b1, bt1 = self.b_conv_1(self.reduce(e9))
        b2, bt2 = self.b_conv_2(b1)
        b3, bt3 = self.b_conv_3(b2)

        et1 = F.interpolate(et1, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et2 = F.interpolate(et2, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et3 = F.interpolate(et3, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et4 = F.interpolate(et4, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et5 = F.interpolate(et5, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et6 = F.interpolate(et6, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et7 = F.interpolate(et7, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et8 = F.interpolate(et8, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        et9 = F.interpolate(et9, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        bt1 = F.interpolate(bt1, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        bt2 = F.interpolate(bt2, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        bt3 = F.interpolate(bt3, size=(s1, s2, s3), mode='trilinear', align_corners=True)

        last1 = torch.concat([et1, et2, et3, et4, et5, et6, et7, et8, et9, bt1, bt2, bt3], axis=1)
        last1 = self.last_1(last1)

        if self.corr_vol:
            c3 = self.corr_module(e3, e3)
            c6 = self.corr_module(e3, e6)
            c9 = self.corr_module(e3, e9)
            e3 = torch.concat([e3, c3], axis=1)
            e6 = torch.concat([e6, c6], axis=1)
            e9 = torch.concat([e9, c9], axis=1)

        if self.ega:
            ea_att_score = self.ega_module(last1)
            e3 = e3*ea_att_score
            e6 = e6*self.reduce(ea_att_score)
            e9 = e9*self.reduce(self.reduce(ea_att_score))

        d9, dt9 = self.d_conv_9(torch.concat([self.expand(b3), e9], axis=1))
        d8, dt8 = self.d_conv_8(d9)

        d6, dt6 = self.d_conv_6(torch.concat([self.expand(d8), e6], axis=1))
        d5, dt5 = self.d_conv_5(d6)

        d3, dt3 = self.d_conv_3(torch.concat([self.expand(d5), e3], axis=1))
        d2, dt2 = self.d_conv_2(d3)

        dt9 = F.interpolate(dt9, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        dt8 = F.interpolate(dt8, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        dt6 = F.interpolate(dt6, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        dt5 = F.interpolate(dt5, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        dt3 = F.interpolate(dt3, size=(s1, s2, s3), mode='trilinear', align_corners=True)
        dt2 = F.interpolate(dt2, size=(s1, s2, s3), mode='trilinear', align_corners=True)

        last2 = torch.concat([bt1, bt2, bt3, dt9, dt8, dt6, dt5, dt3, dt2], axis=1)
        last2 = self.last_2(last2)

        return last1, last2, b3
