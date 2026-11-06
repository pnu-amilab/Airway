'''
UDADS-Net
"Airway Segmentation using Uncertainty-based Double Attention Detail Supplement Network"
Biomedical Signal Processing and Control 105 (2025): 107648.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def printt(x, t=False):
    if t: print(x)


def conv_block(dim_in, dim_out, k=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(dim_in, dim_out, k, stride, padding),
        nn.BatchNorm3d(dim_out),
        nn.ReLU())

# UDA (Uncertainty-based Double Attention Module)


class UDAModule(nn.Module):

    def __init__(self, dim_deep, dim_skip, T_M=0.8, T_m=0.1):
        super(UDAModule, self).__init__()
        self.T_M = T_M
        self.T_m = T_m
        self.compress = nn.Conv3d(dim_deep, 1, kernel_size=1)
        self.spatial_att_conv = nn.Conv3d(1, 1, kernel_size=1)
        self.ams = AMSModule(dim_skip)

    def forward(self, deep_feat, skip_feat):
        deep_feat_up = F.interpolate(deep_feat, size=skip_feat.shape[2:], mode='trilinear', align_corners=False)
        prob = torch.sigmoid(self.compress(deep_feat_up))
        mask = ((prob > self.T_m) & (prob < self.T_M)).float()
        spatial_att = torch.sigmoid(self.spatial_att_conv(mask))

        skip_feat_att = skip_feat * spatial_att
        skip_feat_ams = self.ams(skip_feat_att)

        out = torch.cat([skip_feat_ams, deep_feat_up], dim=1)

        return out

# AMS (Adaptive Multi-Scale Module)


class AMSModule(nn.Module):

    def __init__(self, dim_base):
        super(AMSModule, self).__init__()
        self.dilated_conv1 = nn.Conv3d(dim_base, dim_base, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv3d(dim_base, dim_base, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv3 = nn.Conv3d(dim_base, dim_base, kernel_size=3, padding=3, dilation=3)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim_base, dim_base // 2),
            nn.ReLU(),
            nn.Linear(dim_base // 2, dim_base),
            nn.Sigmoid())

    def forward(self, x):
        f1 = self.dilated_conv1(x)
        f2 = self.dilated_conv2(x)
        f3 = self.dilated_conv3(x)
        f = f1 + f2 + f3

        weight = self.gap(f).view(f.size(0), f.size(1))
        weight = self.fc(weight).view(f.size(0), f.size(1), 1, 1, 1)
        out = f * weight

        return out

# UDADS-Net (Uncertainty-based Double Attention Detail Supplement Network)


class UDADS_Net(nn.Module):

    def __init__(self, dim_base=32, shape=False):
        super().__init__()
        self.shape = shape
        self.encoder_1 = conv_block(1, dim_base)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder_2 = conv_block(dim_base*1, dim_base*2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder_3 = conv_block(dim_base*2, dim_base*4)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder_4 = conv_block(dim_base*4, dim_base*8)

        self.uda3 = UDAModule(dim_base*8, dim_base*4)
        self.uda2 = UDAModule(dim_base*4, dim_base*2)
        self.uda1 = UDAModule(dim_base*2, dim_base*1)

        self.d3 = conv_block(dim_base*12, dim_base*4)
        self.d2 = conv_block(dim_base*6, dim_base*2)
        self.d1 = conv_block(dim_base*3, dim_base*1)
        self.final_conv_3 = nn.Conv3d(dim_base*4, 1, kernel_size=1)
        self.final_conv_2 = nn.Conv3d(dim_base*2, 1, kernel_size=1)
        self.final_conv_1 = nn.Conv3d(dim_base*1, 1, kernel_size=1)

    def forward(self, x):
        printt(f'{"x":<15}: {x.shape}', t=self.shape)
        e1 = self.encoder_1(x)
        printt(f'{"e1":<15}: {e1.shape}', t=self.shape)
        e1p = self.pool1(e1)
        e2 = self.encoder_2(e1p)
        printt(f'{"e2":<15}: {e2.shape}', t=self.shape)
        e2p = self.pool2(e2)
        e3 = self.encoder_3(e2p)
        printt(f'{"e3":<15}: {e3.shape}', t=self.shape)
        e3p = self.pool3(e3)
        e4 = self.encoder_4(e3p)
        emb = e4
        printt(f'{"e4":<15}: {e4.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)

        e_up3 = self.uda3(e4, e3)
        d3 = self.d3(e_up3)
        printt(f'{"e_up3":<15}: {e_up3.shape}', t=self.shape)
        printt(f'{"d3":<15}: {d3.shape}', t=self.shape)

        e_up2 = self.uda2(d3, e2)
        d2 = self.d2(e_up2)
        printt(f'{"e_up2":<15}: {e_up2.shape}', t=self.shape)
        printt(f'{"d2":<15}: {d2.shape}', t=self.shape)

        e_up1 = self.uda1(d2, e1)
        d1 = self.d1(e_up1)
        printt(f'{"e_up1":<15}: {e_up1.shape}', t=self.shape)
        printt(f'{"d1":<15}: {d1.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)

        p1 = torch.sigmoid(self.final_conv_1(d1))
        p2 = torch.sigmoid(F.interpolate(self.final_conv_2(d2), scale_factor=2, mode='trilinear', align_corners=True))
        p3 = torch.sigmoid(F.interpolate(self.final_conv_3(d3), scale_factor=4, mode='trilinear', align_corners=True))
        printt(f'{"p1":<15}: {p1.shape}', t=self.shape)
        printt(f'{"p2":<15}: {p2.shape}', t=self.shape)
        printt(f'{"p3":<15}: {p3.shape}', t=self.shape)

        return p1, p1, emb, p2, p3
