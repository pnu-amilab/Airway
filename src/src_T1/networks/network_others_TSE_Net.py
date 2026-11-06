'''
TSE-Net
"Airway Segmentation based on Topological Structure Enhancement using Multi-Task Learning"
International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def printt(x, t=False):
    if t: print(x)

# Basic 3D Conv Block


def conv_block(dim_in, dim_out, k=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(dim_in, dim_out, k, stride, padding),
        nn.BatchNorm3d(dim_out),
        nn.ReLU())

#  PE (Project & Excite)


class PE_module(nn.Module):

    def __init__(self, dim_base):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(dim_base, dim_base//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(dim_base//2, dim_base, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, d, h, w = x.size()
        proj_d = F.adaptive_avg_pool3d(x, (d, 1, 1))
        proj_h = F.adaptive_avg_pool3d(x, (1, h, 1))
        proj_w = F.adaptive_avg_pool3d(x, (1, 1, w))
        att = self.attention(proj_d + proj_h + proj_w)

        return x * att

# TEAM (Topology Enhanced Attention Module)


class TEAM(nn.Module):

    def __init__(self, dim_m, dim_M):
        super().__init__()
        self.channel_eq = nn.Conv3d(dim_M, dim_m, kernel_size=1, bias=False)
        self.spatial_eq = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True)
        self.pe_attention = PE_module(dim_m*2)

        self.channel_att = nn.Sequential(
            nn.Conv3d(dim_m*2, dim_m, kernel_size=1, bias=False),
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Sigmoid())

    def forward(self, x_m, x_M):
        x_M = self.channel_eq(self.spatial_eq(x_M))

        fusion = torch.cat([x_m, x_M], dim=1)
        fusion = self.pe_attention(fusion)
        att_map = self.channel_att(fusion)
        x_m = x_m*att_map

        res = x_m+x_M

        return res

# TEIM (Topology Embedded Interaction Module)


class TEIM(nn.Module):

    def __init__(self, dim_t, dim_s, dim_c):
        super().__init__()
        self.conv_ts = nn.Sequential(
            nn.Conv3d(dim_t+dim_s, dim_t+dim_s, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(dim_t+dim_s, dim_t+dim_s, kernel_size=3, padding=1, bias=False))
        self.conv_tsc = nn.Conv3d(dim_c+dim_c, dim_c+dim_c, kernel_size=3, padding=1, bias=False)
        self.conv_all = nn.Sequential(
            nn.Conv3d(dim_t+dim_s*2, dim_t+dim_s*2, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(dim_t+dim_s*2, dim_t+dim_s*1, kernel_size=1, bias=False))
        self.conv_c = nn.Conv3d(dim_c*2, dim_c*2, kernel_size=3, padding=1, bias=False)

        self.channel_eq_s2c = nn.Conv3d(dim_t+dim_s, dim_c, kernel_size=1, bias=False)
        self.channel_eq_c2s = nn.Conv3d(dim_c+dim_c, dim_s, kernel_size=1, bias=False)

    def forward(self, t, s, c):
        concat_ts = torch.concat([t, s], axis=1)
        concat_ts = self.conv_ts(concat_ts)

        concat_tsc = torch.concat([self.channel_eq_s2c(concat_ts), c], axis=1)
        concat_tsc = self.conv_tsc(concat_tsc)

        concat_all = torch.concat([self.channel_eq_c2s(concat_tsc), concat_ts], axis=1)
        concat_all = self.conv_all(concat_all)

        s_final = concat_ts+concat_all
        c_final = self.conv_c(concat_tsc)

        return s_final, c_final


class TSE_Net(nn.Module):

    def __init__(self, dim_in=1, dim_base=32, shape=False):
        super().__init__()
        self.shape = shape
        self.encoder_1 = nn.Sequential(
            conv_block(dim_in    , dim_base*1),
            conv_block(dim_base*1, dim_base*1),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.encoder_2 = nn.Sequential(
            conv_block(dim_base*1, dim_base*2),
            conv_block(dim_base*2, dim_base*2),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.encoder_3 = nn.Sequential(
            conv_block(dim_base*2, dim_base*4),
            conv_block(dim_base*4, dim_base*4),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.encoder_4 = nn.Sequential(
            conv_block(dim_base*4, dim_base*8),
            conv_block(dim_base*8, dim_base*8),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.bridge = nn.Sequential(
            conv_block(dim_base*8, dim_base*8),
            conv_block(dim_base*8, dim_base*8))

        self.team_2 = TEAM(dim_base*2, dim_base*1)
        self.team_3 = TEAM(dim_base*4, dim_base*2)
        self.team_4 = TEAM(dim_base*8, dim_base*4)
        self.upsampler = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.decoder_4 = nn.Sequential(
            conv_block(dim_base*8, dim_base*8),
            conv_block(dim_base*8, dim_base*4))
        self.decoder_3 = nn.Sequential(
            conv_block(dim_base*12, dim_base*8),
            conv_block(dim_base*8, dim_base*4))
        self.decoder_2 = nn.Sequential(
            conv_block(dim_base*8, dim_base*4),
            conv_block(dim_base*4, dim_base*2))
        self.decoder_1 = nn.Sequential(
            conv_block(dim_base*4, dim_base*2),
            conv_block(dim_base*2, dim_base*1))

        self.cecoder_4 = nn.Sequential(
            conv_block(dim_base*8, dim_base*8),
            conv_block(dim_base*8, dim_base*4))
        self.cecoder_3 = nn.Sequential(
            conv_block(dim_base*8, dim_base*4),
            conv_block(dim_base*4, dim_base*4))
        self.cecoder_2 = nn.Sequential(
            conv_block(dim_base*8, dim_base*4),
            conv_block(dim_base*4, dim_base*2))
        self.cecoder_1 = nn.Sequential(
            conv_block(dim_base*4, dim_base*2),
            conv_block(dim_base*2, dim_base*1))

        self.teim_4 = TEIM(dim_base*8, dim_base*4, dim_base*4)
        self.teim_3 = TEIM(dim_base*4, dim_base*4, dim_base*4)
        self.teim_2 = TEIM(dim_base*2, dim_base*2, dim_base*2)
        self.teim_1 = TEIM(dim_base*1, dim_base*1, dim_base*1)

        self.last_s = nn.Sequential(
            nn.Conv3d(dim_base*2, 1, kernel_size=1, bias=False),
            nn.Sigmoid())
        self.last_c = nn.Sequential(
            nn.Conv3d(dim_base*2, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        printt(f'{"x":<15}: {x.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)
        b  = self.bridge(e4)
        emb = b
        printt(f'{"e1":<15}: {e1.shape}', t=self.shape)
        printt(f'{"e2":<15}: {e2.shape}', t=self.shape)
        printt(f'{"e3":<15}: {e3.shape}', t=self.shape)
        printt(f'{"e4":<15}: {e4.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)
        printt(f'{"b":<15}: {b.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)
        t1 = e1
        t2 = self.team_2(e2, t1)
        t3 = self.team_3(e3, t2)
        t4 = self.team_4(e4, t3)
        printt(f'{"t4":<15}: {t4.shape}', t=self.shape)
        printt(f'{"t3":<15}: {t3.shape}', t=self.shape)
        printt(f'{"t2":<15}: {t2.shape}', t=self.shape)
        printt(f'{"t1":<15}: {t1.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)
        s4 = self.decoder_4(b)
        c4 = self.cecoder_4(b)
        s4, c4 = self.teim_4(t4, s4, c4)
        s4, c4 = self.upsampler(s4), self.upsampler(c4)
        printt(f'{"t4, s4, c4":<15}: {t4.shape} {s4.shape} {c4.shape}', t=self.shape)
        s3 = self.decoder_3(s4)
        c3 = self.cecoder_3(c4)
        s3, c3 = self.teim_3(t3, s3, c3)
        s3, c3 = self.upsampler(s3), self.upsampler(c3)
        printt(f'{"t3, s3, c3":<15}: {t3.shape} {s3.shape} {c3.shape}', t=self.shape)
        s2 = self.decoder_2(s3)
        c2 = self.cecoder_2(c3)
        s2, c2 = self.teim_2(t2, s2, c2)
        s2, c2 = self.upsampler(s2), self.upsampler(c2)
        printt(f'{"t2, s2, c2":<15}: {t2.shape} {s2.shape} {c2.shape}', t=self.shape)
        s1 = self.decoder_1(s2)
        c1 = self.cecoder_1(c2)
        s1, c1 = self.teim_1(t1, s1, c1)
        s1, c1 = self.upsampler(s1), self.upsampler(c1)
        printt(f'{"t1, s1, c1":<15}: {t1.shape} {s1.shape} {c1.shape}', t=self.shape)
        printt(f'{"="*60}', t=self.shape)
        s_final = self.last_s(s1)
        c_final = self.last_c(c1)
        printt(f'{"s_final":<15}: {s_final.shape}', t=self.shape)
        printt(f'{"c_final":<15}: {c_final.shape}', t=self.shape)

        return s_final, s_final, emb, c_final
