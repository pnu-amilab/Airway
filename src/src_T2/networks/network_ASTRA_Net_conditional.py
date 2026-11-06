import torch
import torch.nn as nn
import torch.nn.functional as F

from src_T1.networks.modules import CorrModule, Droplayer, EGA_module, SE_block, SSE_block


class ASTRA_Net_Conditional_T2(nn.Module):

    def __init__(self, dim_in=15, dim_out=20, dim_base=16, class_dim=64, corr_vol=False,
                 ega=True, ega_learnable=True, ega_recalibration=True, lobe_bias_strength=0.75):
        super().__init__()
        self.corr_vol = corr_vol
        self.ega = ega
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_base = dim_base
        self.dim_context = dim_in-1
        self.lobe_bias_strength = lobe_bias_strength

        self.e_conv_1 = SE_block(dim_in, dim_base)
        self.e_conv_2 = SE_block(dim_base, dim_base*2)
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
        self.d_conv_9 = SSE_block(dim_base*16+q, dim_base*8)
        self.d_conv_8 = SSE_block(dim_base*8, dim_base*8)
        self.d_conv_6 = SSE_block(dim_base*16+q, dim_base*8)
        self.d_conv_5 = SSE_block(dim_base*8, dim_base*4)
        self.d_conv_3 = SE_block(dim_base*8+q, dim_base*4)
        self.d_conv_2 = SE_block(dim_base*4, dim_base*2)

        class_dim_encoder = class_dim//2
        self.class_last_1 = nn.Sequential(
            Droplayer(18+6+self.dim_context, 0.2),
            nn.Conv3d(18+6+self.dim_context, class_dim_encoder, kernel_size=1, bias=False),
            nn.BatchNorm3d(class_dim_encoder),
            nn.SiLU(),
            nn.Conv3d(class_dim_encoder, dim_out, kernel_size=1, bias=True))
        self.class_last_2 = nn.Sequential(
            Droplayer(12+6+dim_base*2+self.dim_context, 0.2),
            nn.Conv3d(12+6+dim_base*2+self.dim_context, class_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(class_dim),
            nn.SiLU(),
            nn.Conv3d(class_dim, dim_out, kernel_size=1, bias=True))
        self.global_class_bias = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim_base*8, dim_out, kernel_size=1, bias=True))

        lobe_compatibility = torch.zeros(5, dim_out, dtype=torch.float32)
        for class_idx in [0, 1, 2, 3]: lobe_compatibility[0, class_idx] = 1
        for class_idx in [4, 5, 6, 7, 19]: lobe_compatibility[1, class_idx] = 1
        for class_idx in [8, 9, 10]: lobe_compatibility[2, class_idx] = 1
        for class_idx in [11, 12]: lobe_compatibility[3, class_idx] = 1
        for class_idx in [13, 14, 15, 16, 17]: lobe_compatibility[4, class_idx] = 1
        self.register_buffer('lobe_compatibility', lobe_compatibility)

        self.expand = nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=False)
        self.reduce = nn.MaxPool3d(2, 2)
        if self.corr_vol: self.corr_module = CorrModule()
        if self.ega: self.ega_module = EGA_module(learnable=ega_learnable, recalib=ega_recalibration)
        nn.init.zeros_(self.class_last_1[-1].bias)
        nn.init.zeros_(self.class_last_2[-1].bias)
        nn.init.zeros_(self.global_class_bias[-1].weight)
        nn.init.zeros_(self.global_class_bias[-1].bias)

    def _get_lobe_bias(self, x):
        if self.lobe_bias_strength == 0: return 0
        lobe_one_hot = x[:, 3:8]
        bias = torch.einsum('blxyz,lc->bcxyz', lobe_one_hot, self.lobe_compatibility)

        return bias*self.lobe_bias_strength

    def forward(self, x):
        s1, s2, s3 = x.shape[-3:]
        geometry_probability = x[:, [1]]
        context = x[:, 1:]

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
        global_class_bias = self.global_class_bias(b3)
        lobe_class_bias = self._get_lobe_bias(x)

        takes = [et1, et2, et3, et4, et5, et6, et7, et8, et9, bt1, bt2, bt3]
        takes = [F.interpolate(t, size=(s1, s2, s3), mode='trilinear', align_corners=False) for t in takes]
        feature_1 = torch.concat(takes, dim=1)
        c1_logits = self.class_last_1(torch.concat([feature_1, context], dim=1))+global_class_bias+lobe_class_bias

        if self.corr_vol:
            c3 = self.corr_module(e3, e3)
            c6 = self.corr_module(e3, e6)
            c9 = self.corr_module(e3, e9)
            e3 = torch.concat([e3, c3], dim=1)
            e6 = torch.concat([e6, c6], dim=1)
            e9 = torch.concat([e9, c9], dim=1)

        if self.ega:
            ea_att_score = self.ega_module(geometry_probability)
            e3 = e3*ea_att_score
            e6 = e6*self.reduce(ea_att_score)
            e9 = e9*self.reduce(self.reduce(ea_att_score))

        d9, dt9 = self.d_conv_9(torch.concat([self.expand(b3), e9], dim=1))
        d8, dt8 = self.d_conv_8(d9)
        d6, dt6 = self.d_conv_6(torch.concat([self.expand(d8), e6], dim=1))
        d5, dt5 = self.d_conv_5(d6)
        d3, dt3 = self.d_conv_3(torch.concat([self.expand(d5), e3], dim=1))
        d2, dt2 = self.d_conv_2(d3)

        decoder_takes = [bt1, bt2, bt3, dt9, dt8, dt6, dt5, dt3, dt2]
        decoder_takes = [F.interpolate(t, size=(s1, s2, s3), mode='trilinear', align_corners=False) for t in decoder_takes]
        feature_2 = torch.concat(decoder_takes, dim=1)
        c2_logits = self.class_last_2(torch.concat([feature_2, d2, context], dim=1))+global_class_bias+lobe_class_bias

        return c1_logits, c2_logits, b3
