import torch
import torch.nn as nn


class Conv_block(nn.Module):

    def __init__(self, dim_in, dim_out, down=1, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.down = down
        stride = 2 if self.down==2 else 1
        self.conv = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim_out),
            self.act,
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(dim_out),
            self.act)

        self.eq_channel = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1)
        self.eq_size_up = nn.Upsample(scale_factor=1/down, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        if self.down==0.5:
            x = self.eq_size_up(x)

        return x


class UNet(nn.Module):

    def __init__(self, dim_in=1, dim_out=1, dim_base=32):
        super().__init__()

        self.encoder_1 = nn.Sequential(
            nn.Conv3d(dim_in, dim_base*1, kernel_size=7, stride=1, padding=7//2),
            nn.BatchNorm3d(dim_base*1),
            nn.ReLU(),
            Conv_block(dim_base*1, dim_base*1, down=2))

        self.encoder_2 = nn.Sequential(
            Conv_block(dim_base*1, dim_base*2, down=2))

        self.encoder_3 = nn.Sequential(
            Conv_block(dim_base*2, dim_base*4, down=2))

        self.encoder_4 = nn.Sequential(
            Conv_block(dim_base*4, dim_base*8, down=2))

        self.decoder_4 = Conv_block(dim_base*8, dim_base*4, down=0.5)
        self.decoder_3 = Conv_block(dim_base*8, dim_base*4, down=0.5)
        self.decoder_2 = Conv_block(dim_base*6, dim_base*3, down=0.5)
        self.decoder_1 = Conv_block(dim_base*4, dim_base*1, down=0.5)

        self.last = nn.Sequential(
            nn.Conv3d(dim_base, dim_out, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)

        d4 = self.decoder_4(e4)
        d3 = self.decoder_3(torch.concat([e3, d4], axis=1))
        d2 = self.decoder_2(torch.concat([e2, d3], axis=1))
        d1 = self.decoder_1(torch.concat([e1, d2], axis=1))

        res = self.last(d1)

        return res, res, e4
