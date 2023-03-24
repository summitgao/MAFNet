# coding=utf-8
# Version:python 3.7

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn

from torchstat import stat

def Conv2d(input, output, k=3, s=1, dilation=1, activate='relu', group=1):
    model = []
    # dilation: dilation rate of dilated convolutions
    model.append(nn.Conv2d(input, output, kernel_size=k, stride=s, dilation=dilation, padding=((k - 1) * dilation // 2),
                           groups=group))
    if activate == 'relu':
        model.append(nn.LeakyReLU(0.2, inplace=True))
    if activate == 'tanh':
        model.append(nn.Tanh())

    return nn.Sequential(*model)


# initial layer
class DownSample(nn.Module):
    def __init__(self, scale, channel):
        super(DownSample, self).__init__()
        self.scale = scale
        self.downsample = nn.AvgPool2d(scale)
        self.conv = Conv2d(channel, channel * scale, activate=None)

    def forward(self, x):
        out = self.downsample(x)
        out = self.conv(out)
        return out


class Initial_Layer(nn.Module):
    # middle = [64, 128, 256]
    def __init__(self, input, middle):
        super(Initial_Layer, self).__init__()
        # Original scale
        self.conv1_1 = Conv2d(input, middle[0], activate=None)
        self.conv1_2 = Conv2d(middle[0], middle[0], activate=None)
        # Downsampling-Intermediate Scale
        self.down2_1 = DownSample(2, input)
        self.conv2_2 = Conv2d(input*2, middle[1], activate=None)
        # Downsampling-low scale
        self.down3_1 = DownSample(4, input)
        self.conv3_2 = Conv2d(input*4, middle[2], activate=None)

    def forward(self, x):
        in_x = x
        out1 = self.conv1_2(self.conv1_1(in_x))
        out2 = self.conv2_2(self.down2_1(in_x))
        out3 = self.conv3_2(self.down3_1(in_x))

        return out1, out2, out3


# coarse-fusion network
class UpSample(nn.Module):
    # Given input channel and scale, upsample the input x to shape_size
    def __init__(self, scale, channel):
        super(UpSample, self).__init__()
        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(channel, channel // scale, kernel_size=scale, stride=scale))

    def forward(self, x, shape_size):
        yh, yw = shape_size
        out = self.transpose(x)
        xh, xw = out.size()[-2:]
        h = yh - xh
        w = yw - xw
        pt = h // 2
        pb = h - pt
        pl = w // 2
        pr = w - w // 2
        out = F.pad(out, (pl, pr, pt, pb), mode='reflect')
        return out


class AdaIN(nn.Module):
    # input: number of channels in the noise estimation map, guidance information
    # channel: input x
    def __init__(self, down, input, channel):
        super(AdaIN, self).__init__()
        self.down = down
        if down:
            self.up = UpSample(down,input)
        self.con1 = Conv2d(input//down, channel, k=3, activate=None)
        self.con1_f = Conv2d(channel, channel, k=3, activate=None)
        self.con2 = Conv2d(channel, channel, activate=None)
        self.con3 = Conv2d(channel, channel, activate=None)

    def forward(self, de_map, guide_info):
        _, c, h, w = de_map.size()
        if self.down:
            guide_info = self.up(guide_info, de_map.size()[-2:] )
        # normlize
        mu = torch.mean(de_map.view(-1, c, h * w), dim=2)[:, :, None, None]
        sigma = torch.std(de_map.view(-1, c, h * w), dim=2)[:, :, None, None] + 10e-5
        de_map = (de_map - mu) / sigma

        guide_info = self.con1(guide_info)
        guide_info = self.con1_f(guide_info)
        gama = self.con2(guide_info)
        beta = self.con3(guide_info)   

        de_map = de_map * gama + beta

        return de_map


class AdaINBlock(nn.Module):
    def __init__(self, down, input, channel):
        super(AdaINBlock, self).__init__()
        self.con = Conv2d(channel, channel, activate=None)
        self.adain1 = AdaIN(down, input, channel)
        self.con1 = Conv2d(channel, channel, activate=None)

    def forward(self, demap, est_noise):
        x = self.con(demap)
        x = self.adain1(x, est_noise)
        x = self.con1(x)

        return demap + x


class Coarse_Fusion(nn.Module):
    def __init__(self, input, middle):
        super(Coarse_Fusion, self).__init__()

        self.adain1_1 = AdaINBlock(2, middle[1], middle[0])
        self.adain1_2 = AdaINBlock(2, middle[1], middle[0])

        self.adain2_1 = AdaINBlock(2, middle[2], middle[1])
        self.adain2_2 = AdaINBlock(2, middle[2], middle[1])

        self.con3_1 = Conv2d(middle[2], middle[2], activate=None)
        self.con3_2 = Conv2d(middle[2], middle[2], activate=None)

    def forward(self, x1, x2, x3):
        out1_1 = self.adain1_1(x1, x2)
        out2_1 = self.adain2_1(x2, x3)
        out3_1 = self.con3_1(x3)

        out1_2 = self.adain1_2(out1_1, out2_1)
        out2_2 = self.adain2_2(out2_1, out3_1)
        out3_2 = self.con3_2(out3_1)

        return out1_2, out2_2, out3_2


# fine-fusion network
# self-calibration
class SelfCalibration(nn.Module):
    def __init__(self, channel):
        super(SelfCalibration, self).__init__()
        self.conin = Conv2d(channel, channel, k=1, activate=None)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.sq = nn.Sequential(
            nn.Linear(16, 4, False),
            nn.LeakyReLU(True),
            nn.Linear(4, 16),
        )
        self.sa = nn.Sequential(
            Conv2d(channel, 16, k=1, activate=None),
        )
        self.act = nn.Sigmoid()
        self.conout = Conv2d(channel, channel, k=1, activate=None)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conin(x)
        avg = self.pooling(x).view(b * c, -1)
        cat = self.sq(avg).view(b, c, -1)
        sat = self.sa(x).view(b, -1, h * w)
        att = self.act(torch.bmm(cat, sat).view(x.size()))
        att = att * x
        out = self.conout(att)
        return out

# Concatenation and split
class CoAttention(nn.Module):
    def __init__(self, scale, channel, ratio):
        super(CoAttention, self).__init__()
        self.scale = scale

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.sq = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.LeakyReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        out = self.pooling(x).view(b, c)
        out = self.sq(out).view(b, c, 1, 1)
        out = out * x
        out = torch.sum(out.view(b, self.scale, c // self.scale, h, w), dim=1, keepdim=False)
        return out





class CASC(nn.Module):
    def __init__(self, scale, inchannels, ratio=4):
        super(CASC, self).__init__()
        self.CA = CoAttention(scale, inchannels * scale, ratio)
        self.SC = SelfCalibration(inchannels)

    def forward(self, x):
        fm = self.CA(x)
        out = self.SC(fm)
        return out + fm

class MSCASC(nn.Module):
    def __init__(self, n_scale, middle, ratio=4):
        super(MSCASC, self).__init__()
        self.n_scale = n_scale
        self.sample_dict = nn.ModuleDict()

        self.dm = nn.ModuleList([CASC(n_scale, middle[i], ratio) for i in range(n_scale)])

        for i in range(n_scale):
            for j in range(n_scale):
                if i < j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': DownSample(2 ** (j - i), middle[i])})
                if i > j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': UpSample(2 ** (i - j), middle[i])})

    def select_sample(self, x, shape_size, i, j):
        if i == j:
            return x
        else:
            if i > j:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x, shape_size)
            else:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x)

    def forward(self, x):
        res = []
        for i in range(self.n_scale):
            # 尺度归一
            fuse = [self.select_sample(x[j], x[i].size()[-2:], j, i) for j in range(self.n_scale)]
            # co-attention
            res.append(self.dm[i](torch.cat(fuse, dim=1)))

        return res


class Fine_Fusion(nn.Module):
    def __init__(self, middle, output, ratio=4):
        super(Fine_Fusion, self).__init__()

        self.msfda1 = MSCASC(3, middle, ratio)
        self.msfda2 = MSCASC(3, middle, ratio)
        self.fda = CASC(3, middle[0])
        self.up2_1 = UpSample(2, middle[1])
        self.up3_1 = UpSample(4, middle[2])
        self.out = Conv2d(middle[0], output, activate=None)

    def forward(self, x1, x2, x3):
        out1_1, out2_1, out3_1 = self.msfda1([x1, x2, x3])
        out1_2, out2_2, out3_2 = self.msfda2([out1_1, out2_1, out3_1])

        out2_3 = self.up2_1(out2_2, out1_2.size()[-2:])
        out3_3 = self.up3_1(out3_2, out1_2.size()[-2:])

        out1_3 = self.fda(torch.cat([out1_2, out2_3, out3_3], dim=1))

        out = self.out(out1_3)
        return out


# 网络主体部分
class MAFNet(nn.Module):
    def __init__(self, input, output, demiddle=[64, 128, 256]):
        super(MAFNet, self).__init__()
        self.initial_net = Initial_Layer(input, demiddle)
        self.coarse_net = Coarse_Fusion(input, demiddle)
        self.fine_net = Fine_Fusion(demiddle,output,ratio=4)

    def forward(self, x):
        in_x = x
        out1, out2, out3 = self.initial_net(in_x)
        # print('===================initial_net complete=======================')
        out1, out2, out3 = self.coarse_net(out1, out2, out3)
        # print('===================coarse_net complete=======================')
        out = self.fine_net(out1, out2, out3)
        # print('===================fine_net complete=======================')

        return out
