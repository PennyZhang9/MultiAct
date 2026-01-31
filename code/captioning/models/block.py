import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), bn_tag=True, ul_tag=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding,
                               bias=True, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.unlinearity = nn.ELU()
        # self.unlinearity = nn.Threshold(-1, -1, inplace=True)
        self.bn_tag = bn_tag
        self.ul_tag = ul_tag

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # 对于BN2d，weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.bn_tag:
            x = self.bn(x)
        if self.ul_tag:
            x = self.unlinearity(x)
        return x