from misc.require_lib import *


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.unlinearity = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)
        self.w = nn.Linear(in_features=input_size, out_features=input_size, bias=True)
        self.v = nn.Linear(in_features=input_size, out_features=1, bias=False)
        self.v.weight.data = torch.ones_like(self.v.weight.data)

    def forward(self, input):
        x = input
        x = self.w(x)
        x = self.unlinearity(x)
        x = self.v(x)
        α = self.softmax(x)
        out = torch.mul(input, α)
        return out


class StatisticsPooling(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatisticsPooling, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class AttentionStatisticPooling(nn.Module):
    def __init__(self, input_size):
        super(AttentionStatisticPooling, self).__init__()
        self.unlinearity = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.w = nn.Linear(in_features=input_size, out_features=input_size, bias=True)
        self.v = nn.Linear(in_features=input_size, out_features=1, bias=False)
        self.v.weight.data = torch.ones_like(self.v.weight.data)
        self.VAR2STD_EPSILON = 1e-12

    def forward(self, input):
        x = input
        x = self.w(x)
        x = self.unlinearity(x)
        x = self.v(x)
        α = self.softmax(x)
        mean = torch.mul(input, α).sum(-2)

        variance = ((torch.mul(input, α)) * input).sum(-2) - mean * mean
        mask = (variance[:, :] <= self.VAR2STD_EPSILON).float()
        variance = (1.0 - mask) * variance + mask * self.VAR2STD_EPSILON
        stddev = torch.sqrt(variance)
        pooling_out = torch.cat((mean, stddev), dim=1)
        return pooling_out


class SelfAttenPooling(nn.Module):
    def __init__(self, input_size, T=1):
        super(SelfAttenPooling, self).__init__()
        self.unlinearity = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.w = nn.Linear(in_features=input_size, out_features=input_size)  # MLP网络
        self.v = nn.Linear(in_features=input_size, out_features=1, bias=False)
        # self.v.weight.data = torch.ones_like(self.v.weight.data)
        # self.T = T

    def forward(self, input):
        x = input
        x = self.w(x)
        x = self.unlinearity(x)
        x = self.v(x)
        α = self.softmax(x)
        out = torch.mul(input, α).sum(-2)
        return out
    

class SelfAttenPoolingMask(nn.Module):
    def __init__(self, input_size, T=1):
        super(SelfAttenPoolingMask, self).__init__()
        self.unlinearity = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.w = nn.Linear(in_features=input_size, out_features=input_size)  # MLP网络
        self.v = nn.Linear(in_features=input_size, out_features=1, bias=False)

    def forward(self, input, mask):
        x = input
        x = self.w(x)
        x = self.unlinearity(x)
        x = self.v(x)

        mask = mask.unsqueeze(2) # -> [B, T_max, 1]
        x.masked_fill_(mask == 0, -1e9)

        α = self.softmax(x)
        out = torch.mul(input, α).sum(-2)
        return out


class MultiHeadSelfAttenPooling(nn.Module):
    def __init__(self, input_size, head):
        super(MultiHeadSelfAttenPooling, self).__init__()
        self.head = head
        self.feat_dim = int(input_size / head)
        for h in range(self.head):
            self.add_module('ap_%d' % h, SelfAttenPooling(input_size=self.feat_dim))

    def forward(self, input):
        input = input[:, :, :(self.feat_dim * self.head)]
        b, t, f = input.size()
        multi_input = input.contiguous().view(b, t, -1, self.head)
        atten_out_list = []
        for h in range(self.head):
            atten_out_list.append(self.__getattr__('ap_%d' % h)(multi_input[:, :, :, h]))
        multi_out = torch.cat(atten_out_list, dim=1)
        return multi_out


class GlobalMultiHeadSelfAttenPooling(nn.Module):
    def __init__(self, input_size, head):
        super(GlobalMultiHeadSelfAttenPooling, self).__init__()
        self.head = head
        self.input_size = input_size
        for h in range(self.head):
            self.add_module('ap_%d' % h, SelfAttenPooling(input_size=self.input_size))

    def forward(self, input):
        atten_out_list = []
        for h in range(0, self.head):
            atten_out_list.append(self.__getattr__('ap_%d' % h)(input))
        multi_out = torch.cat(atten_out_list, dim=1)
        return multi_out


class MultiResolutionMultiHeadSelfAttenPooling(nn.Module):
    def __init__(self, input_size, head):
        super(MultiResolutionMultiHeadSelfAttenPooling, self).__init__()
        self.head = head
        self.input_size = input_size
        for h in range(self.head):
            T = max(1, int(h / 2) * 5)
            self.add_module('ap_%d' % h, SelfAttenPooling(input_size=self.input_size, T=T))

    def forward(self, input):
        atten_out_list = []
        for h in range(0, self.head):
            atten_out_list.append(self.__getattr__('ap_%d' % h)(input))
        multi_out = torch.cat(atten_out_list, dim=1)
        return multi_out


if __name__ == '__main__':
    pooling = MultiHeadSelfAttenPooling(input_size=96, head=2)
    input = torch.randn(10, 199, 96)
    out = pooling(input)
    print(out)

