from misc.require_lib import *


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, labels):
        loss = self.ce(input, labels)
        return loss


class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()
        self.criteria = CE()

    def forward(self,input, label, scale=10.0, margin=0.30):
        cos_theta = input
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)
        final_gt = gt - margin
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= scale
        loss = self.criteria(cos_theta, label)
        return loss


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, label):
        loss = self.mse(input, label)
        return loss


class CircleLoss(nn.Module):
    def __init__(self, m=0.35, s=256):
        super().__init__()
        self.s, self.m = s, m
        self.criteria = CE()

    def forward(self, input, label):
        cosine = input
        alpha_p = F.relu(1 + self.m - cosine).detach()
        margin_p = 1 - self.m
        alpha_n = F.relu(cosine + self.m).detach()
        margin_n = self.m

        sp_y = alpha_p * (cosine - margin_p)
        sp_j = alpha_n * (cosine - margin_n)

        one_hot = torch.zeros(cosine.size()).to(label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * sp_y + ((1.0 - one_hot) * sp_j)
        output *= self.s
        loss = self.criteria(output, label)
        return loss


if __name__ == '__main__':
    input = torch.randn(10, 512)
    label = torch.randn(10, 154)

