import torch
import torch.nn as nn
import torch.nn.functional as F


class InBlock(nn.Module):
    def __init__(self):
        super(InBlock, self).__init__()
        self.c1 = nn.Conv2d(24, 256, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, 24, 8, 8)
        x = F.relu(self.b1(self.c1(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ResBlock, self).__init__()
        self.c1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(256)
        self.c2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        res = x
        out = F.elu(self.b1(self.c1(x)))
        out = self.dropout(out)
        out = self.b2(self.c2(out))
        out += res
        out = F.elu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(OutBlock, self).__init__()
        self.c1 = nn.Conv2d(256, 1, 1)
        self.b1 = nn.BatchNorm2d(1)
        self.f1 = nn.Linear(8 * 8, 64)
        self.f2 = nn.Linear(64, 1)

        self.c2 = nn.Conv2d(256, 128, 1)
        self.b2 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.f3 = nn.Linear(8 * 8 * 128, 8 * 8 * 73)
        self.log = nn.LogSoftmax(1)

    def forward(self, x):
        v = F.elu(self.b1(self.c1(x)))
        v = v.view(-1, 8 * 8)
        v = F.elu(self.f1(v))
        v = self.dropout(v)
        v = torch.tanh(self.f2(v))

        p = F.elu(self.b2(self.c2(x)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.dropout(p)
        p = self.f3(p)
        p = self.log(p).exp()

        return p, v


class Net(nn.Module):
    def __init__(self, res_blocks=20):
        super(Net, self).__init__()

        self.res_blocks = res_blocks
        self.inblock = InBlock()

        for i in range(self.res_blocks):
            setattr(self, f'res{i}', ResBlock())

        self.outblock = OutBlock()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.inblock(x)

        for i in range(self.res_blocks):
            x = getattr(self, f'res{i}')(x)

        p, v = self.outblock(x)
        return p, v
