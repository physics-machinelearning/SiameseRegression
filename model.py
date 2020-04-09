import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstBlock(nn.Module):
    def __init__(self):
        super(FirstBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


class ConvRegression(nn.Module):
    def __init__(self):
        super(ConvRegression, self).__init__()
        self.conv = nn.Sequential(
            FirstBlock(),
            Block(),
            Block(),
        )
        self.linear = nn.Sequential(
            nn.Linear(6*6*64, 100),
        )
        self.final = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv(x.float())
        x = x.view(-1, 6*6*64)
        feature = self.linear(x)
        out = self.final(feature).view(-1)
        return feature, out


class SiameseRegression(nn.Module):
    def __init__(self):
        super(SiameseRegression, self).__init__()
        self.conv = nn.Sequential(
            FirstBlock(),
            Block(),
            Block(),
        )
        self.linear = nn.Sequential(
            nn.Linear(6*6*64, 100),
            # nn.Sigmoid(),
        )
        self.final = nn.Linear(100, 1)

    def forward(self, x0, x1):
        feature0 = self._forward_one(x0.float())
        feature1 = self._forward_one(x1.float())
        out0 = self.final(feature0).view(-1)
        out1 = self.final(feature1).view(-1)
        return feature0, feature1, out0, out1

    def _forward_one(self, x):
        x = self.conv(x)
        x = x.view(-1, 6*6*64)
        feature = self.linear(x)
        return feature


class SiameseRegressionLoss(nn.Module):
    def __init__(self):
        super(SiameseRegressionLoss, self).__init__()

    def forward(self, feature0, feature1, out0, out1, y0, y1):
        diff_feature = feature0 - feature1
        dist_feature = torch.sum(torch.pow(diff_feature, 2), 1)
        diff_y = y0 - y1
        dist_y = torch.pow(diff_y, 2)
        loss_f = torch.sum(torch.abs(dist_feature - dist_y))
        dif_y0 = out0 - y0
        dif_y1 = out1 - y1
        loss_r = torch.sum(torch.pow(dif_y0, 2), 0)
        loss_r += torch.sum(torch.pow(dif_y1, 2), 0)
        loss = loss_f + loss_r
        return loss


if __name__ == "__main__":
    net = SiameseRegression()
    print(net)