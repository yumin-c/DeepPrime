# Losses

import torch
import torch.nn as nn


class BalancedMSELoss(nn.Module):

    def __init__(self, scale=True):
        super(BalancedMSELoss, self).__init__()

        self.factor = [1, 0.7, 0.6]

        self.mse = nn.MSELoss()
        if scale:
            self.mse = ScaledMSELoss()
            print("Applying ScaledMSELoss")
        else:
            print("Applying MSELoss without scaling")

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))

        l1 = self.mse(pred[actual[:, 1] == 1], y[actual[:, 1] == 1]) * self.factor[0]
        l2 = self.mse(pred[actual[:, 2] == 1], y[actual[:, 2] == 1]) * self.factor[1]
        l3 = self.mse(pred[actual[:, 3] == 1], y[actual[:, 3] == 1]) * self.factor[2]

        return l1 + l2 + l3


class ScaledMSELoss(nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, y):
        mu = torch.minimum(torch.exp(6 * (y-3)) + 1, torch.ones_like(y) * 5) # Reciprocal of the square root of the original dataset distribution.

        return torch.mean(mu * (y-pred) ** 2)


class OffTargetLoss(nn.Module):

    def __init__(self, dataset=0):
        super(OffTargetLoss, self).__init__()

        self.dataset = dataset
        self.factor = [0.25, 1] if dataset == 0 else [1, 1]
        self.mse = nn.MSELoss(reduction='sum')

    def _scale_mseloss(self, pred, y):
        mu = torch.minimum(0.00003*(y**3-100*y**2+2700*y)+0.15, torch.ones_like(y)) # Reciprocal of the square root of the off-target dataset distribution (under development, never been used).

        return torch.sum(mu * (y-pred)**2)

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))
        idx = actual[:, -1] == 7

        if self.dataset==0:
            l1 = self.mse(pred[idx], y[idx]) * self.factor[0]
            l2 = self.mse(pred[~idx], y[~idx]) * self.factor[1]
        # elif self.dataset==1:
        #     l1 = self._scale_mseloss(pred[idx], y[idx]) * self.factor[0]
        #     l2 = self._scale_mseloss(pred[~idx], y[~idx]) * self.factor[1]
            
        loss = (l1 + l2) / pred.size(0)

        return loss