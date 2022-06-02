import torch
import torch.nn.functional as F
import torch.nn as nn


class GeneInteractionModel(nn.Module):

    def __init__(self, hidden_size, num_layers, num_features=24, dropout=0.1):
        super(GeneInteractionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, 12, bias=False)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class ConvNet(nn.Module):

    def __init__(self, hidden_size, num_layers, num_features=24, dropout=0.1):
        super(ConvNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(in_channels=128, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, 12, bias=False)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class BalancedMSELoss(nn.Module):

    def __init__(self, scale=True):
        super(BalancedMSELoss, self).__init__()

        self.factor = [1, 0.7, 0.6]

        self.mse = nn.MSELoss()
        if scale:
            self.mse = ScaledMSELoss()

    def forward(self, pred, actual):
        pred = pred.view(-1, 1)
        y = torch.log1p(actual[:, 0].view(-1, 1))

        l1 = self.mse(pred[actual[:, 1] == 1], y[actual[:, 1] == 1]) * self.factor[0]
        l2 = self.mse(pred[actual[:, 2] == 1], y[actual[:, 2] == 1]) * self.factor[1]
        l3 = self.mse(pred[actual[:, 3] == 1], y[actual[:, 3] == 1]) * self.factor[2]

        return l1 + l2 + l3


class MSLELoss(nn.Module):

    def __init__(self):
        super(MSLELoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        y = torch.log1p(actual.view(-1, 1))

        loss = self.mse(pred, y)

        return loss


class ScaledMSELoss(nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def forward(self, pred, y):
        mu = torch.minimum(torch.exp(6 * (y-3)) + 1, torch.ones_like(y) * 5) # SQRT-inverse

        return torch.mean(mu * (y-pred) ** 2)
