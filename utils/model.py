# Candidate models used for DeepPrime development.

# class GeneInteractionModel: final DeepPrime architecture, which is composed of CNN+GRU / Linear layers.
# class GRUOnly: model without convolutional layers, which is composed of GRU / Linear layers.
# class LSTMOnly: variant of GRUOnly model, where GRU module is switched into LSTM.
# class ConvTransformer: a model which uses CNN+Transformer for sequence processing / Linear layers for biofeature processing.
# class Transformer: a model which uses Transformer for sequence processing / Linear layers for biofeature processing.

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class GeneInteractionModel(nn.Module):

    def __init__(self, hidden_size=128, num_layers=1, num_features=24, dropout=0.1):
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


class GRUOnly(nn.Module):

    def __init__(self, hidden_size=128, num_layers=1, num_features=24, dropout=0.1):
        super(GRUOnly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.r = nn.GRU(8, hidden_size, num_layers, batch_first=True, bidirectional=True)

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
        g = torch.transpose(g.reshape(-1, 8, 74), 1, 2)
        g, _ = self.r(g)
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class LSTMOnly(nn.Module):

    def __init__(self, hidden_size=128, num_layers=1, num_features=24, dropout=0.1):
        super(LSTMOnly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.r = nn.LSTM(8, hidden_size, num_layers, batch_first=True, bidirectional=True)

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
        g = torch.transpose(g.reshape(-1, 8, 74), 1, 2)
        g, _ = self.r(g)
        g = self.s(g[:, -1, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)
        

class ConvTransformer(nn.Module):

    def __init__(self, hidden_size=128, nhead=4, dim_feedforward=256, layer_norm_eps=1e-5, num_encoder_layers=2, num_features=24, dropout=0.1):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size

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

            nn.Conv1d(in_channels=108, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.register_parameter(name='reg_token', param=nn.Parameter(torch.randn(size=(1, 1, hidden_size))))
        self.register_parameter(name='pos_embedding', param=nn.Parameter(torch.randn(9+1, 1, hidden_size)))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=dim_feedforward, nhead=nhead, dropout=0.15, activation="gelu")
        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + 128),
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 128, 1, bias=True),
        )

    def forward(self, g, x):
        n_samples = x.size(0)
        repeat_token = self.reg_token.repeat(1, n_samples, 1)

        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g) # B F S

        g = torch.cat((repeat_token, torch.permute(g, (2, 0, 1))), dim=0) # S B F
        g += self.pos_embedding

        g = self.encoder(g)
        g = torch.squeeze(g[0, :, :])

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)


class Transformer(nn.Module):

    def __init__(self, hidden_size=128, nhead=4, dim_feedforward=256, layer_norm_eps=1e-5, num_encoder_layers=2, num_features=24, dropout=0.1, use_trainable_pe=False, use_reg_token=False):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Linear(4, hidden_size)

        self.use_reg_token = use_reg_token
        self.register_parameter(name='reg_token', param=nn.Parameter(torch.randn(size=(1, 1, hidden_size))))

        seq_len = 148
        if use_reg_token:
            seq_len += 1

        if use_trainable_pe:
            self.register_parameter(name='pos_embedding', param=nn.Parameter(torch.randn(seq_len, hidden_size)))
        else:
            pe = torch.zeros(seq_len, hidden_size)
            for pos in range(seq_len):
                for i in range(0, hidden_size, 2):
                    pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/hidden_size)))
                    pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/hidden_size)))
                    
            self.register_buffer('pos_embedding', pe)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=dim_feedforward, nhead=nhead, dropout=0.1, activation="gelu")
        encoder_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + 128),
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 128, 1, bias=True),
        )

    def forward(self, g, x):
        g = g.reshape(-1, 148, 4)
        g = (g + 1) / 2

        g = self.embedding(g) # B S F

        if self.use_reg_token:
            repeat_token = self.reg_token.repeat(x.size(0), 1, 1)
            g = torch.cat((repeat_token, g), dim=1) # B S F

        g += self.pos_embedding # B S F
        g = torch.permute(g, (1, 0, 2)) # S B F

        g = self.encoder(g)
        
        if self.use_reg_token:
            g = torch.squeeze(g[0, :, :])
        else:
            g = torch.mean(g, dim=0)

        x = self.d(x)

        out = self.head(torch.cat((g, x), dim=1))

        return F.softplus(out)