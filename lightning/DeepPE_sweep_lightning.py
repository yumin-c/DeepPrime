import sys
import os
import math

import numpy as np
from numpy.random import shuffle
import scipy
import pandas as pd

from typing import Tuple, Optional, Any, Dict, List, Type

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
from torch.backends import cudnn

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

# Pytorch Lightning Module Add

from copy import deepcopy
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.loops.fit_loop import FitLoop


# Hyper Params

data_dir = 'data'
batch_size = 2048

T_0 = 12
T_mult = 1

n_epochs = 12
n_models = 1

hyperparameter_defaults = dict(
    c_1=128,
    c_2=64,
    c_3=32,
    c_4=16,
    d_1=128,
    d_2=64,
    d_3=32,
    hidden_size=128,
    num_layers=1,
    learning_rate=4e-3,
    weight_decay=1e-2,
)

wandb.init(config=hyperparameter_defaults, project="DeepSpCas9")
config = wandb.config


class GeneInteractionModel(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_layers: int,
            c_1: int,
            c_2: int,
            c_3: int,
            c_4: int,
            d_1: int,
            d_2: int,
            d_3: int,
            **_
    ):
        super(GeneInteractionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=c_1,
                      kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.GELU(),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=c_1, out_channels=c_2,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=c_2, out_channels=c_2,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=c_2, out_channels=c_3,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(c_3, hidden_size, num_layers,
                        batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, c_4, bias=False)

        self.d = nn.Sequential(
            nn.Linear(27, d_1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_1, d_2, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_2, d_3, bias=False)
        )

        self.head = nn.Sequential(
            # nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(c_4 + d_3, 1, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(g.transpose(1, 2))
        g = self.s(g[:, -1, :])
        x = self.d(x)
        out = self.head(torch.cat([g, x], dim=1))
        return out


class GeneInteractionRegressor(pl.LightningModule):

    def __init__(
            self,
            # Model Params
            hidden_size: int,
            num_layers: int,
            c_1: int,
            c_2: int,
            c_3: int,
            c_4: int,
            d_1: int,
            d_2: int,
            d_3: int,
            # Opt Params
            learning_rate: int,
            weight_decay: int,
            T_0 = 12,
            T_mult = 1,
            **_
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GeneInteractionModel(**self.hparams)
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=self.hparams.T_0,
            T_mult=self.hparams.T_mult,
            eta_min=self.hparams.learning_rate / 100)
        return [opt], [sch]

    def forward(self, g, x):
        return self.model(g, x)

    @staticmethod
    def cast_batch(batch):
        g, x, y = batch
        return g.permute((0, 3, 1, 2)), x, y.reshape(-1, 1)

    def compute_out_and_loss(self, g, x, y):
        out = self(g, x)
        loss = self.cross_entropy(out, y)
        return out, loss

    def training_step(self, batch, batch_idx):
        g, x, y = self.cast_batch_dtype(batch)
        out, loss = self.compute_out_and_loss(g, x, y)
        return {"loss": loss, "total": x.size(0), "progress_bar": {"train/loss": loss.item()}}

    def training_epoch_end(self, outputs):
        losses = np.array([o["loss"].item() for o in outputs])
        totals = np.array([o["total"] for o in outputs])
        losses = float(np.sum(losses * totals) / np.sum(totals))
        self.log("train/loss", losses)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        g, x, y = self.cast_batch_dtype(batch)
        out, loss = self.compute_out_and_loss(g, x, y)
        return {"val_loss": loss, "total": x.size(0),
                "pred": out.detach().cpu().numpy(), "y": y.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        losses = np.array([o["val_loss"].item() for o in outputs])
        totals = np.array([o["total"] for o in outputs])
        pred_ = np.concatenate([o["pred"] for o in outputs])
        y_ = np.concatenate([o["y"] for o in outputs])
        SPR = scipy.stats.spearmanr(pred_, y_).correlation
        losses = float(np.sum(losses * totals) / np.sum(totals))
        self.log("val/loss", losses)
        self.log("val/Spearman_score", SPR)
        return {"val_loss": losses, "Spearman_score": SPR}


class GeneFeatureDataset(Dataset):

    def __init__(
        self,
        gene: torch.Tensor = None,
        features: torch.Tensor = None,
        target: torch.Tensor = None,
        fold: int = None,
        mode: str = 'train',
        fold_list: np.ndarray = None,
    ):
        self.fold = fold
        self.mode = mode
        self.fold_list = fold_list

        if self.fold_list is not None:
            self.indices = self._select_fold()
            self.gene = gene[self.indices]
            self.features = features[self.indices]
            self.target = target[self.indices]
        else:
            self.gene = gene
            self.features = features
            self.target = target

    def _select_fold(self):
        selected_indices = []
        if self.mode == 'valid':  # SELECT A SINGLE GROUP
            for i in range(len(self.fold_list)):
                if self.fold_list[i] == self.fold:
                    selected_indices.append(i)
        elif self.mode == 'train':  # SELECT OTHERS
            for i in range(len(self.fold_list)):
                if self.fold_list[i] != self.fold:
                    selected_indices.append(i)
        else:  # FOR FINALIZING
            for i in range(len(self.fold_list)):
                selected_indices.append(i)

        return selected_indices

    def __len__(self) -> int:
        return len(self.gene)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene = self.gene[idx]
        features = self.features[idx]
        target = self.target[idx]

        return gene, features, target


class GeneFeatureDataModule(pl.LightningDataModule):

    g_train: ...
    x_train: ...
    y_train: ...
    g_test: ...
    x_test: ...
    y_test: ...
    num_folds: ...

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds

    def train_dataloader(self) -> DataLoader:
        train_set = GeneFeatureDataset(
            self.g_train, self.x_train, self.y_train, self.num_folds, 'train', self.train_fold)
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        valid_set = GeneFeatureDataset(
            self.g_train, self.x_train, self.y_train, self.num_folds, 'valid', self.train_fold)
        valid_loader = DataLoader(
            dataset=valid_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return valid_loader

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            num_workers: how many workers to use for loading data
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def preprocess_seq(data):
        print("Start preprocessing the sequence done 2d")
        length = 74

        DATA_X = np.zeros((len(data), 1, length, 4), dtype=float)
        print(np.shape(data), len(data), length)
        for l in tqdm(range(len(data))):
            for i in range(length):

                try:
                    data[l][i]
                except Exception:
                    print(data[l], i, length, len(data))

                if data[l][i] in "Aa":
                    DATA_X[l, 0, i, 0] = 1
                elif data[l][i] in "Cc":
                    DATA_X[l, 0, i, 1] = 1
                elif data[l][i] in "Gg":
                    DATA_X[l, 0, i, 2] = 1
                elif data[l][i] in "Tt":
                    DATA_X[l, 0, i, 3] = 1
                elif data[l][i] in "Xx":
                    pass
                else:
                    print("Non-ATGC character " + data[l])
                    print(i)
                    print(data[l][i])
                    sys.exit()

        print("Preprocessed the sequence")
        return DATA_X

    @classmethod
    def seq_concat(cls, data):
        wt = cls.preprocess_seq(data.WT74_On)
        ed = cls.preprocess_seq(data.Edited74_On)
        g = np.concatenate((wt, ed), axis=1)
        g = 2 * g - 1
        return g

    def setup(self, stage: Optional[str] = None):
        # LOAD DATA
        train_PECV = pd.read_csv(os.path.join(self.data_dir, 'DeepPrime_PECV__train_220214.csv'))
        test_PECV = pd.read_csv(os.path.join(self.data_dir, 'DeepPrime_PECV__test_220214.csv'))

        # PREPROCESS GENES

        if not os.path.isfile(os.path.join(self.data_dir, 'g_train.npy')):
            g_train = self.seq_concat(train_PECV)
            np.save(os.path.join(self.data_dir, 'g_train.npy'), g_train)
        else:
            g_train = np.load(os.path.join(self.data_dir, 'g_train.npy'))

        if not os.path.isfile(os.path.join(self.data_dir, 'g_test.npy')):
            g_test = self.seq_concat(test_PECV)
            np.save(os.path.join(self.data_dir, 'g_test.npy'), g_test)
        else:
            g_test = np.load(os.path.join(self.data_dir, 'g_test.npy'))

        # FEATURE SELECTION
        self.train_fold = train_PECV.Fold
        train_features = train_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                            'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                            'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                            'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
        train_target = train_PECV.Measured_PE_efficiency
        test_features = test_PECV.loc[:, ['PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos', 'Edit_len', 'RHA_len', 'type_sub',
                                          'type_ins', 'type_del', 'Tm1', 'Tm2', 'Tm2new', 'Tm3', 'Tm4', 'TmD',
                                          'nGCcnt1', 'nGCcnt2', 'nGCcnt3', 'fGCcont1', 'fGCcont2', 'fGCcont3',
                                          'MFE1', 'MFE2', 'MFE3', 'MFE4', 'MFE5', 'DeepSpCas9_score']]
        test_target = test_PECV.Measured_PE_efficiency

        # NORMALIZATION

        x_train = (train_features - train_features.mean()) / train_features.std()
        y_train = (train_target - train_target.mean()) / train_target.std()
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        x_test = (test_features - train_features.mean()) / train_features.std()
        y_test = (test_target - train_target.mean()) / train_target.std()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()

        self.g_train = torch.tensor(g_train, dtype=torch.float32)
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

        self.g_test = torch.tensor(g_test, dtype=torch.float32)
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)


class EnsembleVotingModel(pl.LightningModule):
    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.nll_loss(logits, batch[1])
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)


class KFoldLoop(Loop):

    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        # checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        # voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        # voting_model.trainer = self.trainer
        # # This requires to connect the new model and move it the right device.
        # self.trainer.strategy.connect(voting_model)
        # self.trainer.strategy.model_to_device()
        # self.trainer.test_loop.run()
        pass

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]


# TRAINING & VALIDATION

if __name__ == "__main__":

    for m in range(n_models):

        random_seed = m
        pl.seed_everything(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = GeneInteractionRegressor(**config)
        datamodule = GeneFeatureDataModule(
            data_dir='data', batch_size=batch_size)
        trainer = pl.Trainer(
            max_epochs=n_epochs,
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="ddp",
            logger=WandbLogger()
        )
        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = KFoldLoop(5, export_path="./")
        trainer.fit_loop.connect(internal_fit_loop)  # type: ignore
        trainer.fit(model, datamodule)
