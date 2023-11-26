"""
PCA implementation practice with Pytorch
"""

import copy
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset


class TensorIndicesDataset(TensorDataset):
    """Extend TensorDataset to return index along with sample"""

    def __getitem__(self, index):
        dtuple = super(TensorIndicesDataset, self).__getitem__(index)
        return tuple(list(dtuple) + [index])


class _PCAModule(nn.Module):
    def __init__(
        self,
        X_shape: Tuple[int, int],
        K: int = 2,
    ):
        super(_PCAModule, self).__init__()
        # sizing
        self.N, self.P = X_shape
        self.K = K
        # model components
        self.z = Parameter(torch.randn(self.N, self.K))
        self.W = Parameter(torch.randn(self.P, self.K))
        self.mu = Parameter(torch.randn(self.P))

    def forward(self, X, indices):
        return self.z[indices, :] @ self.W.T + self.mu


class PytorchPCA:
    """Trying to achieve a simple PCA implementation using generic Pytorch optimizers"""

    def __init__(
        self,
        n_components: int = 2,
    ):
        self.n_components = n_components
        self.history = {"loss": [], "model": []}
        # CPU or GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

    def _init_model(self, X: np.ndarray, lr=1e-3):
        """Create torch model given starting data"""
        self.N, self.P = X.shape
        # pytorch objects
        # data
        self.X_ds = TensorIndicesDataset(torch.tensor(X))
        self.X_train_dl = DataLoader(
            self.X_ds, batch_size=min(X.shape[0], 100), shuffle=True
        )
        # simply performing eval across full dataset
        self.X_valid_dl = DataLoader(
            self.X_ds, batch_size=min(X.shape[0], 100), shuffle=False
        )

        # model
        self.model = _PCAModule(X.shape, self.n_components).to(self.device)

        # loss and optimizer
        self._data_loss = nn.MSELoss()
        self._alpha = 1e-2  # regularization coefficient
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _regular_loss(self):
        """Compute regularization loss on parameters

        includes regularization coefficient
        """
        return self._alpha * torch.norm(self.model.z, p=2)

    def _compute_loss(self, target_pred, target):
        """Combine data loss and regularization loss

        Parameters
        ----------
        target_pred
            model predictions
        target
            target values

        Returns
        -------
        total loss
        """
        return self._data_loss(target_pred, target) + self._regular_loss()

    def _train(self, n_epoch=10, save_epoch=1):
        """Main training loop.

        Assumes training and validation data loaders, model, loss, and optimizer
        have already been initialized and configured.

        Parameters
        ----------
        n_epoch
            number of training epochs
        save_epoch
            every `save_epoch` validation loss and model state is recorded in `history`
        """
        total_start = time.time()

        for epoch in range(n_epoch):
            # training
            train_start = time.time()
            self.model.train()
            for batch, (Xb, ib) in enumerate(self.X_train_dl):
                # transfer data if necessary
                Xb, ib = Xb.to(self.device), ib.to(self.device)

                # calculate loss
                X_recon = self.model(Xb, ib)
                loss = self._compute_loss(X_recon, Xb)

                # backprop loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # batch update
                if batch % 100 == 0:
                    print(
                        f"    batch_loss: {loss.item():>7f} "
                        f"item: [{(batch + 1) * len(Xb):>7d}/{len(self.X_train_dl.dataset):>7d}]"
                    )
            train_end = time.time()
            train_time = train_end - train_start

            # evaluation
            eval_start = time.time()
            self.model.eval()
            vloss = 0
            with torch.no_grad():
                for Xv, iv in self.X_valid_dl:
                    X_recon = self.model(Xv, iv)
                    vloss += self._compute_loss(X_recon, Xv).item() * len(Xv)
            vloss /= len(self.X_valid_dl.dataset)

            eval_end = time.time()
            eval_time = eval_end - eval_start

            # record history
            if epoch % save_epoch == 0:
                self.history["loss"].append(vloss)
                self.history["model"].append(copy.deepcopy(self.model.state_dict()))

            print(
                f"epoch {epoch:>3d} validation_loss: {vloss:>7f} "
                f"train time: {train_time:>4f}s eval time: {eval_time:>4f}s"
            )
        total_end = time.time()
        total_time = total_end - total_start
        print(f"total time: {total_time:>4f}s")

    @property
    def latent(self):
        """latent features per sample (N x K)"""
        return self.model.z.detach().cpu().numpy()

    @property
    def components(self):
        """fitted principal component vectors (K x P)"""
        return self.model.W.detach().cpu().numpy()

    @property
    def offset(self):
        """fitted mean offset per feature"""
        return self.model.mu.detach().cpu().numpy()

    def fit_transform(self, X: np.ndarray, lr=1e-3, n_epoch=10):
        """Optimize model and return latent loadings

        Parameters
        ----------
        X: `np.ndarray`
            input data matrix (N samples x P features)

        Returns
        -------
        z: `np.ndarray`
            inferred latent loadings (N samples x K components)
        """
        # coerce data to float
        self._init_model(X.astype("float32"), lr=lr)
        self._train(n_epoch=n_epoch)

        return self.latent
