"""
MF implementation practice with Pytorch
"""

import copy
import time

import numpy as np
import torch
import torchinfo
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin, clone
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrizations
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


class TensorIndicesDataset(TensorDataset):
    """Extend TensorDataset to return index along with sample."""

    def __getitem__(self, index):
        dtuple = super(TensorIndicesDataset, self).__getitem__(index)
        return tuple(list(dtuple) + [index])


class _MFModule(nn.Module):
    def __init__(self, X_shape: tuple[int, int], K: int = 2, configuration: str = None):
        super(_MFModule, self).__init__()
        # sizing
        self.N, self.P = X_shape
        self.K = K
        # model components
        self.z = Parameter(torch.randn(self.N, self.K))
        self.W = Parameter(torch.randn(self.P, self.K))
        self.mu = Parameter(torch.randn(self.P))

        # configure parameters
        if configuration is not None:
            self._configure(configuration)

    def _configure(self, configuration: str):
        """Configure and/or parametrize components"""
        if configuration == "orthogonal":
            parametrizations.orthogonal(self, "W")

    def forward(self, X, indices):
        return self.z[indices, :] @ self.W.T + self.mu


class PytorchMF(BaseEstimator, TransformerMixin):
    """Trying to achieve a simple MF implementation using generic Pytorch optimizers."""

    def __init__(
        self,
        n_components: int = 2,
        configuration: str = None,
        n_epoch: int = 10,
        early_stopping: int = 5,
        save_epoch: int = 1,
        save_model: bool = False,
        lr: float = 1e-3,
        alpha: float = 1e-2,
    ):
        self.n_components = n_components
        self.configuration = configuration
        self.n_epoch = n_epoch
        self.early_stopping = early_stopping
        self.save_epoch = save_epoch
        self.save_model = save_model
        self.lr = lr
        self.alpha = alpha  # regularization coefficient

    def _init_estimator(self):
        """Initialize estimator for fitting."""
        # automatically generated attributes
        self.history = {"loss": []}
        if self.save_model:
            self.history["model"] = []
        # CPU or GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

    def _init_model(self, X: np.ndarray, lr: float):
        """Create torch model given starting data."""
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
        self.model = _MFModule(X.shape, self.n_components, self.configuration).to(
            self.device
        )

        # loss and optimizer
        self._data_loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _regular_loss(self):
        """Compute regularization loss on parameters.

        includes regularization coefficient and size normalized
        """
        return self.alpha * torch.norm(self.model.z, p=2) / self.model.z.size(0)

    def _compute_loss(self, target_pred, target):
        """Combine data loss and regularization loss.

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

    def _train(self, n_epoch: int, save_epoch: int):
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

        # for early stopping
        best_epoch, best_loss = -1, np.inf

        for epoch in trange(n_epoch, desc="epochs"):
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
                    logger.info(
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
                    # sum data loss over batches
                    vloss += self._data_loss(X_recon, Xv).item() * len(Xv)
            vloss /= len(self.X_valid_dl.dataset)
            vloss += self._regular_loss().item()

            eval_end = time.time()
            eval_time = eval_end - eval_start

            # record history
            if epoch % save_epoch == 0:
                self.history["loss"].append(vloss)
                if self.save_model:
                    self.history["model"].append(copy.deepcopy(self.model.state_dict()))

            logger.info(
                f"epoch {epoch:>3d} validation_loss: {vloss:>7f} "
                f"train time: {train_time:>4f}s eval time: {eval_time:>4f}s"
            )

            # early stopping
            if vloss > best_loss or np.isclose(vloss, best_loss):
                # no improvement
                if epoch - best_epoch >= self.early_stopping:
                    logger.info(
                        f"early stopping: no improvement in {self.early_stopping} epochs"
                    )
                    break
            else:
                best_epoch, best_loss = epoch, vloss

        # early stopping warning
        if n_epoch - best_epoch < self.early_stopping:
            logger.info(
                "early stopping condition not reached: consider increasing n_epoch or lr"
            )

        total_end = time.time()
        total_time = total_end - total_start
        logger.info(f"total time: {total_time:>4f}s")

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

    def fit(self, X: np.ndarray):
        """Fit model and return estimator."""
        self._init_estimator()
        # coerce data to float
        self._init_model(X.astype("float32"), lr=self.lr)
        self._train(n_epoch=self.n_epoch, save_epoch=self.save_epoch)
        return self

    def transform(self, X: np.ndarray):
        """Transform with fitted component and bias."""
        # check if fitted
        if not hasattr(self, "model"):
            raise RuntimeError("Must call fit before transform.")

        self.eval_estimator = clone(self)
        # set and freeze params after initialization
        self.eval_estimator._init_estimator()
        self.eval_estimator._init_model(X.astype("float32"), lr=self.eval_estimator.lr)
        self.eval_estimator.model.W = nn.Parameter(
            self.model.W.clone().detach(), requires_grad=False
        )
        self.eval_estimator.model.mu = nn.Parameter(
            self.model.mu.clone().detach(), requires_grad=False
        )
        # manual train
        self.eval_estimator._train(
            n_epoch=self.eval_estimator.n_epoch,
            save_epoch=self.eval_estimator.save_epoch,
        )
        return self.eval_estimator.latent

    def fit_transform(self, X: np.ndarray):
        """Optimize model and return latent loadings.

        Parameters
        ----------
        X: `np.ndarray`
            input data matrix (N samples x P features)

        Returns
        -------
        z: `np.ndarray`
            inferred latent loadings (N samples x K components)
        """
        self.fit(X)
        return self.latent

    def summary(self):
        """Generate torchinfo summary of model."""
        return torchinfo.summary(self.model, next(iter(self.X_train_dl)))
