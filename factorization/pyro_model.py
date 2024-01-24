"""
Pyro Module Routines
 - probabilistic PCA
 - sparse/regularized PCA
"""

import copy
import time

import numpy as np
import torch
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

import pyro
import pyro.distributions as dist
from factorization.model import TensorIndicesDataset, _MFModule
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample


class MFModel(_MFModule, PyroModule):
    def __init__(
        self,
        X_shape: tuple[int, int],
        K: int = 2,
        configuration: str = None,
        glm: str = None,
        sparse: bool = False,
    ):
        # explicitly match initialization signature
        super(MFModel, self).__init__(X_shape, K, configuration, glm)
        self.sparse = sparse

        # indicate param sampling
        self.z = PyroSample(
            lambda self: dist.Normal(0, 1).expand([self.N, self.K]).to_event(1)
        )  # [N, K]
        if not self.sparse:
            self.W = PyroSample(
                lambda self: dist.Normal(0, 1).expand([self.P, self.K]).to_event(1)
            )  # [P, K]
        else:
            # ARD sparse prior
            self.feature_precision = PyroParam(torch.zeros(self.P))
            self.W = PyroSample(
                lambda self: dist.Normal(
                    0, torch.exp(self.feature_precision.unsqueeze(1))
                )
                .expand([self.P, self.K])
                .to_event(1)
            )

    def forward(self, X, indices):
        """MF Model Generative Process

        any sampling logic pertaining to module parameters must be here.
        not minibatched yet
        """
        # relevant dimensions
        N, P, K = self.N, self.P, self.K
        B = len(X)  # minibatch size

        # globals
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(0, 2))

        # plates
        latent_plate = pyro.plate("latent", size=N, dim=-1)
        data_plate = pyro.plate("data", size=N, dim=-2, subsample=X)  # minibatched
        feature_plate = pyro.plate("feature", size=P, dim=-1)

        with latent_plate:
            # trigger sampling
            z = self.z  # [N, K]
            assert z.shape == (N, K)

        with feature_plate:
            W = self.W  # [P, K]
            assert W.shape == (P, K)

        # transform sampled factors
        Xt = super(MFModel, self).forward(X, indices)
        # observations
        with data_plate, feature_plate:
            obs = pyro.sample("obs", dist.Normal(Xt, obs_scale), obs=X)  # [B, P]
            assert obs.shape == (B, P)

        return Xt


class PyroMF(BaseEstimator, TransformerMixin):
    """Bayesian version of MF using pyro."""

    def __init__(
        self,
        n_components: int = 2,
        # configuration: str = None,
        # glm: str = None,
        sparse: bool = False,
        guide_type: str = "diag",
        n_epoch: int = 10,
        early_stopping: int = 5,
        save_epoch: int = 1,
        save_model: bool = False,
        tensorboard_dir: str = None,
        lr: float = 1e-3,
    ):
        self.n_components = n_components
        # self.configuration = configuration
        # self.glm = glm
        self.sparse = sparse
        self.guide_type = guide_type  # guide configuration
        self.n_epoch = n_epoch
        self.early_stopping = early_stopping
        self.save_epoch = save_epoch
        self.save_model = save_model
        self.tensorboard_dir = tensorboard_dir
        self.lr = lr

    def _init_estimator(self):
        """Initialize estimator for fitting."""
        # automatically generated attributes
        self.history = {"loss": []}
        if self.save_model:
            self.history["model"] = []
            self.history["guide"] = []
        # CPU or GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

    def _get_guide(self, guide_type, model):
        if guide_type == "map":
            return AutoDelta(model)
        elif guide_type == "diag":
            return AutoNormal(model)
        elif guide_type == "lowrank":
            pass
        elif guide_type == "fullrank":
            pass

    def _init_model(self, X: np.ndarray, lr: float):
        """Initialization for pyro model components.

        including dataloaders, model, guide, elbo, optimizer, svi
        """
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
        self.model = MFModel(X.shape, self.n_components, sparse=self.sparse).to(
            self.device
        )

        # guide
        self.guide = self._get_guide(self.guide_type, self.model)

        # loss and optimizer
        self.elbo = Trace_ELBO(num_particles=1)
        self.optim = pyro.optim.ClippedAdam({"lr": lr})
        self.svi = SVI(self.model, self.guide, self.optim, self.elbo)

    def _train(self, n_epoch: int, save_epoch: int, tensorboard_dir: str):
        """Main training loop.

        Parameters
        ----------
        n_epoch : int
            number of training epochs.
        save_epoch : int
            epoch frequency to save history
        tensorboard_dir : str
            directory to save tensorboard info
        """
        total_start = time.time()

        # tensorboard writer
        tensorboard_record = tensorboard_dir is not None
        if tensorboard_record:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        # for early stopping
        best_epoch, best_loss = -1, np.inf

        pyro.clear_param_store()
        for epoch in trange(n_epoch, desc="epochs"):
            # training
            train_start = time.time()
            for batch, (Xb, ib) in enumerate(self.X_train_dl):
                # transfer data if necessary
                Xb, ib = Xb.to(self.device), ib.to(self.device)

                # calculate loss and take gradient step
                loss = self.svi.step(Xb, ib)

                # batch update
                if batch % 100 == 0:
                    logger.info(
                        f"    batch_loss: {loss:>7f} "
                        f"item: [{(batch + 1) * len(Xb):>7d}/{len(self.X_train_dl.dataset):>7d}]"
                    )
                    if tensorboard_record:
                        tb_writer.add_scalars(
                            "Loss",
                            {"batch_loss": loss},
                            epoch * len(self.X_train_dl) + batch,
                        )

            train_end = time.time()
            train_time = train_end - train_start

            # evaluation
            eval_start = time.time()
            self.model.eval()
            vloss = 0
            with torch.no_grad():
                for Xv, iv in self.X_valid_dl:
                    # sum ELBO loss over batches
                    vloss += self.svi.evaluate_loss(Xv, iv)
            vloss /= len(self.X_valid_dl)

            eval_end = time.time()
            eval_time = eval_end - eval_start

            # record history
            if epoch % save_epoch == 0:
                self.history["loss"].append(vloss)
                if self.save_model:
                    self.history["model"].append(copy.deepcopy(self.model.state_dict()))
                    self.history["guide"].append(copy.deepcopy(self.guide.state_dict()))

            logger.info(
                f"epoch {epoch:>3d} validation_loss: {vloss:>7f} "
                f"train time: {train_time:>4f}s eval time: {eval_time:>4f}s"
            )

            if tensorboard_record:
                tb_writer.add_scalars(
                    "Loss",
                    {"validation_loss": vloss},
                    epoch * len(self.X_train_dl) + batch,
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
        if tensorboard_record:
            # stop tensorboard recording
            tb_writer.flush()
            tb_writer.close()

    @property
    def latent(self):
        """latent features per sample (N x K), posterior median"""
        return self.guide.median()["z"].detach().cpu().numpy()

    @property
    def components(self):
        """fitted principal component vectors (K x P), posterior median"""
        return self.guide.median()["W"].detach().cpu().numpy()

    @property
    def offset(self):
        """fitted mean offset per feature, posterior median"""
        return self.model.mu.detach().cpu().numpy()

    def sample_latent_rvs(self):
        """Returrn dictionary containnig one random sample of latent variables from the guide."""
        return dict(self.guide(*next(iter(self.X_train_dl))).items())

    def named_parameters(self):
        """Return dictionary containing torch parameters between model and guide."""
        return dict(pyro.get_param_store().named_parameters())

    def sample_posterior(self, num_samples=1000):
        """Generate posterior predictive samples."""
        predictive = Predictive(
            pyro.poutine.uncondition(
                self.model
            ),  # must uncondition obs to sample predictive
            guide=self.guide,
            num_samples=num_samples,
            return_sites=list(self.sample_latent_rvs().keys()) + ["obs", "_RETURN"],
        )
        X_tensor = self.X_ds.tensors[0]
        return predictive(X_tensor, range(len(X_tensor)))

    def fit(self, X: np.ndarray):
        """Fit model and return estimator."""
        self._init_estimator()
        # coerce data to float
        self._init_model(X.astype("float32"), lr=self.lr)
        self._train(
            n_epoch=self.n_epoch,
            save_epoch=self.save_epoch,
            tensorboard_dir=self.tensorboard_dir,
        )
        return self

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

    def transform(self):
        pass
