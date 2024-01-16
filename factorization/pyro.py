"""
Pyro Module Routines
 - probabilistic PCA
 - sparse/regularized PCA
"""

import torch
from loguru import logger

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from factorization.model import _MFModule
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample


class MFModel(_MFModule, PyroModule):
    def __init__(
        self,
        X_shape: tuple[int, int],
        K: int = 2,
        configuration: str = None,
        glm: str = None,
    ):
        # explicitly match initialization signature
        super(MFModel, self).__init__(X_shape, K, configuration, glm)
        self.z = PyroSample(
            lambda self: dist.Normal(0, 1).expand([self.N, self.K]).to_event(1)
        )  # [N, K]
        self.W = PyroSample(
            lambda self: dist.Normal(0, 1).expand([self.P, self.K]).to_event(1)
        )  # [P, K]

    def forward(self, X, indices):
        """MF Model Generative Process

        any sampling logic pertaining to module parameters must be here.
        not minibatched yet
        """
        # relevant dimensions
        N, P, K = self.N, self.P, self.K

        # globals
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(0, 2))

        # plates
        latent_plate = pyro.plate("latent", size=N, dim=-1)
        data_plate = pyro.plate("data", size=N, dim=-2)
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
            obs = pyro.sample("obs", dist.Normal(Xt, obs_scale), obs=X)  # [N, P]
            assert obs.shape == (N, P)

        return Xt


class PyroMF:
    """Bayesian version of MF using pyro."""

    def __init__(
        self,
        n_components: int = 2,
        # configuration: str = None,
        # glm: str = None,
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
        # CPU or GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

    def _get_guide(self, guide_type, model):
        if guide_type == "map":
            pass
        elif guide_type == "diag":
            pass
        elif guide_type == "lowrank":
            pass
        elif guide_type == "fullrank":
            pass

    def _init_model(self):
        """Initialization for pyro model components.

        including dataloaders, model, guide, elbo, optimizer, svi
        """

    def _train(self):
        pyro.clear_param_store()
        pass

    def fit(self):
        pass

    def fit_transform(self):
        pass

    def transform(self):
        pass
