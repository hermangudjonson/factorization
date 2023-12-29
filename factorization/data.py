"""Data Generation Routines"""

import sklearn.datasets as skdata
from scipy.stats import special_ortho_group, bernoulli, poisson
from scipy.special import expit
import numpy as np


class MoonMaker:
    def __init__(
        self,
        n_samples: int = 100,
        latent_dim: int = 5,
        latent_std: float = 0.05,
        latent_scale: float = 1.0,
        latent_shift: float = 0.0,
        noise_dim: int = 5,
        noise_std: float | None = None,
        noise_var_prop: float | None = 0.2,
        nonnegative: bool = False,
        glm: str = None,
        random_state=None,
    ):
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.latent_std = latent_std
        self.latent_scale = latent_scale
        self.latent_shift = latent_shift
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_var_prop = noise_var_prop
        self.nonnegative = nonnegative
        self.glm = glm
        self.random_state = random_state

    def _glm_transform(self, X, glm):
        if glm == "binomial":
            X_glm = bernoulli.rvs(expit(X), random_state=self.random_state)
        if glm == "poisson":
            X_glm = poisson.rvs(np.exp(X), random_state=self.random_state)
        return X_glm

    def make_moons(self):
        # generate latent data
        self.X_latent, self.y = skdata.make_moons(
            self.n_samples,
            shuffle=True,
            noise=self.latent_std,
            random_state=self.random_state,
        )

        # rotate (K x 2)
        if self.latent_dim > 2:
            # only rotate if we're increasing the latent dimension
            self.rot = special_ortho_group.rvs(
                self.latent_dim, random_state=self.random_state
            )[:, :2]
        else:
            self.rot = np.eye(2)
        X_rot = self.X_latent @ self.rot.T

        # latent scale and shift
        X_rot = self.latent_scale * X_rot + self.latent_shift

        # add noise
        if self.noise_dim > 0:
            if self.noise_var_prop is not None:
                # calculate noise_std to satisfy noise proportion
                noise_var_total = (
                    self.noise_var_prop
                    * (X_rot.var(axis=0).sum())
                    / (1 - self.noise_var_prop)
                )
                self.noise_std = np.sqrt(noise_var_total / self.noise_dim)
            X = np.hstack(
                [
                    X_rot,
                    self.noise_std * np.random.randn(self.n_samples, self.noise_dim),
                ]
            )
        else:
            X = X_rot

        if self.nonnegative:
            X = X - X.min(axis=0)

        if self.glm:
            X = self._glm_transform(X, self.glm)
        return X, self.y
