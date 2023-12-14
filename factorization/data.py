"""Data Generation Routines"""

import sklearn.datasets as skdata
from scipy.stats import special_ortho_group
import numpy as np


class MoonMaker:
    def __init__(
        self,
        n_samples: int = 100,
        latent_dim: int = 5,
        latent_std: float = 0.05,
        noise_dim: int = 5,
        noise_std: float | None = None,
        noise_var_prop: float | None = 0.2,
        nonnegative: bool = False,
        random_state=None,
    ):
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.latent_std = latent_std
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_var_prop = noise_var_prop
        self.nonnegative = nonnegative
        self.random_state = random_state

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
        # add noise
        if self.noise_var_prop is not None:
            # calculate noise_std to satisfy noise proportion
            noise_var_total = (
                self.noise_var_prop
                * (X_rot.var(axis=0).sum())
                / (1 - self.noise_var_prop)
            )
            self.noise_std = np.sqrt(noise_var_total / self.noise_dim)
        if self.noise_dim > 0:
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
        return X, self.y
