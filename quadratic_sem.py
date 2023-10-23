import logging
from typing import Literal, Union
import numpy as np


class QuadraticSEM:
    r"""Causal structural equation model with additive Gaussian noise and quadratic link function.

    Input is a topologically ordered DAG with adjacency matrix A.

    Model is
        z_i = f_i(z_{\Pa(i)}) + \epsilon_i
    where
        f_{i}(z_{\Pa(i)}) = z_{\Pa(i)}^{\top} z_{\Pa(i)}
    and \epsilon_i is Gaussian random variable with known variance.

    For hard interventions, we have z_i = \epsilon_i but dist. of \epsilon_i changes.
    1. First hard intervention (q_i), increase the noise variance by 1.
    2. Second hard intervention (\tilde q_i), increase the noise variance by 2.
    """

    def __init__(
        self,
        A: np.ndarray,
        variances: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        assert A.dtype == bool and A.ndim == 2 and A.shape[0] == A.shape[1]
        self.n = A.shape[0]
        assert variances.ndim == 1 and variances.shape[0] == self.n
        self.rng = rng
        self.A = A
        self.variances = variances
        self.pa = [[j for j in range(self.n) if A[j, i]] for i in range(self.n)]
        # self.a_mat = [
        #     rng.random((len(self.pa[i]), len(self.pa[i]))) for i in range(self.n)
        # ]
        # self.a_mat = [a_mati.T @ a_mati for a_mati in self.a_mat]

        # make sure that min. singular value is at least 0.1
        self.a_mat = [np.zeros((len(self.pa[i]), len(self.pa[i]))) for i in range(self.n)]
        for i in range(self.n):
            if len(self.pa[i]) == 0:
                self.a_mat[i] = np.zeros((0, 0))
            else:
                a_mat_i = rng.random((len(self.pa[i]), len(self.pa[i])))
                a_mat_svs = np.linalg.svd(a_mat_i, compute_uv=False)
                while np.min(np.abs(a_mat_svs)) < 0.1:
                    a_mat_i = rng.random((len(self.pa[i]), len(self.pa[i])))
                    a_mat_svs = np.linalg.svd(a_mat_i, compute_uv=False)
                self.a_mat[i] = a_mat_i.T @ a_mat_i


    def sample(
        self,
        shape: tuple[int, ...],
        environment: list[int] = [],
        inc_noise_variance = 1.0,
    ) -> np.ndarray:
        samples = np.zeros(shape + (self.n,))
        noises = self.rng.normal(0.0, inc_noise_variance, shape + (self.n,))

        for i in range(self.n):
            if i in environment:
                samples[..., i] = np.sqrt(self.variances[i] + inc_noise_variance) * noises[..., i]

            else:
                samples[..., i] = (
                    np.sqrt(
                        (
                            samples[..., None, self.pa[i]]
                            @ self.a_mat[i]
                            @ samples[..., self.pa[i], None]
                        )[..., 0, 0]
                    )
                    + np.sqrt(self.variances[i]) * noises[..., i]
                )
        return samples

    def score_fn(
        self,
        x: np.ndarray,
        environment: list[int] = [],
        inc_noise_variance = 1.0,
    ) -> np.ndarray:
        # First compute the link function values, noise realizations, and score of noise realizations.
        n = np.empty_like(x)
        lf = np.empty_like(x)
        r = np.empty_like(x)
        for i in range(self.n):
            if i in environment:
                lf[..., i] = 0
                n[..., i] = x[..., i]
                r[..., i] = -n[..., i] / (self.variances[i] + inc_noise_variance)

            else:
                lf[..., i] = np.sqrt(
                    (
                        x[..., None, self.pa[i]]
                        @ self.a_mat[i]
                        @ x[..., self.pa[i], None]
                    )[..., 0, 0]
                )
                n[..., i] = x[..., i] - lf[..., i]
                r[..., i] = -n[..., i] / (self.variances[i])

        # Next, compute the score function values
        s = r.copy()
        for i in range(self.n):
            # if i is intervened, already copied from r
            # if i is not intervened, compute
            if i not in environment:
                s[..., self.pa[i]] -= (
                    r[..., i : i + 1]
                    * (self.a_mat[i] @ x[..., self.pa[i], None])[..., 0]
                    / lf[..., i : i + 1]
                )
        return s
