"""Additional statistical functions used for the simulation experiments."""

# ===============================================================================
# Imports
# ===============================================================================
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

import scipy
from tqdm import trange
from numpy_sugar import epsilon

from glimix_core.lmm import LMM


def sample_nb(
    mu: ArrayLike, phi: ArrayLike, random: np.random.Generator, size: int = None,
) -> ArrayLike:
    """Samples from a negative binomial distribution.

    Parameterization using mean (mu) and dispersion (phi).
    """
    n = 1 / phi
    p = n / (n + mu)
    return random.negative_binomial(n=n, p=p, size=size)


def lrt_pvalues(null_lml: ArrayLike, alt_lml: ArrayLike, dof: int = 1) -> ArrayLike:
    """Compute p-values from likelihood ratios.

    Parameters
    ----------
    null_lml
        Log of the marginal likelihood under the null hypothesis.
    alt_lmls
        Log of the marginal likelihoods under the alternative hypotheses.
    dof
        Degrees of freedom.

    Returns
    -------
    pvalues
        P-values.
    """
    null_lml = np.asarray(null_lml, float)
    alt_lml = np.asarray(alt_lml, float)
    lrs = np.clip(-2 * null_lml + 2 * alt_lml, epsilon.super_tiny, np.inf)
    pv = scipy.stats.chi2(df=dof).sf(lrs)
    return np.clip(pv, epsilon.super_tiny, 1 - epsilon.tiny)


def run_cellregmap_fixed(
    y: ArrayLike,
    M: ArrayLike,
    E0: ArrayLike,
    E1: ArrayLike,
    G: ArrayLike,
    QS: Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike],
    joint: bool = False,
):
    """Test for GxE effects using the fixed effect version of CellRegMap.

    Parameters
    ----------
    y
        Phenotype vector.
    M
        Covariate matrix.
    E0
        Environments to test.
    E1
        Background environments.
    G
        Genotype matrix.
    QS
        QS decomposition of K * EE^T.
    joint
        Use joint LRT or test individually and adjust using Bonferroni.

    Returns
    -------
    pvalues
        P-values.
    """
    if joint:
        pvals = list()
        dof = E0.shape[1]
        for i in trange(G.shape[1]):
            g = G[:, i, np.newaxis]

            # fit null model (no interactions)
            M = np.concatenate([M, g, E1], axis=1)
            lmm = LMM(y, M, QS, restricted=False)
            lmm.fit(verbose=False)
            lml0 = lmm.lml()

            # fit alternative model
            M = np.concatenate([M, g, E1, E0 * g], axis=1)
            lmm = LMM(y, M, QS, restricted=False)
            lmm.fit(verbose=False)
            lml1 = lmm.lml()

            pvals.append(lrt_pvalues(lml0, lml1, dof))
        return pvals

    lml0 = list()
    lml1 = list()
    dof = 1
    for i in trange(G.shape[1]):
        g = G[:, i, np.newaxis]
        M = np.concatenate([M, g, E1], axis=1)
        lmm = LMM(y, M, QS, restricted=False)
        lmm.fit(verbose=False)
        scanner = lmm.get_fast_scanner()
        d = scanner.fast_scan(E0 * g, verbose=False)
        lml0.append(scanner.null_lml())
        # only record max likelihood solution across environments
        lml1.append(d["lml"].max())

        # # free instance for GC
        # del lmm._logistic
        # del lmm._variables
        # _clear_lru_cache()

    # compute LRT and adjust for number of environments (Bonferroni)
    return lrt_pvalues(lml0, lml1, dof) * E0.shape[1]


# def _clear_lru_cache():
#     import functools
#     import gc
#     wrappers = [
#         a for a in gc.get_objects()
#         if isinstance(a, functools._lru_cache_wrapper)]
#     for wrapper in wrappers:
#         wrapper.cache_clear()
