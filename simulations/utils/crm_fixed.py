"""Fixed effect versions of CellRegMap."""

from typing import List

import numpy as np
from numpy.typing import ArrayLike

from tqdm import trange

from glimix_core.lmm import LMM
from cellregmap._cellregmap import CellRegMap, lrt_pvalues


def run_cellregmap_fixed_single_env(
    y: ArrayLike,
    W: ArrayLike,
    E: ArrayLike,
    E1: ArrayLike,
    Ls: List[ArrayLike],
    G: ArrayLike,
    idx_G=None,
):
    """Likelihood ratio test for single contexts as fixed effects in LMM.

    Arguments mirror CellRegMap. P-values are Bonferroni-adjusted for the number of contexts.
    """
    pv = list()
    crm = CellRegMap(y=y, E=E, W=W, E1=E1, Ls=Ls)
    for i in trange(G.shape[1]):
        g = G[:, i, np.newaxis]
        crm._W = np.concatenate([W, g], axis=1)
        if idx_G is None:
            Eg = E * g
        else:
            Eg = E * g[idx_G, :]
        pvals = crm.scan_association_fast(Eg)[0]
        pv.append(pvals.min() * E.shape[1])

    return pv


def run_cellregmap_fixed_multi_env(
    y: ArrayLike,
    W: ArrayLike,
    E: ArrayLike,
    E1: ArrayLike,
    Ls: List[ArrayLike],
    G: ArrayLike,
    idx_G=None,
):
    """Likelihood ratio test for all contexts as fixed effects in LMM.

    Arguments mirror CellRegMap.
    """
    pv = list()
    crm = CellRegMap(y=y, E=E, W=W, E1=E1, Ls=Ls)

    def scan_interaction_multi_env(g):
        # null model
        X = np.concatenate([W, g], axis=1)
        best = {"lml": -np.inf, "rho1": 0}
        for rho1 in crm._rho1:
            QS = crm._Sigma_qs[rho1]
            # LRT for fixed effects requires ML rather than REML estimation
            lmm = LMM(crm._y, X, QS, restricted=False)
            lmm.fit(verbose=False)

            if lmm.lml() > best["lml"]:
                best["lml"] = lmm.lml()
                best["rho1"] = rho1
                best["lmm"] = lmm
        null_lmm = best["lmm"]

        # alternative model
        if idx_G is None:
            Eg = E * g
        else:
            Eg = E * g[idx_G, :]
        X = np.concatenate((W, g, Eg), axis=1)
        QS = crm._Sigma_qs[best["rho1"]]
        alt_lmm = LMM(crm._y, X, QS, restricted=False)
        alt_lmm.fit(verbose=False)

        return lrt_pvalues(null_lmm.lml(), alt_lmm.lml(), dof=E.shape[1])

    for i in trange(G.shape[1]):
        g = G[:, i, np.newaxis]
        pv.append(scan_interaction_multi_env(g))

    return pv
