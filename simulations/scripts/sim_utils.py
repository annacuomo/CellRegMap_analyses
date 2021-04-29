#===============================================================================
# Simulation utilities
#===============================================================================
from collections import namedtuple
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from numpy_sugar import ddot
from numpy_sugar.linalg import economic_svd

from struct_lmm2._simulate import (
    sample_maf,
    sample_genotype,
    column_normalize,
    sample_persistent_effsizes,
    sample_persistent_effects,
    sample_gxe_effects,
    sample_random_effect,
    sample_noise_effects,
    Variances,
    _symmetric_decomp,
    jitter
)

from settings import ENDO_PCS_PATH, ENDO_META_PATH


def _ncells_to_indices(
    n_individuals: int,
    n_cells: Union[int, List[int]],
):
    """Computes number of samples and indices for cells of each individual.
    
    Parameters
    ----------
    n_individuals
        Number of individuals to simulate.
    n_cells 
        Cells per individual.
    """
    if np.isscalar(n_cells):
        n_samples = n_cells * n_individuals
        individual_groups = np.array_split(
            range(n_cells * n_individuals), n_individuals)
    else:
        n_samples = sum(n_cells)
        individual_groups = np.split(range(n_samples), np.cumsum(n_cells))[:-1]
    return n_samples, individual_groups


def sample_clusters(
    n_clusters: int,
    n_individuals: int,
    n_cells: Union[int, List[int]],
    random: np.random.Generator,
    dirichlet_alpha: float = None
) -> ArrayLike:
    """Creates one-hot encoding for cell clusters in multi-donor data.

    Passing dirichlet_alpha allows sampling individual-specific cluster distributions.

    Parameters
    ----------
    n_clusters
        Number of clusters to simulate.
    n_individuals
        Number of individuals to simulate.
    n_cells 
        Cells per individual.
    random
        Random number generator.
    dirichlet_alpha
        If not None, sample the cluster assignment probabilities for each
        individual from a Dirichlet distribution with concentration parameters
        dirichlet_alpha * np.ones(n_clusters). 
    """
    n_samples, individual_groups = _ncells_to_indices(n_individuals, n_cells)

    if dirichlet_alpha is not None:
        # non-uniform donor-specific cell distribution
        probs = random.dirichlet(
            dirichlet_alpha * np.ones(n_clusters),
            size=n_individuals)
    else:
        # uniform cell distribution
        probs = np.ones((n_individuals, n_clusters)) / n_clusters
    # create cluster kernel from one-hot encodings

    E = np.zeros((n_samples, n_clusters))
    for i, g in enumerate(individual_groups):
        E[g, :] = random.multinomial(1, pvals=probs[i, :], size=len(g))
    return E


def sample_endo(
    d_env: int,
    n_individuals: int,
    n_cells: Union[int, List[int]],
    random: np.random.Generator,
    respect_individuals: bool = True
) -> ArrayLike:
    """Samples from Endoderm differentiation PCs.
    
    Parameters
    ----------
    d_env 
        Number of PCs to sample.
    n_individuals
        Number of individuals to simulate.
    n_cells 
        Cells per individual.
    random
        Random number generator.
    respect_individuals
        Use meta information to sample cells for each synthetic individual from
        only one real individual. Otherwise sample from the combined set of 
        cells.
    """
    n_samples, individual_groups = _ncells_to_indices(n_individuals, n_cells)
    cells_by_individual = [len(g) for g in individual_groups]

    endo_pcs = pd.read_csv(ENDO_PCS_PATH, index_col=0).iloc[:, :d_env]
    if not respect_individuals:
        return endo_pcs.loc[random.choice(endo_pcs.index, n_samples)].to_numpy()

    endo_meta = pd.read_csv(ENDO_META_PATH, sep='\t')
    ids = endo_pcs.index.intersection(endo_meta.index)

    endo_pcs = endo_pcs.loc[ids]
    endo_meta = endo_meta.loc[ids]

    top_donors = endo_meta['donor'].value_counts(sort=True)[:n_individuals].index

    E = np.zeros((n_samples, d_env))
    for i, gi in enumerate(reversed(np.argsort(cells_by_individual))):
        donor = top_donors[i]
        try:
            ids = random.choice(
                endo_meta.query('donor == @donor').index,
                len(individual_groups[gi]),
                replace=False)
        except ValueError:
            raise ValueError('Not enough real cells per donor. '
                'Consider using respect_individuals=False.')
        E[individual_groups[gi], :] = endo_pcs.loc[ids].to_numpy()
    return E


EnvDecomp = namedtuple('EnvDecomp', 'E U S')
def create_environment_factors(E: ArrayLike) -> EnvDecomp:
    """Normalizes and decomposes environments."""
    K = E @ E.T
    K /= K.diagonal().mean()
    jitter(K)
    E = _symmetric_decomp(K)

    [U, S, _] = economic_svd(E)
    return EnvDecomp(E, U, S)
    

def create_kinship_matrix(
    n_individuals: int,
    n_cells: Union[int, List[int]],
) -> ArrayLike:
    """Creates block-diagonal kinship matrix."""
    n_samples, individual_groups = _ncells_to_indices(n_individuals, n_cells)
    K = np.zeros((n_samples, len(individual_groups)))

    for i, idx in enumerate(individual_groups):
        K[idx, i] = 1.0
    return K @ K.T


KinDecomp = namedtuple('KinDecomp', 'Lk K')
def create_kinship_factors(K: ArrayLike) -> KinDecomp:
    """Normalizes and decomposes kinship matrix."""
    K /= K.diagonal().mean()
    jitter(K)
    return KinDecomp(_symmetric_decomp(K), K)
 

Simulation = namedtuple(
    'Simulation', 
    'mafs y beta_g y_g y_gxe y_k y_e y_n G Ls'
)
def simulate_data(
    offset: float,
    n_individuals: int,
    n_snps: int,
    n_cells: Union[int, List[int]],
    env: EnvDecomp,
    Lk: ArrayLike,
    maf_min: float,
    maf_max: float,
    g_causals: list,
    gxe_causals: list,
    variances: Variances,
    random: np.random.Generator,
) -> Simulation:
    """Simulates data from StructLMM2 model."""
    n_samples = env.E.shape[0]

    mafs = sample_maf(n_snps, maf_min, maf_max, random)
    G = sample_genotype(n_individuals, mafs, random)
    G = np.repeat(G, n_cells, axis=0)
    G = column_normalize(G)

    us = env.U * env.S
    Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

    beta_g = sample_persistent_effsizes(n_snps, g_causals, variances.g, random)

    y_g = sample_persistent_effects(G, beta_g, variances.g)
    y_gxe = sample_gxe_effects(G, env.E, gxe_causals, variances.gxe, random)
    y_k = sample_random_effect(Ls, variances.k, random)
    y_e = sample_random_effect(env.E, variances.e, random)
    y_n = sample_noise_effects(n_samples, variances.n, random)

    y = offset + y_g + y_gxe + y_k + y_e + y_n

    simulation = Simulation(
        mafs=mafs,
        y=y,
        beta_g=beta_g,
        y_g=y_g,
        y_gxe=y_gxe,
        y_k=y_k,
        y_e=y_e,
        y_n=y_n,
        G=G,
        Ls=Ls,
    )
    return simulation


def sample_nb(
    mu: ArrayLike,
    phi: ArrayLike,
    random: np.random.Generator,
    size: int=None,
) -> ArrayLike:
    """Samples from a negative binomial distribution.

    Parameterization using mean (mu) and dispersion (phi).
    """
    n = 1/phi
    p = n / (n + mu) 
    return random.negative_binomial(n=n, p=p, size=size)

    