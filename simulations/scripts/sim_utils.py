#===============================================================================
# Simulation utilities
#===============================================================================
from collections import namedtuple
from typing import List, Union

import numpy as np
import pandas as pd

from numpy_sugar import ddot
from numpy_sugar.linalg import economic_svd

from struct_lmm2._simulate import (
    sample_maf,
    sample_genotype,
    column_normalize,
    sample_covariance_matrix,
    sample_persistent_effsizes,
    sample_persistent_effects,
    sample_gxe_effects,
    sample_random_effect,
    sample_noise_effects,
    Simulation,
    Variances
)

from settings import ENDO_PATH


def set_causal_snps(n_causal_g, n_causal_gxe):
    g_causals = list(range(0, n_causal_g))
    gxe_causals = list(range(n_causal_g, n_causal_g + n_causal_gxe))
    return g_causals, gxe_causals


def set_cells_per_donor(cells_per_individual, n_individuals):
    if cells_per_individual == 'fixed':
        n_cells = 50
    elif cells_per_individual == 'variable':
        n_cells = np.arange(n_individuals) + 1
    else:
        raise NotImplementedError
    return n_cells


EnvDecomp = namedtuple('EnvDecomp', 'E U S')
def generate_environment_matrix(
    env: str,
    d_env: int,
    n_cells: Union[int, List[int]],
    n_individuals: int,
    random: np.random.Generator
) -> EnvDecomp:

    if np.isscalar(n_cells):
        n_samples = n_cells * n_individuals
    else:
        n_samples = n_cells.sum()

    if env == 'endo':
        endo_file = ENDO_PATH
        df = pd.read_csv(endo_file, index_col=0).iloc[:, :d_env]
        E = random.choice(df.to_numpy(), n_samples)
    elif env == 'neuro':
        raise NotImplementedError
    else:
        env_groups = np.array_split(random.permutation(range(n_samples)), d_env)
        E = sample_covariance_matrix(n_samples, env_groups)[0]

    [U, S, _] = economic_svd(E)
    return EnvDecomp(E, U, S)


def simulate_data(
    offset: float,
    n_individuals: int,
    n_snps: int,
    n_cells: Union[int, List[int]],
    env: EnvDecomp,
    maf_min: float,
    maf_max: float,
    g_causals: list,
    gxe_causals: list,
    variances: Variances,
    random: np.random.Generator,
) -> Simulation:
    """Simulate data from StructLMM2 model.

    Parameters
    ----------
    n_cells
         Integer number or array of integers.
    """
    mafs = sample_maf(n_snps, maf_min, maf_max, random)

    G = sample_genotype(n_individuals, mafs, random)
    G = np.repeat(G, n_cells, axis=0)

    G = column_normalize(G)

    n_samples = G.shape[0]

    if np.isscalar(n_cells):
        individual_groups = np.array_split(range(n_samples), n_individuals)
    else:
        individual_groups = np.split(range(n_samples), np.cumsum(n_cells))[:-1]

    Lk, K = sample_covariance_matrix(n_samples, individual_groups)

    E, U, S = env

    us = U * S
    Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

    beta_g = sample_persistent_effsizes(n_snps, g_causals, variances.g, random)

    y_g = sample_persistent_effects(G, beta_g, variances.g)

    y_gxe = sample_gxe_effects(G, E, gxe_causals, variances.gxe, random)

    y_k = sample_random_effect(Ls, variances.k, random)

    y_e = sample_random_effect(E, variances.e, random)

    y_n = sample_noise_effects(n_samples, variances.n, random)

    M = np.ones((K.shape[0], 1))
    y = offset + y_g + y_gxe + y_k + y_e + y_n

    simulation = Simulation(
        mafs=mafs,
        offset=offset,
        beta_g=beta_g,
        y_g=y_g,
        y_gxe=y_gxe,
        y_k=y_k,
        y_e=y_e,
        y_n=y_n,
        y=y,
        variances=variances,
        Lk=Lk,
        Ls=Ls,
        K=K,
        E=E,
        G=G,
        M=M,
    )
    return simulation
