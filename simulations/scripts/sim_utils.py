"""Additional utility functions used for the simulation experiments.

To be ingrated into struct_lmm2._simulate?"""
# ===============================================================================
# Imports
# ===============================================================================
from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from numpy_sugar.linalg import economic_svd

from cellregmap._simulate import (
    sample_maf,
    sample_genotype,
    column_normalize,
    sample_persistent_effsizes,
    sample_persistent_effects,
    sample_gxe_effects,
    sample_random_effect,
    sample_noise_effects,
    _symmetric_decomp,
    jitter,
)

from pandas_plink import read_plink1_bin

import settings


# ===============================================================================

Variances = namedtuple("Variances", "g gxe k e n")


def create_variances(r0, v0, has_kinship=True, include_noise=True) -> Variances:
    """
    Remember that:
        cov(𝐲) = 𝓋₀(1-ρ₀)𝙳𝟏𝟏ᵀ𝙳 + 𝓋₀ρ₀𝙳𝙴𝙴ᵀ𝙳 + 𝓋₁ρ₁EEᵀ + 𝓋₁(1-ρ₁)𝙺 + 𝓋₂𝙸.
    Let us define:
        σ²_g   = 𝓋₀(1-ρ₀) (variance explained by persistent genetic effects)
        σ²_gxe = 𝓋₀ρ₀     (variance explained by GxE effects)
        σ²_e   = 𝓋₁ρ₁     (variance explained by environmental effects)
        σ²_k   = 𝓋₁(1-ρ₁) (variance explained by population structure)
        σ²_n   = 𝓋₂       (residual variance, noise)
    We set the total variance to sum up to 1: TODO keep this in?
        1 = σ²_g + σ²_gxe + σ²_e + σ²_k + σ²_n
    We set the variances explained by the non-genetic terms to be equal:
        v = σ²_e = σ²_k = σ²_n
    For `has_kinship=False`, we instead set the variances such that:
        v = σ²_e = σ²_n
    Parameters
    ----------
    r0 : float
        This is ρ₀.
    v0 : float
        This is 𝓋₀.
    """
    v_g = v0 * (1 - r0)
    v_gxe = v0 * r0

    v_k = 0.0
    if has_kinship:
        n_terms = 3 if include_noise else 2
        v = (1 - v_gxe - v_g) / n_terms
        v_e = v
        v_k = v
        v_n = v if include_noise else None
    else:
        n_terms = 2 if include_noise else 1
        v = (1 - v_gxe - v_g) / n_terms
        v_e = v
        v_n = v if include_noise else None

    variances = {"g": v_g, "gxe": v_gxe, "e": v_e, "n": v_n}
    if has_kinship:
        variances["k"] = v_k
    else:
        variances["k"] = None

    return Variances(**variances)


def set_causal_ids(
    n_causal_g: int, n_causal_gxe: int, n_causal_shared: int
) -> Tuple[List[int], List[int]]:
    """Set ids of causal SNPs for persistent and gxe effects.

    The first n_causal_g SNPs will be persistent genetic effects, while
    n_causal_gxe indices starting at n_causal_g - n_causal_shared will be GxE
    effects.

    Parameters
    ----------
    n_causal_g
        Number of SNPS with persistent effects.
    n_causal_gxe
        Number of SNPs with GxE effects.
    n_causal_shared
        Number of SNPs with both persistent and GxE effects.
    """
    if n_causal_shared > min(n_causal_g, n_causal_gxe):
        raise ValueError(
            "n_causal_shared has to be smaller" "than either n_causal_g or n_causal_gxe"
        )
    g_causals = list(range(0, n_causal_g))
    gxe_causals = list(
        range(n_causal_g - n_causal_shared, n_causal_g + n_causal_gxe - n_causal_shared)
    )
    return (g_causals, gxe_causals)


def _ncells_to_indices(
    n_individuals: int, n_cells: Union[int, List[int]],
):
    """Computes number of samples (total number of cells across all individuals)
    and indices for cells of each individual.

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
            range(n_cells * n_individuals), n_individuals
        )
    else:
        n_samples = sum(n_cells)
        individual_groups = np.split(range(n_samples), np.cumsum(n_cells))[:-1]
    return n_samples, individual_groups


def sample_endo(
    n_env: int,
    n_individuals: int,
    n_cells: Union[int, List[int]],
    random: np.random.Generator,
    respect_individuals: bool = True,
) -> ArrayLike:
    """Samples from Endoderm differentiation PCs.

    Parameters
    ----------
    n_env
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

    Returns
    ----------
    E
        Environmental variables.
    """
    n_samples, individual_groups = _ncells_to_indices(n_individuals, n_cells)
    if np.isscalar(n_cells):
        n_cells = (np.ones(n_individuals) * n_cells).astype(int)

    endo_pcs = pd.read_csv(settings.ENDO_PCS_PATH, index_col=0).iloc[:, :n_env]
    if not respect_individuals:
        return endo_pcs.loc[random.choice(endo_pcs.index, n_samples)].to_numpy()

    endo_meta = pd.read_csv(settings.ENDO_META_PATH, sep="\t")
    endo_meta = endo_meta.loc[endo_pcs.index.intersection(endo_meta.index)]

    top_donors = (
        endo_meta["donor_short_id"].value_counts(sort=True)[:n_individuals].index
    )

    E = np.zeros((n_samples, n_env))
    # sample cells for each donor, beginning with the donor with most cells
    for i, gi in enumerate(np.argsort(n_cells)[::-1]):
        donor = top_donors[i]
        try:
            ids = random.choice(
                endo_meta[endo_meta["donor_short_id"] == donor].index,
                len(individual_groups[gi]),
                replace=False,
            )
        except ValueError:
            raise ValueError(
                "Not enough real cells per donor. "
                "Consider using respect_individuals=False."
            )
        E[individual_groups[gi], :] = endo_pcs.loc[ids].to_numpy()
    return column_normalize(E)


EnvDecomp = namedtuple("EnvDecomp", "E U S")


def create_environment_factors(E: ArrayLike) -> EnvDecomp:
    """Normalizes and decomposes environments."""
    K = E @ E.T
    K /= K.diagonal().mean()
    jitter(K)
    E = _symmetric_decomp(K)

    [U, S, _] = economic_svd(E)
    return EnvDecomp(E, U, S)


KinDecomp = namedtuple("KinDecomp", "Lk K")


def create_kinship_factors(K: ArrayLike) -> KinDecomp:
    """Normalizes and decomposes kinship matrix."""
    K /= K.diagonal().mean()
    jitter(K)
    return KinDecomp(_symmetric_decomp(K), K)


Simulation = namedtuple("Simulation", "snp_ids y beta_g y_g y_gxe y_k y_e y_n G")


def simulate_data(
    offset: float,
    n_individuals: int,
    n_snps: int,
    n_cells: Union[int, List[int]],
    env: EnvDecomp,
    env_gxe_active: List[int],
    Ls: Tuple[ArrayLike],
    maf_min: float,
    maf_max: float,
    real_genotypes: bool,
    donor_ids: List[str],
    g_causals: List[int],
    gxe_causals: List[int],
    variances: Variances,
    random: np.random.Generator,
) -> Simulation:
    """Simulates data from CellRegMap model.

    Parameters
    ----------
    offset
        Constant intercept.
    n_individuals
        Number of individuals to simulate.
    n_snps
        Number of SNPs to simulate.
    n_cells
        The number of cells per individuals. Either int (if same number of cells
        per individual) or a list of integers.
    env
        EnvDecomp named tuple containing the environment matrix and its SVD.
    env_gxe_active
        Indices of environments with GxE effects.
    Ls
        Factorization of K * EE^T (interaction of repeat structure and
        environment; see documentation of CellRegMap).
    maf_min
        Minimum minor allele frequency.
    maf_max
        Maximum minor allele frequency.
    real_genotypes
        Use real genotypes from HipSci.
    donor_ids
        Donors to simulate (if real_genotypes is True). Length has to match
        n_individuals.
    g_causals
        Indices of SNPs with genetic effects.
    gxe_causals
        Indices of SNPs with GxE effects.
    variances
        Named tuple with scaling parameters for each variance component.
    random
        Random number generator.


    Returns
    -------
    Simulation named tuple.
    """
    v_g = variances.g if len(g_causals) > 0 else 0
    v_gxe = variances.gxe if len(gxe_causals) > 0 else 0

    n_samples = env.E.shape[0]

    # sample genotypes

    # TODO think of better solution ...
    MAX_ATTEMPTS_NON_CONSTANT = (
        5  # attempt to sample genotypes with non-zero variance across individuals
    )
    MAX_ATTEMPTS_SMP = (
        5  # attempt to sample requested number of SNPs from a 2MB window of real SNPs
    )

    attempts_1 = 0
    attempts_2 = 0
    while attempts_1 < MAX_ATTEMPTS_NON_CONSTANT:
        if not real_genotypes:
            # synthetic genotypes
            mafs = sample_maf(n_snps, maf_min, maf_max, random)
            G = sample_genotype(n_individuals, mafs, random)
            snp_ids = None
        else:
            # sample real variants from a random 2Mb window (HipSci)
            G = read_plink1_bin(settings.FILTERED_GENO_PATH, verbose=False)
            G = G[pd.Series(G.sample).isin(donor_ids), :]
            while attempts_2 < MAX_ATTEMPTS_SMP:
                region_start = random.choice(48 * 10 ** 6 - 2 * 10 ** 6)
                G_region = G[
                    :, (G.pos >= region_start) & (G.pos < region_start + 2 * 10 ** 6)
                ]
                region_snps = G_region.shape[1]
                if region_snps >= n_snps:
                    G_region = G_region[
                        :, random.choice(region_snps, n_snps, replace=False)
                    ]
                    G = G_region.values
                    snp_ids = G_region.snp.to_numpy()
                    break
                attempts_2 += 1
            if attempts_2 == MAX_ATTEMPTS_SMP:
                raise ValueError(
                    "Could not sample 2Mb region with %d SNPs. Try again or lower n_snps."
                    % n_snps
                )
        if (np.abs(G - G[0, :]).sum(0) == 0).any():
            attempts_1 += 1
            if attempts_1 == MAX_ATTEMPTS_NON_CONSTANT:
                raise ValueError(
                    "Could not sample non-constant G. Try increasing the number of individuals."
                )
        else:
            break

    G = np.repeat(G, n_cells, axis=0)
    G = column_normalize(G)

    # sample effects
    beta_g = sample_persistent_effsizes(n_snps, g_causals, v_g, random)
    y_g = sample_persistent_effects(G, beta_g, v_g)
    y_gxe = sample_gxe_effects(G, env.E[:, env_gxe_active], gxe_causals, v_gxe, random)
    y_k = sample_random_effect(Ls, variances.k, random)
    y_e = sample_random_effect(env.E, variances.e, random)
    if variances.n is None:
        y_n = np.zeros_like(y_e)
    else:
        y_n = sample_noise_effects(n_samples, variances.n, random)

    simulation = Simulation(
        snp_ids=snp_ids,
        y=offset + y_g + y_gxe + y_k + y_e + y_n,
        beta_g=beta_g,
        y_g=y_g,
        y_gxe=y_gxe,
        y_k=y_k,
        y_e=y_e,
        y_n=y_n,
        G=G,
    )
    return simulation

