"""Main simulation script.

By default, i.e. when run as 'python simulate.py', this script will use the 
simulation parameters defined in settings.DEFAULT_PARAMS.

Alternatively, this script can be called from snakemake, as described here:
https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#parameter-space-exploration

In this case, default parameter values from settings.DEFAULT_PARAMS will be
updated if the corresponding keyword is contained in snakemake.params.simulation.
Parameters not contained in settings.DEFAULT_PARAMS will be ignored.
"""
#===============================================================================
# Imports
#===============================================================================
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs

from limix.qc import quantile_gaussianize
from struct_lmm2 import StructLMM2, create_variances

from settings import DEFAULT_PARAMS
from sim_utils import (
    set_causal_ids,
    sample_clusters,
    sample_endo,
    create_environment_factors,
    create_kinship_matrix,
    create_kinship_factors,
    simulate_data,
    sample_nb,
    run_scstructlmm2_fixed
)


#===============================================================================
# Set parameters
#===============================================================================
# check if run from snakemake
SNAKEMODE = 'snakemake' in locals() or 'snakemake' in globals()

def update_variable(var_name, default):
    """If run from snakemake, update parameter values."""
    if not SNAKEMODE:
        return default
    try:
        x = snakemake.params.simulation[var_name]
        if isinstance(x, pd.Series):
            x = x[0]
        return type(default)(x) # bit hacky
    except KeyError:
        if default is None:
            raise ValueError('%s needs to be specified' % var_name)
        else:
            return default


print('Setting parameters ...')
params = {}
for key, value in DEFAULT_PARAMS.items():
    # load default parameters & update if specified through snakemake
    params[key] = update_variable(key, value)
    print('%20s: %s' % (key, str(params[key])))
params['threads'] = snakemake.threads if SNAKEMODE else 1
params['out_file'] = snakemake.output[0] if SNAKEMODE else 'pvals.txt'


#===============================================================================
# Check values
#===============================================================================
n_causal = params['n_causal_g'] + params['n_causal_gxe'] - params['n_causal_shared']
if n_causal > params['n_snps']:
    raise ValueError('Number SNPs with genetic effects has to be < n_snps.')
if (params['n_causal_g'] == 0) ^ (params['r0'] == 0):
    print('Warning: Only one of n_causal_g or r0 is zero. Simulating no persistent effect.')
if (params['n_causal_gxe'] == 0) ^ (params['r0'] == 1):
    print('Warning: Only one of n_causal_g or (1-r0) is zero. Simulating no gxe effect.')


#===============================================================================
# Simulate data & run tests in parallel across multiple genes
#===============================================================================
# initialize random number generator
random = np.random.default_rng(params['seed'])

# (1) set parameters which are fixed for all genes, such as the envrionment
# matrix, repeat structure and their respective factorizations:

# set indices of causal SNPs
(g_causals, gxe_causals) =  set_causal_ids(
    n_causal_g=params['n_causal_g'],
    n_causal_gxe=params['n_causal_gxe'],
    n_causal_shared=params['n_causal_shared'])

# set cells per donor
if params['cells_per_individual'] == 'fixed':
    n_cells = params['n_cells']
elif params['cells_per_individual'] == 'variable':
    n_cells = np.arange(params['n_individuals']) + 1
else:
    raise ValueError(
        'Invalid cells_per_individual value: %s' % params['cells_per_individual'])

print('Setting up environment matrix ...')
# create environment matrix and decomposition
if params['env'] == 'endo':
    respect_individuals = True
    if params['n_individuals'] > 124:
        respect_individuals = False
    E = sample_endo(params['n_env'],
        params['n_individuals'],
        n_cells,
        random,
        respect_individuals=respect_individuals)
elif params['env'] == 'cluster_uniform':
    E = sample_clusters(
        params['n_env'],
        params['n_individuals'],
        n_cells,
        random)
elif params['env'] == 'cluster_biased':
    E = sample_clusters(
        params['n_env'],
        params['n_individuals'],
        n_cells,
        random,
        params['dirichlet_alpha'])
else:
    raise ValueError('Invalid env value: %s' % params['env'])
env = create_environment_factors(E)

# set ids of environments with GxE effects
env_gxe_active = list(range(params['n_env_gxe']))

print('Setting up kinship matrix ...')
# create factors of kinship matrix
Lk = create_kinship_factors(create_kinship_matrix(
    params['n_individuals'], n_cells)).Lk

# create factors of E + K * E
us = env.U * env.S
Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

# compute QS if using fixed effect model
QS = None
if params['model'] == 'structlmm2_fixed':
    print('Factorizing K * EE ... ')
    QS = economic_qs((Lk @ Lk.T) * (env.E @ env.E.T))

# set variances
v = create_variances(params['r0'], params['v0'])


# (2) simulate data for each gene and run tests:

def sim_and_test(random: np.random.Generator):
    # simulates data for one gene and computes p-values

    s = simulate_data(
        offset=params['offset'],
        n_individuals=params['n_individuals'],
        n_snps=params['n_snps'],
        n_cells=n_cells,
        env=env,
        env_gxe_active=env_gxe_active,
        Ls=Ls,
        maf_min=params['maf_min'],
        maf_max=params['maf_max'],
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    # adjust likelihood model
    if params['likelihood'] == 'gaussian':
        y = s.y
    elif params['likelihood'] == 'negbin':
        mu = np.exp(params['offset'] + s.y_g + s.y_gxe + s.y_k + s.y_e)
        y = sample_nb(mu=mu, phi=params['nb_dispersion'], random=random)
        y = np.log(y + 1)
    elif params['likelihood'] == 'zinb':
        mu = np.exp(params['offset'] + s.y_g + s.y_gxe + s.y_k + s.y_e)
        y = sample_nb(mu=mu, phi=params['nb_dispersion'], random=random)
        y *= random.binomial(1, 1 - params['p_dropout'], size=y.size)
        y = np.log(y + 1)
    elif params['likelihood'] == 'poisson':
        lam = np.exp(params['offset'] + s.y_g + s.y_gxe + s.y_k + s.y_e)
        y = random.poisson(lam=lam)
        y = np.log(y + 1)
    else:
        raise ValueError('Unknown likelihood %s' % params['likelihood'])
    
    if params['normalize'] and params['likelihood'] != 'gaussian':
        y = quantile_gaussianize(y)

    # set up model & run test
    y = y.reshape(y.shape[0], 1)
    M = np.ones_like(y)
    
    if params['model'] == 'structlmm2':
        model = StructLMM2(
            y=y,
            W=M,
            E=None,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E,
            Ls=s.Ls)
        pv = model.scan_interaction(s.G)[0]
    elif params['model'] == 'structlmm':
        model = StructLMM2(
            y=y,
            W=M,
            E=None,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E)
        pv = model.scan_interaction(s.G)[0]
    elif params['model'] == 'structlmm2_fixed':
        pv = run_scstructlmm2_fixed(
            y=y,
            M=M,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E,
            G=s.G,
            QS=QS)
    else:
        raise ValueError('Unknown model %s' % params['model'])
    return pv

print('Running simulations for %d gene(s) ... ' % params['n_genes'])
threads = min(params['n_genes'], params['threads'])
if params['n_genes'] == 1:
    pvals = sim_and_test(random)
else:
    # set random state for each simulated gene
    random_state = random.integers(
        0, np.iinfo(np.int32).max,
        size=params['n_genes'])

    # run simulations in parallel
    pvals = Parallel(n_jobs=params['threads'])(
        delayed(sim_and_test)(np.random.default_rng(s)) for s in random_state)
print('Done.')

# save p-values
pd.DataFrame(pvals).to_csv(params['out_file'], header=False, index=False)

