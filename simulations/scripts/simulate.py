#===============================================================================
# Simulation script
#===============================================================================
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

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
    sample_nb
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
# Simulate data & run tests
#===============================================================================
# initialize random number generator
random = np.random.default_rng(params['seed'])

# set indices of causal SNPs
(g_causals, gxe_causals) =  set_causal_ids(
    n_causal_g=params['n_causal_g'],
    n_causal_gxe=params['n_causal_gxe'],
    n_causal_shared=params['n_causal_shared'])

# set cells per donor
if params['cells_per_individual'] == 'fixed':
    n_cells = 50
elif params['cells_per_individual'] == 'variable':
    n_cells = np.arange(params['n_individuals']) + 1
else:
    raise ValueError(
        'Invalid cells_per_individual value: %s' % params['cells_per_individual'])

print('Setting up environment matrix ...')
# create environment matrix and decomposition
if params['env'] == 'endo':
    E = sample_endo(params['d_env'],
        params['n_individuals'],
        n_cells,
        random)
elif params['env'] == 'cluster_uniform':
    E = sample_clusters(
        params['d_env'],
        params['n_individuals'],
        n_cells,
        random)
elif params['env'] == 'cluster_biased':
    E = sample_clusters(
        params['d_env'],
        params['n_individuals'],
        n_cells,
        random,
        params['dirichlet_alpha'])
else:
    raise ValueError('Invalid env value: %s' % params['env'])
env = create_environment_factors(E)

print('Setting up kinship matrix ...')
# create kinship matrix
Lk = create_kinship_factors(create_kinship_matrix(
    params['n_individuals'], n_cells)).Lk

# set variances
v = create_variances(params['r0'], params['v0'])

print('Running simulations for %d gene(s) ... ' % params['n_genes'])
def sim_and_test(random: np.random.Generator):
    s = simulate_data(
        offset=params['offset'],
        n_individuals=params['n_individuals'],
        n_snps=params['n_snps'],
        n_cells=n_cells,
        env=env,
        Lk=Lk,
        maf_min=params['maf_min'],
        maf_max=params['maf_max'],
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

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
    
    if params['normalize']:
        y = quantile_gaussianize(y)

    # set up model
    y = y.reshape(y.shape[0], 1)
    M = np.ones_like(y)
    slmm2 = StructLMM2(y, M, env.E, s.Ls)
    pv = slmm2.scan_interaction(s.G)
    return pv[0]

threads = min(params['n_genes'], params['threads'])
if threads == 1:
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

