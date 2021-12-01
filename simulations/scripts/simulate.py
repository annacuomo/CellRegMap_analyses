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
import time

from numpy_sugar import ddot
from numpy_sugar.linalg import economic_qs

from limix.qc import quantile_gaussianize
from cellregmap import CellRegMap

from settings import DEFAULT_PARAMS, FILTERED_KINSHIP_PATH
from sim_utils import (
    create_variances,
    set_causal_ids,
    sample_clusters,
    sample_endo,
    create_environment_factors,
    # create_kinship_matrix,
    create_kinship_factors,
    simulate_data,
    sample_nb,
    run_cellregmap_fixed
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
params['out_prefix'] = snakemake.output[0] if SNAKEMODE else 'results'


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
    # n_cells = np.arange(params['n_individuals']) + 1
    # make sure n_cells.sum() equals params['n_cells'] * params['n_individuals']
    n = params['n_individuals']
    n_cells = np.linspace(1, n, n) * 2 * params['n_cells'] / (n + 1)
    n_cells = n_cells.round().astype(int)
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
# env_gxe_active = list(range(params['n_env_gxe']))
env_gxe_active = random.choice(E.shape[1], params['n_env_gxe'])

print('Setting up kinship matrix ...')
# create factors of kinship matrix
if params['real_genotypes']:
    if params['n_individuals'] > 100:
        raise ValueError('n_individuals > 100. Use artificial genotypes.')
    K = pd.read_csv(FILTERED_KINSHIP_PATH, index_col=0)
    donor_ids = K.index[:params['n_individuals']].tolist()
    K = K.iloc[:params['n_individuals'], :params['n_individuals']].to_numpy()
else:
    K = np.eye(params['n_individuals'])
    donor_ids = None
Lk = create_kinship_factors(K).Lk
# expand from donors to cells
Lk = Lk[np.repeat(range(params['n_individuals']), n_cells), :]

# create factors of E + K * E
us = env.U * env.S
Ls = tuple([ddot(us[:, i], Lk) for i in range(us.shape[1])])

# compute QS if using fixed effect model
QS = None
if params['model'] == 'structlmm2_fixed':
    print('Factorizing K * EE ... ')
    QS = economic_qs((Lk @ Lk.T) * (env.E @ env.E.T))

# set variances
v = create_variances(
    r0=params['r0'],
    v0=params['v0'],
    include_noise=params['likelihood'] == 'gaussian'
)


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
        real_genotypes=params['real_genotypes'],
        donor_ids=donor_ids,
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
    
    t_start = time.time()
    if params['model'] == 'cellregmap':
        model = CellRegMap(
            y=y,
            W=M,
            E=None,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E,
            Ls=s.Ls)
        pv = model.scan_interaction(s.G)[0]
    elif params['model'] == 'structlmm':
        model = CellRegMap(
            y=y,
            W=M,
            E=None,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E)
        pv = model.scan_interaction(s.G)[0]
    elif params['model'] == 'cellregmap_fixed':
        pv = run_cellregmap_fixed(
            y=y,
            M=M,
            E0=env.E[:, :params['n_env_tested']],
            E1=env.E,
            G=s.G,
            QS=QS)
    else:
        raise ValueError('Unknown model %s' % params['model'])
    time_elapsed = time.time() - t_start
    return pv, s.donor_ids, s.snp_ids, time_elapsed

print('Running simulations for %d gene(s) ... ' % params['n_genes'])
threads = min(params['n_genes'], params['threads'])
if params['n_genes'] == 1:
    pvals, times = sim_and_test(random)
    times = [times]
else:
    # set random state for each simulated gene
    random_state = random.integers(
        0, np.iinfo(np.int32).max,
        size=params['n_genes'])

    # run simulations in parallel
    results = Parallel(n_jobs=params['threads'])(
        delayed(sim_and_test)(np.random.default_rng(s)) for s in random_state)
    pvals = [r[0] for r in results]
    donor_ids = [r[1] for r in results]
    snp_ids = [r[2] for r in results]
    times = [r[3] for r in results]
print('Done.')

# save
pd.DataFrame(pvals).to_csv(params['out_prefix'] + '_pvals.txt', header=False, index=False)
pd.DataFrame(donor_ids).to_csv(params['out_prefix'] + '_donor_ids.txt', header=False, index=False)
pd.DataFrame(snp_ids).to_csv(params['out_prefix'] + '_snp_ids.txt', header=False, index=False)
pd.DataFrame(times).to_csv(params['out_prefix'] + '_runtime.txt', header=False, index=False)

