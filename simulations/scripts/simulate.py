#===============================================================================
# Simulation script
#===============================================================================
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from struct_lmm2 import StructLMM2, create_variances

from settings import DEFAULT_PARAMS
from sim_utils import (
    set_causal_snps,
    set_cells_per_donor,
    generate_environment_matrix,
    simulate_data
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
# Run tests
#===============================================================================
# initialize random number generator
random = np.random.default_rng(params['seed'])

# set indices of causal SNPs
g_causals, gxe_causals = set_causal_snps(
    params['n_causal_g'], params['n_causal_gxe'])

# set cells per donor
n_cells = set_cells_per_donor(
    params['cells_per_individual'], params['n_individuals'])

# create environment matrix and decomposition
env = generate_environment_matrix(
    params['env'], params['d_env'], n_cells, params['n_individuals'], random)

# set variances
v = create_variances(params['r0'], params['v0'])

print('Running simulations for %d genes ... ' % params['n_genes'])
def sim_and_test(random: np.random.Generator):
    s = simulate_data(
        offset=params['offset'],
        n_individuals=params['n_individuals'],
        n_snps=params['n_snps'],
        n_cells=n_cells,
        env=env,
        maf_min=params['maf_min'],
        maf_max=params['maf_max'],
        g_causals=g_causals,
        gxe_causals=gxe_causals,
        variances=v,
        random=random,
    )

    # set up model
    M = np.ones((s.y.shape[0], 1)) # covariates
    y = s.y.reshape(s.y.shape[0],1)
    slmm2 = StructLMM2(y, M, s.E, s.Ls)
    pv = slmm2.scan_interaction(s.G)
    return pv[0]

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

