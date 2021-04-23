#===============================================================================
# Global variables & default simulation parameters
#===============================================================================
import os


DEFAULT_PARAMS = {
    'maf_min': 0.2,
    'maf_max': .45,
    'n_snps': 25,
    'n_individuals': 100,
    'n_genes': 50,
    'cells_per_individual': 'fixed',
    'env': 'endo',
    'd_env': 10,
    'offset': .3,
    'n_causal_g': 1,
    'n_causal_gxe': 1,
    'r0': .5,
    'v0': .5,
    'seed': 123,
}

ENDO_PATH = os.path.dirname(__file__) +  '/../data/endodiff_100PCs.csv.zip'