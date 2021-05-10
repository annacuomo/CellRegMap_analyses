#===============================================================================
# Global variables & default simulation parameters
#===============================================================================
import os


DEFAULT_PARAMS = {
    'n_snps': 25,
    'n_individuals': 100,
    'n_genes': 100,
    'cells_per_individual': 'fixed',
    'maf_min': 0.2,
    'maf_max': .45,
    'env': 'endo',
    'd_env': 10,
    'offset': 2.5,
    'n_causal_g': 1,
    'n_causal_gxe': 1,
    'n_causal_shared': 0,
    'r0': .2, # fraction of genetic variance explained by gxe
    'v0': .1, # total variance explained by genetics
    'likelihood': 'gaussian',
    'nb_dispersion': 1.5,
    'p_dropout': 0.05,
    'normalize': False,
    'dirichlet_alpha': 5,
    'seed': 124823,
}

# Locations of the Endoderm PC and meta-data files.
ENDO_PCS_PATH = os.path.dirname(__file__) +  '/../data/endodiff_100PCs.csv.zip'
ENDO_META_PATH = os.path.dirname(__file__) +  '/../data/cell_metadata_cols.tsv'
