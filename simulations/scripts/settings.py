#===============================================================================
# Global variables & default simulation parameters
#===============================================================================
import os


DEFAULT_PARAMS = {
    'n_snps': 25,
    'n_individuals': 100,
    'n_genes': 1,
    'cells_per_individual': 'fixed', # either 'fixed' or 'variable'
    'n_cells': 50, # cells per individual if cells_per_individual = 'fixed'
    'maf_min': 0.2, # minimum minor allele frequency
    'maf_max': .45, # maximum minor allele frequency
    'env': 'endo', # 'endo', 'cluster_uniform' or 'cluster_biased;
    'n_env': 20, # number of environments
    'n_env_gxe': 10, # environments 1:n_env_gxe with active GxE effects
    'n_env_tested': 10, # test 1:n_env_tested
    'offset': 2.5,
    'n_causal_g': 1, # number of SNPs with persistent genetic effect
    'n_causal_gxe': 1, # number of SNPs with GxE effect
    'n_causal_shared': 0, # number of SNPs with both persistent and GxE effects
    'r0': .5, # fraction of genetic variance explained by gxe
    'v0': .05, # total variance explained by genetics
    'likelihood': 'gaussian', # simulated likelihood model; 'negbin', 'gaussian', or 'zinb'
    'nb_dispersion': 1.5, # dispersion parameter for negative binomial
    'p_dropout': 0.05, # dropout probability zero-inflated negative binomial
    'normalize': True, # apply quantilze normalization before testing
    'dirichlet_alpha': 2, # prior for cell to cluster assignment if env = 'cluster_biased'
    'seed': 124823,
    'model': 'structlmm2' # test to run
}

# Locations of the Endoderm PC and meta-data files.
ENDO_PCS_PATH = os.path.dirname(__file__) +  '/../data/endodiff_100PCs.csv.zip'
ENDO_META_PATH = os.path.dirname(__file__) +  '/../data/cell_metadata_cols.tsv'
