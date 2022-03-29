"""Paths and simulation settings."""
import os

# locations of the endoderm data and meta-data files.
DATA_DIR = os.path.dirname(__file__) + "/../data"

HIPSCI_GENO_PATH = (
    "/omics/groups/OE0540/internal/projects/HipSci/openAccess/Genotypes/Imputed"
    "/Rel_2018_01/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20180102."
    "genotypes.norm.renamed.recode.openAcces.bed"
)

HIPSCI_KINSHIP_PATH = (
    "/omics/groups/OE0540/internal/projects/HipSci/openAccess/Genotypes/Imputed"
    "/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327."
    "genotypes.norm.renamed.recode.filtered.rel"
)

N_DONORS = 50  # number of donors to include
N_CELLS = 100  # cells per donor
N_GENES = 500  # number of genes to simulate
N_SNPS = 1  # number of SNPs per gene

GENETIC_VAR = [0.025]
# GENETIC_VAR = [0.01, 0.025]

FEV_GXC = [0.0, 0.25, 0.5, 0.75, 1.0]
FEV_GXC_DEFAULT = 0.5

NUM_CONTEXTS = [2, 5, 10, 15, 20]
NUM_CONTEXTS_DEFAULT = 10

NUM_TESTED_DEFAULT = 10
NUM_TESTED = NUM_CONTEXTS
