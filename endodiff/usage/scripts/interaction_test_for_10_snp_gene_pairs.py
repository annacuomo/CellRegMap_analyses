import re
import os
import sys
import time
import pandas as pd
import xarray as xr
from numpy import ones
from numpy.linalg import cholesky
from pandas_plink import read_plink1_bin
from limix.qc import quantile_gaussianize

from cellregmap import run_interaction

arg = {}

# gene index
arg["i"] = int(sys.argv[1])

# SNP-gene index
arg["j"] = int(sys.argv[2])
j = arg["j"]

revision_folder = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"

####### right away check if this was already run for this gene
# filter file (columns: snp_id, gene)
#fvf_filename = revision_folder+"/CRM_interaction_chr22/fvf.csv"
#fvf_filename = revision_folder+"/CRM_interaction_chr21/fvf.csv"
fvf_filename = revision_folder+"/CRM_interaction_chr20/fvf.csv"
fvf = pd.read_csv(fvf_filename, index_col = 0)
#print(fvf.head())

genes = fvf['feature'].unique()
#print(genes)

gene_name = genes[arg["i"]]
trait_name = re.sub("_.*","",gene_name)
print(gene_name)
print(trait_name)

fvf_gene = fvf[fvf['feature']==gene_name]

folder = revision_folder+"CRM_interaction_chr22/results/"
outfilename = f"{folder}{trait_name}_{j}.tsv"
print(outfilename)

if os.path.exists(outfilename):
    print("File already exists, exiting")
    sys.exit()


# input files directory
input_files_dir = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/new/input_files/"

############################################
########## Sample mapping file #############
############################################

## this file will map cells to donors 
## it will also only include donors we have single-cell data for (a subset of all of HipSci donors)
sample_mapping_file = input_files_dir+"sample_mapping_file.csv"
sample_mapping = pd.read_csv(sample_mapping_file, dtype={"genotype_individual_id": str, "phenotype_sample_id": str})

## genotype_individual_id are donor IDs, as found in the genotype matrix (G) and GRM covariance (K)
## phenotype_sample_id are cell IDs, as found in the scRNA-seq phenotype vector (y) and cell context covariance (C)
sample_mapping.head()

## extract unique individuals
donors = sample_mapping["genotype_individual_id"].unique()
donors.sort()
print("Number of unique donors: {}".format(len(donors)))

############################################
############# Kinship matrix ###############
############################################

## read in GRM (genotype relationship matrix; kinship matrix)
#kinship_folder="/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink-F/"
kinship_folder = "/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/"
kinship_file=kinship_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.kinship"
K = pd.read_csv(kinship_file, sep="\t", index_col=0)
assert all(K.columns == K.index) #symmetric matrix, donors x donors

K = xr.DataArray(K.values, dims=["sample_0", "sample_1"], coords={"sample_0": K.columns, "sample_1": K.index})
K = K.sortby("sample_0").sortby("sample_1")
donors = sorted(set(list(K.sample_0.values)).intersection(donors))
print("Number of donors after kinship intersection: {}".format(len(donors)))

## subset to relevant donors
K = K.sel(sample_0=donors, sample_1=donors)
assert all(K.sample_0 == donors)
assert all(K.sample_1 == donors)

## and decompose such as K = hK @ hK.T (using Cholesky decomposition)
hK = cholesky(K.values)
hK = xr.DataArray(hK, dims=["sample", "col"], coords={"sample": K.sample_0.values})
assert all(hK.sample.values == K.sample_0.values)

del K
print("Sample mapping number of rows BEFORE intersection: {}".format(sample_mapping.shape[0]))
## subsample sample mapping file to donors in the kinship matrix
sample_mapping = sample_mapping[sample_mapping["genotype_individual_id"].isin(donors)]
print("Sample mapping number of rows AFTER intersection: {}".format(sample_mapping.shape[0]))

############################################
##### expand from donors to cells ##########

## use sel from xarray to expand hK (using the sample mapping file)
hK_expanded = hK.sel(sample=sample_mapping["genotype_individual_id"].values)
assert all(hK_expanded.sample.values == sample_mapping["genotype_individual_id"].values)

#####################################
############ Genotypes ##############
#####################################

## read in genotype file (plink format)
plink_folder = "/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/"
#plink_folder = "/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink/"
plink_file = plink_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
G = read_plink1_bin(plink_file)

######################################
########## Cell contexts #############
######################################

# cells by MOFA factors (20)
C_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/mofa_logcounts_model_factors.csv"
C = pd.read_csv(C_file, index_col = 0)
C = xr.DataArray(C.values, dims=["cell", "pc"], coords={"cell": C.index.values, "pc": C.columns.values})
C = C.sel(cell=sample_mapping["phenotype_sample_id"].values)
assert all(C.cell.values == sample_mapping["phenotype_sample_id"].values)

# quantile normalise cell contexts
C_gauss = quantile_gaussianize(C)

#####################################
############ Phenotypes #############
#####################################

# Phenotype (single-cell expression)
phenotype_file = input_files_dir+"phenotype.csv.pkl"
phenotype = pd.read_pickle(phenotype_file)
print("Phenotype shape BEFORE selection: {}".format(phenotype.shape))
phenotype = xr.DataArray(phenotype.values, dims=["trait", "cell"], coords={"trait": phenotype.index.values, "cell": phenotype.columns.values})
phenotype = phenotype.sel(cell=sample_mapping["phenotype_sample_id"].values)
print("Phenotype shape AFTER selection: {}".format(phenotype.shape))
assert all(phenotype.cell.values == sample_mapping["phenotype_sample_id"].values)

fvf_sel = fvf_gene.iloc[j:(j+10)]

# SNP selection
leads = fvf_sel[fvf_sel['feature']==gene_name]['snpID'].unique()
G_sel = G[:,G['snp'].isin(leads)]

############################################
##### expand from donors to cells ##########

# expand out genotypes from cells to donors (and select relevant donors in the same step)
G_expanded = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
assert all(hK_expanded.sample.values == G_expanded.sample.values)

print(G_expanded.shape)

# select gene
y = phenotype.sel(trait=gene_name)
# quantile normalise
y = quantile_gaussianize(y)
# reshape
y = y.values.reshape(y.shape[0],1)
print(y.shape)

######################################
############ Covariates ##############
######################################

# just an intercept in this case
n_cells = phenotype.shape[1]
W = ones((n_cells, 1))

# unpack G
GG = G_expanded.values

print("Running for gene {}".format(trait_name))

# run association test using CellRegMap
pvals = run_interaction(y=y, W=W, E=C_gauss.values[:,0:10], E1=C_gauss.values[:,0:10], E2=C.values[:,0:20], G=GG, hK=hK_expanded)[0]

pv = pd.DataFrame({"chrom":G_expanded.chrom.values,
               "pv":pvals,
               "variant":G_expanded.snp.values})
pv.head()

pv.to_csv(outfilename)
