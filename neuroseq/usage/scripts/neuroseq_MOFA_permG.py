import numpy as np
from numpy import ones
from numpy_sugar import ddot
import os
import sys
import pandas as pd
from pandas_plink import read_plink1_bin
from numpy.linalg import cholesky
from numpy_sugar.linalg import economic_svd
import xarray as xr
from limix.qc import quantile_gaussianize

from cellregmap._math import PMat, QSCov, ScoreStatistic
from cellregmap import CellRegMap, run_interaction

arg = {}


# gene index
arg["i"] = int(sys.argv[1])

# # SNP-gene index
arg["j"] = int(sys.argv[2])
seed = arg["j"]


revision_folder = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/REVISION/"

####### right away check if this was already run for this gene
# filter file (columns: snp_id, gene)
fvf_filename = revision_folder+"CRM_interaction_chr22/fvf.csv"
fvf = pd.read_csv(fvf_filename, index_col = 0)

genes = fvf['feature'].unique()
#print(genes)

gene_name = genes[arg["i"]]
trait_name = gene_name
#trait_name = re.sub("_.*","",gene_name)
print(gene_name)
print(trait_name)

fvf_gene = fvf[fvf['feature']==gene_name]
n = fvf_gene.shape[0]
print(n)

folder = revision_folder+"CRM_interaction_chr22/results_permG/"
outfilename = f"{folder}{trait_name}_{seed}.tsv"
print(outfilename)

if os.path.exists(outfilename):
    print("File already exists, exiting")
    sys.exit()

############################################
########## Sample mapping file #############
############################################

## this file will map cells to donors, it will also only including donors we have single cell data (a subset of all of HipSci donors)
sample_mapping_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/sample_mapping_file.csv"
sample_mapping = pd.read_csv(sample_mapping_file, dtype={"genotype_individual_id": str, "phenotype_sample_id": str})

## extract unique individuals
donors = sample_mapping["genotype_individual_id"].unique()
donors.sort()
print("Number of unique donors: {}".format(len(donors)))

############################################
############# Kinship matrix ###############
############################################

## read in GRM kinship matrix
kinship_folder="/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/"
kinship_file=kinship_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.kinship"
K = pd.read_csv(kinship_file, sep="\t", index_col=0)
assert all(K.columns == K.index)
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
plink_file = plink_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
G = read_plink1_bin(plink_file)

j=seed
fvf_sel = fvf_gene.iloc[j:(j+2)]

leads = fvf_sel[fvf_sel['feature']==gene_name]['snpID'].unique()
G_sel = G[:,G['snp'].isin(leads)]

#### to permute G, create shuffled index
# step 1 - shuffle G across donors (prior to expanding)
# step 2 - expand normally
# this is such as all cells from a given donor will keep the same genotype, but it will be that from another donor

random = np.random.RandomState(int(seed))
idx = random.permutation(G_sel.shape[0])
Idx = xr.DataArray(idx, dims=["sample"], coords = {"sample": G_sel.sample.values})
idx_G = Idx.sel(sample=sample_mapping["genotype_individual_id"].values)

############################################
##### expand from donors to cells ##########

# expand out genotypes from cells to donors (and select relevant donors in the same step)
G_expanded = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
assert all(hK_expanded.sample.values == G_expanded.sample.values)

print(G_expanded.shape)

######################################
########## Cell contexts #############
######################################
	
# cells by MOFA factors (20)
E_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/MOFA_20.csv"
E = pd.read_csv(E_file, index_col = 0)
E = xr.DataArray(E.values, dims=["cell", "pc"], coords={"cell": E.index.values, "pc": E.columns.values})
E = E.sel(cell=sample_mapping["phenotype_sample_id"].values)
assert all(E.cell.values == sample_mapping["phenotype_sample_id"].values)


# get eigendecomposition of EEt
[U, S, _] = economic_svd(E)
us = U * S
# get decomposition of K*EEt
Ls = [ddot(us[:,i], hK_expanded) for i in range(us.shape[1])]

#####################################
############ Phenotypes #############
#####################################

# Phenotype (meta-cell gene expression)
phenotype_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/phenotype.csv.pkl"
phenotype = pd.read_pickle(phenotype_file)

print("Phenotype shape BEFORE selection: {}".format(phenotype.shape))
phenotype = xr.DataArray(phenotype.values, dims=["trait", "cell"], coords={"trait": phenotype.index.values, "cell": phenotype.columns.values})
phenotype = phenotype.sel(cell=sample_mapping["phenotype_sample_id"].values)
print("Phenotype shape AFTER selection: {}".format(phenotype.shape))
assert all(phenotype.cell.values == sample_mapping["phenotype_sample_id"].values)


n_samples = phenotype.shape[1]
M = ones((n_samples, 1))

y = phenotype.sel(trait=trait_name)
y = quantile_gaussianize(y)
y = y.values.reshape(y.shape[0],1)

E_gauss = quantile_gaussianize(E)

# unpack G
GG = G_expanded.values

print("Running for gene {}".format(trait_name))
crm = CellRegMap(y=y, W=M, E=E_gauss.values[:,0:10], Ls=Ls)
# run association test using CellRegMap
pvals = crm.scan_interaction(G=GG, idx_G = idx_G)[0]

pv = pd.DataFrame({"chrom":G_expanded.chrom.values,
               "pv":pvals,
               "variant":G_expanded.snp.values})
pv.head()

pv.to_csv(outfilename)




















