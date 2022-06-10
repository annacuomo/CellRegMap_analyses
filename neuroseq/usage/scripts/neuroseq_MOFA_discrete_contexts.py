import numpy as np
from numpy import ones
from numpy_sugar import ddot
import re
import os
import sys
import pandas as pd
from pandas_plink import read_plink1_bin
from numpy.linalg import cholesky
from numpy_sugar.linalg import economic_svd
import xarray as xr
from cellregmap import run_interaction
from limix.qc import quantile_gaussianize

arg = {}

# chrom
arg["chrom"] = str(sys.argv[1])


## this file will map cells to donors, it will also only including donors we have single cell data (a subset of all of HipSci donors)
sample_mapping_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/sample_mapping_file.csv"
sample_mapping = pd.read_csv(sample_mapping_file, dtype={"genotype_individual_id": str, "phenotype_sample_id": str})

## extract unique individuals
donors = sample_mapping["genotype_individual_id"].unique()
donors.sort()
print("Number of unique donors: {}".format(len(donors)))

# read in genotypes
plink_folder = "/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/" 
plink_file = plink_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
G = read_plink1_bin(plink_file)

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

## and decompose such as K = hK @ hK.T
hK = cholesky(K.values)
hK = xr.DataArray(hK, dims=["sample", "col"], coords={"sample": K.sample_0.values})
assert all(hK.sample.values == K.sample_0.values)
del K

print("Sample mapping number of rows BEFORE intersection: {}".format(sample_mapping.shape[0]))
sample_mapping = sample_mapping[sample_mapping["genotype_individual_id"].isin(donors)]
print("Sample mapping number of rows AFTER intersection: {}".format(sample_mapping.shape[0]))

############################################
##### expand from donors to cells ##########

## use sel from xarray to expand hK (using the sample mapping file)
hK_expanded = hK.sel(sample=sample_mapping["genotype_individual_id"].values)
assert all(hK_expanded.sample.values == sample_mapping["genotype_individual_id"].values)

# environments
# cells by discrete clusters based on MOFA factors (18)
E_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/REVISION/C_discrete_18clusters.csv"
E = pd.read_csv(E_file, index_col = 0)
E = xr.DataArray(E.values, dims=["cell", "pc"], coords={"cell": E.index.values, "pc": E.columns.values})
E = E.sel(cell=sample_mapping["phenotype_sample_id"].values)
assert all(E.cell.values == sample_mapping["phenotype_sample_id"].values)


# Phenotype (meta-cell gene expression)
phenotype_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/phenotype.csv.pkl"
phenotype = pd.read_pickle(phenotype_file)

print("Phenotype shape BEFORE selection: {}".format(phenotype.shape))
phenotype = xr.DataArray(phenotype.values, dims=["trait", "cell"], coords={"trait": phenotype.index.values, "cell": phenotype.columns.values})
phenotype = phenotype.sel(cell=sample_mapping["phenotype_sample_id"].values)
print("Phenotype shape AFTER selection: {}".format(phenotype.shape))
assert all(phenotype.cell.values == sample_mapping["phenotype_sample_id"].values)

#breakpoint()

# Filter on specific gene-SNP pairs
# eQTL from neuroseq DA (day30 + day52 + day52 ROT treated)
neuro_eqtl_file = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/DA_eqtl_allconditions_FDR5pct.csv" # consider filter further (significant only)
neuro_eqtl = pd.read_csv(neuro_eqtl_file)
neuro_eqtl["chrom"] = [int(i[:i.find("_")]) for i in neuro_eqtl["snp_id"]]
genes = neuro_eqtl[neuro_eqtl['chrom']==int(arg["chrom"])]['feature'].unique()

n_samples = phenotype.shape[1]
M = ones((n_samples, 1))

outdir = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/REVISION/CRM_interaction_discrete_contexts/18clusters/"

for trait_name in genes:
	gene_name = re.sub("-",".",trait_name)
	outfilename = outdir + str(gene_name) + ".tsv"
	if os.path.exists(outfilename):
		print("File already exists, exiting")
		continue
	leads = neuro_eqtl[neuro_eqtl['feature']==trait_name]['snp_id'].unique()
	G_sel = G[:,G['snp'].isin(leads)]
	G_expanded = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
	assert all(hK_expanded.sample.values == G_expanded.sample.values)

	trait_name = re.sub("-",".",trait_name)
	y = phenotype.sel(trait=trait_name)
	y = quantile_gaussianize(y)
	y = y.values.reshape(y.shape[0],1)
	n_cells = phenotype.shape[1]
	W = ones((n_cells, 1))
	GG = G_expanded.values
	print("Running for gene {}".format(trait_name))
	pvals = run_interaction(y=y, W=W, E=E.values[:,0:10], E1=E.values[:,0:10], E2=E.values[:,0:20], G=GG, hK=hK_expanded)[0]
	pv = pd.DataFrame({"chrom":G_expanded.chrom.values,
	   "pv":pvals,
	   "variant":G_expanded.snp.values})
	pv.to_csv(outfilename)

