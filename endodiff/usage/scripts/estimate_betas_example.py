import os
import re
import pandas as pd
import xarray as xr
from numpy import ones
from numpy.linalg import cholesky
from pandas_plink import read_plink1_bin
from limix.qc import quantile_gaussianize

from cellregmap import estimate_betas

# input files directory
input_files_dir = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/new/input_files/"

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
kinship_folder="/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink-F/"
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
plink_folder = "/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink/"
plink_file = plink_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
G = read_plink1_bin(plink_file)

#############################
###### SNP selection

# filter file (columns: snp_id, gene)
revision_folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"
fvf_filename = revision_folder+"fvf_new_outliers.csv"
fvf = pd.read_csv(fvf_filename, index_col = False)

genes = fvf['feature'].unique()

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

######################################
########## Cell contexts #############
######################################

# cells by MOFA factors (20)
C_file = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/mofa_logcounts_model_factors.csv"
C = pd.read_csv(C_file, index_col = 0)
C = xr.DataArray(C.values, dims=["cell", "pc"], coords={"cell": C.index.values, "pc": C.columns.values})
C = C.sel(cell=sample_mapping["phenotype_sample_id"].values)
assert all(C.cell.values == sample_mapping["phenotype_sample_id"].values)

# quantile normalise cell contexts
C = quantile_gaussianize(C)

######################################
############ Covariates ##############
######################################

n_cells = phenotype.shape[1]
W = ones((n_cells, 1))

######################################
############ Run and save ############
######################################

out_dir = revision_folder+"CRM_association_outliers_betas/"

#breakpoint()

for gene_name in genes:
    # gene name (feature_id)
    trait_name = re.sub("_.*","",gene_name)
    out_filename = out_dir + str(trait_name)
    outfilename_betaGxC = out_filename+"_betaGxC.csv"
    if os.path.exists(outfilename_betaGxC):
        print("File already exists, skip gene")
        continue

    # select gene
    y = phenotype.sel(trait=gene_name)
    # quantile normalise
    y = quantile_gaussianize(y)

    ## select eQTLs for that gene only (from filter file)
    leads = fvf[fvf['feature']==gene_name]['snp_id'].unique()
    G_sel = G[:,G['snp'].isin(leads)]

    # expand out genotypes from cells to donors (and select relevant donors in the same step)
    G_expanded = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
    assert all(hK_expanded.sample.values == G_expanded.sample.values)

    # run association test using CellRegMap
    betas = estimate_betas(y.values, W, C.values[:,0:10], G=G_expanded, hK=hK_expanded)
    
    beta_G = betas[0]
    beta_GxC = betas[1][0]
    
    beta_G_df = pd.DataFrame({"chrom":G_expanded.chrom.values,
               "betaG":beta_G,
               "variant":G_expanded.snp.values})

    beta_G_df.to_csv(out_filename+"_betaG.csv")

    cells = phenotype["cell"].values
    snps = G_expanded["variant"].values

    beta_GxC_df = pd.DataFrame(data = beta_GxC, columns = snps, index = cells)
    beta_GxC_df.to_csv(outfilename_betaGxC)
