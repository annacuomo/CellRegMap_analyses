import re
import pandas as pd
import xarray as xr
from numpy import ones
from numpy.linalg import cholesky
from pandas_plink import read_plink1_bin
from limix.qc import quantile_gaussianize

from cellregmap import run_association

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
endo_eqtl_file = input_files_dir+"endodiff_eqtl_allconditions_FDR10pct.csv"
endo_eqtl = pd.read_csv(endo_eqtl_file, index_col = False)
endo_eqtl["chrom"] = [int(i[:i.find("_")]) for i in endo_eqtl["snp_id"]]
endo_eqtl.head(2)

chrom = 19
# consider genes on that chromosome
genes = endo_eqtl[endo_eqtl['chrom']==int(chrom)]['feature'].unique()

#########################################################
# cis window around a specific gene (discovery)

def cis_snp_selection(feature_id, annotation_df, G, window_size):
    feature = annotation_df.query("feature_id==\"{}\"".format(feature_id)).squeeze()
    chrom = str(feature['chromosome'])
    start = feature['start']
    end = feature['end']
    # make robust to features self-specified back-to-front
    lowest = min([start,end])
    highest = max([start,end])
    # for cis, we sequentially add snps that fall within each region
    G = G.where((G.chrom == str(chrom)) & (G.pos > (lowest-window_size)) & (G.pos < (highest+window_size)), drop=True)
    return G

# (1) gene name (feature_id)
gene_name = genes[0]
trait_name = re.sub("_.*","",gene_name)
trait_name

# (2) annotation linking gene to genomic position
annotation_file = "/hps/nobackup/hipsci/scratch/processed_data/rna_seq/annotationFiles/Ensembl_75_Limix_Annotation_FC_Gene.txt"
anno_df = pd.read_csv(annotation_file, sep="\t", index_col=0)
anno_df.head(2)

# (3) window size (cis)
w = 100000

G_sel = cis_snp_selection(trait_name, anno_df, G, w)
G_sel.shape

############################################
##### expand from donors to cells ##########

# expand out genotypes from cells to donors (and select relevant donors in the same step)
G_expanded = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
assert all(hK_expanded.sample.values == G_expanded.sample.values)

G_expanded.shape

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

# select gene
y = phenotype.sel(trait=gene_name)
# quantile normalise
y = quantile_gaussianize(y)

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

# just an interceot in this case
n_cells = phenotype.shape[1]
W = ones((n_cells, 1))

out_filename = out_dir + str(trait_name)

# run association test using CellRegMap
pvals = run_association(y.values, W, C.values[:,0:10], G=G_expanded, hK=hK_expanded)[0]

pv = pd.DataFrame({"chrom":G_expanded.chrom.values,
               "pv":pvals,
               "variant":G_expanded.snp.values})
pv.head()

pv.to_csv(out_filename)