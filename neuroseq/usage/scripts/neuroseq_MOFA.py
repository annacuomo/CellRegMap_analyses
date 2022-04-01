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
from struct_lmm2 import StructLMM2
from limix.qc import quantile_gaussianize

arg = {}

# chrom
arg["chrom"] = str(sys.argv[1])

# Possible values: yes, no
arg["perm"] = sys.argv[2]
assert arg["perm"] in ["yes", "no"]

if len(sys.argv) > 3:
    assert arg["perm"] == "yes"
    arg["seed"] = int(sys.argv[3])

def get_slmm2_G():

	## this file will map cells to donors, it will also only including donors we have single cell data (a subset of all of HipSci donors)
	sample_mapping_file = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/sample_mapping_file.csv"
	sample_mapping = pd.read_csv(sample_mapping_file, dtype={"genotype_individual_id": str, "phenotype_sample_id": str})

	## extract unique individuals
	donors = sample_mapping["genotype_individual_id"].unique()
	donors.sort()
	print("Number of unique donors: {}".format(len(donors)))

	## read in genotype file
	plink_file = "/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
    # to do try file below
    #plink_file = "/hps/nobackup/hipsci/scratch/genotypes/imputed/REL-2018-01/Full_Plink/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20180102.genotypes.norm.renamed.recode.vcf.gz.bed"
	G = read_plink1_bin(plink_file)

	## read in GRM kinship matrix
	kinship_file = "/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink-F/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.kinship"
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

	## and decompose such as K = L @ L.T
	L_kinship = cholesky(K.values)
	L_kinship = xr.DataArray(L_kinship, dims=["sample", "col"], coords={"sample": K.sample_0.values})
	assert all(L_kinship.sample.values == K.sample_0.values)
	del K

	print("Sample mapping number of rows BEFORE intersection: {}".format(sample_mapping.shape[0]))
	sample_mapping = sample_mapping[sample_mapping["genotype_individual_id"].isin(donors)]
	print("Sample mapping number of rows AFTER intersection: {}".format(sample_mapping.shape[0]))

	# expand from donors to cells
	L_expanded = L_kinship.sel(sample=sample_mapping["genotype_individual_id"].values)
	assert all(L_expanded.sample.values == sample_mapping["genotype_individual_id"].values)

	# environments
	# cells by MOFA factors (20)
	E_file = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/MOFA_20.csv"
	E = pd.read_csv(E_file, index_col = 0)
	E = xr.DataArray(E.values, dims=["cell", "pc"], coords={"cell": E.index.values, "pc": E.columns.values})
	E = E.sel(cell=sample_mapping["phenotype_sample_id"].values)
	assert all(E.cell.values == sample_mapping["phenotype_sample_id"].values)

	# subselect to only SNPs on right chromosome
	G_sel = G.where(G.chrom == str(arg["chrom"]), drop=True)
	# and to individuals in smf
	G_exp = G_sel.sel(sample=sample_mapping["genotype_individual_id"].values)
	assert all(L_expanded.sample.values == G_exp.sample.values)
	return G_exp, sample_mapping, E, L_expanded

G_exp, sample_mapping, E, L_expanded = get_slmm2_G()

#breakpoint()

#n_factors = 15
# E = E.values[:,0:n_factors]

# get eigendecomposition of EEt
[U, S, _] = economic_svd(E)
us = U * S
# get decomposition of K*EEt
Ls = [ddot(us[:,i], L_expanded) for i in range(us.shape[1])]

# Phenotype (meta-cell gene expression)
phenotype_file = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/phenotype.csv.pkl"
phenotype = pd.read_pickle(phenotype_file)

print("Phenotype shape BEFORE selection: {}".format(phenotype.shape))
phenotype = xr.DataArray(phenotype.values, dims=["trait", "cell"], coords={"trait": phenotype.index.values, "cell": phenotype.columns.values})
phenotype = phenotype.sel(cell=sample_mapping["phenotype_sample_id"].values)
print("Phenotype shape AFTER selection: {}".format(phenotype.shape))
assert all(phenotype.cell.values == sample_mapping["phenotype_sample_id"].values)

#breakpoint()

# Filter on specific gene-SNP pairs
# eQTL from neuroseq DA (day30 + day52 + day52 ROT treated)
neuro_eqtl_file = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/input_files/DA_eqtl_allconditions_FDR5pct.csv" # consider filter further (significant only)
neuro_eqtl = pd.read_csv(neuro_eqtl_file)
neuro_eqtl["chrom"] = [int(i[:i.find("_")]) for i in neuro_eqtl["snp_id"]]
genes = neuro_eqtl[neuro_eqtl['chrom']==int(arg["chrom"])]['feature'].unique()

n_samples = phenotype.shape[1]
M = ones((n_samples, 1))

for trait_name in genes:

    folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/MOFA/MOFA10"

    if arg["perm"] == "yes":
        outfilename = f"{folder}/{trait_name}_perm{arg['seed']}.tsv"
    else:
        outfilename = f"{folder}/{trait_name}.tsv"

    print(outfilename)

    if os.path.exists(outfilename):
        print("File already exists, skipping gene")
        continue

    y = phenotype.sel(trait=trait_name)
    y = quantile_gaussianize(y)
    y = np.asarray(y)

    E = quantile_gaussianize(E)

    # null model
    slmm2 = StructLMM2(y, M, E[:,0:10], Ls)
    leads = neuro_eqtl[neuro_eqtl['feature']==trait_name]['snp_id'].unique()
    G_tmp = G_exp[:,G_exp['snp'].isin(leads)]

    ok = np.var(G_tmp.values, axis=0) > 0.0
    pvals = np.full(G_tmp.shape[1], np.nan)

    if arg["perm"] == "yes":
        perm = arg["seed"]
    else:
        perm = None

    try:
        pvals[ok] = slmm2.scan_interaction(G_tmp[:, ok], perm)[0]
        pv = pd.DataFrame({"chrom":G_tmp.chrom.values,
            "pv":pvals,
            "variant":G_tmp.snp.values})

        pv.to_csv(outfilename, sep='\t')

    except: continue





















