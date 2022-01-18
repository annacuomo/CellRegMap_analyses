import pandas as pd
from pandas_plink import read_plink1_bin
import os

revision_folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"

# top SNP per gene file
myfile = revision_folder+"Marcs_results_qvalue.csv"
df = pd.read_csv(myfile, index_col=0)
print(df.shape)

T = 0.1
genes = df[df['q_value']<T]['feature_id'].unique()
print(len(genes))

# all results
myfile0 = revision_folder+"Marcs_results_all.csv"
df0 = pd.read_csv(myfile0)

## read in genotype file (plink format)
plink_folder = "/hps/nobackup/hipsci/scratch/genotypes/imputed/2017-03-27/Full_Filtered_SNPs_Plink/"
plink_file = plink_folder+"hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.bed"
G = read_plink1_bin(plink_file)

## select relevant samples
input_files_dir = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/new/input_files/"
sample_mapping_file = input_files_dir+"sample_mapping_file.csv"
sample_mapping = pd.read_csv(sample_mapping_file, dtype={"genotype_individual_id": str, "phenotype_sample_id": str})

samples = sample_mapping['genotype_individual_id'].unique()
print(len(samples))

G_sel = G[G['sample'].isin(samples),:]
print(G_sel.shape)

for gene in genes:

    out_filename = revision_folder+"LD_matrices_for_susie/"+str(gene)+".csv"
    
    if os.path.exists(out_filename):
        print("File already exists, skip gene")
        continue

    df1 = df0[df0['feature_id']==gene]
    snps = df1['snp_id'].unique()

    G_sel2 = G_sel[:,G_sel['snp'].isin(snps)]
    print(G_sel2.shape)

    GG = pd.DataFrame(G_sel2.values, columns = G_sel2.snp.values, index = G_sel2.sample.values)

    ## compute correlations (LD matrix)
    R = GG.corr()
    print(R.shape)

    R.to_csv(out_filename)