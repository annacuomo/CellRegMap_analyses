import os
import pandas as pd

if __name__ == '__main__':

    revision_folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"
    fvf_filename = revision_folder+"/CRM_interaction_chr22/fvf.csv"
    fvf = pd.read_csv(fvf_filename, index_col = 0)

    genes = fvf['feature'].unique()

    print("prepare job to submit")
    bsub = "bsub -R \"rusage[mem=80000]\" -M 80000"
    flags = "MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE"

    for i in range(len(genes)):

        py = f"python interaction_test_2snps_per_gene_permuteG.py {i} 0"
        cmd = f"{bsub} \"{flags} {py}\""
        print(cmd)
        os.system(cmd)

