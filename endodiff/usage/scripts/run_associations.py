import os
#import re
import pandas as pd
#import sys

if __name__ == '__main__':

    for j in range(22):
        chrom = j+1
        print(chrom)
        #if chrom != 19: continue

        input_files_dir = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/new/input_files/"
        # filter file (columns: snp_id, gene)
        #endo_eqtl_file = input_files_dir+"endodiff_eqtl_allconditions_FDR10pct.csv"
        #endo_eqtl = pd.read_csv(endo_eqtl_file, index_col = False)
        #endo_eqtl["chrom"] = [int(i[:i.find("_")]) for i in endo_eqtl["snp_id"]]              
        #genes = endo_eqtl[endo_eqtl['chrom']==int(chrom)]['feature'].unique()
        # all genes
        genes_file = input_files_dir+"genes_tested.csv"
        g = pd.read_csv(genes_file, index_col = False)
        genes = g[g['chrom']==int(chrom)]['feature'].unique()
        
        print("prepare job to submit")
        bsub = "bsub -R \"rusage[mem=80000]\" -M 80000"
        flags = "MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE"
        
        for i in range(len(genes)):

            py = f"python association_test_for_one_gene.py {chrom} {i}"
            cmd = f"{bsub} \"{flags} {py}\""
            print(cmd)
            #sys.exit(0)
            os.system(cmd)

