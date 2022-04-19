import os
#import glob
#import pandas as pd
#import dask.dataframe as dd
#import sys

if __name__ == '__main__':

    for i in range(22):
        chrom = i+1
        print(chrom)
        #if chrom == 8: continue
        #if chrom == 9: continue
        #if chrom == 10: continue
        #if chrom == 12: continue
        #if chrom == 13: continue
        #if chrom == 14: continue
        #if chrom == 16: continue
        #if chrom == 18: continue
        #if chrom == 20: continue
        #if chrom == 21: continue
        #if chrom == 22: continue
        bsub = "bsub -q standard -n 10 -o out/ -e err/ -R \"rusage[mem=80000]\" -M 80000"
        flags = "MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE"
        #for j in range(100):
        py = f"python neuroseq_MOFA_discrete_contexts.py {chrom}"
        #py = f"python neuroseq_MOFA.py {chrom} no"
        #py = f"python neuroseq_MOFA_ge.py {chrom} no"
        cmd = f"{bsub} \"{flags} {py}\""
        print(cmd)
        #sys.exit(0)
        os.system(cmd)

