import sys
import re
import glob
import os
import h5py
import pdb
import pandas as pd
import scipy as sp
import numpy as np
import statsmodels
# import limix.stats.fdr as fdr

def smartAppend(table,name,value):
    """ helper function for appending in a dictionary """
    if name not in table.keys():
        table[name] = []
    table[name].append(value)

def dumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array """
    for key in RV.keys():
        o.create_dataset(name=key,data=sp.array(RV[key]),chunks=True,compression='gzip')

def smartDumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array or still a dictionary (in this case, it iterates) """
    for key in RV.keys():
        if type(RV[key])==dict:
            g = o.create_group(key)
            smartDumpDictHdf5(RV[key],g)
        else:
            o.create_dataset(name=key,data=sp.array(RV[key]),chunks=True,compression='gzip')

def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

#path_results = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/CRM_association/"
path_results = "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/CRM_association_KE/"

if __name__ == '__main__':

    fname = os.path.join(path_results,"*.tsv")
    #prinm(outfilename)
    files = glob.glob(fname)
    # print (files)
    #import pdb; pdb.set_trace()
    x = 0
    table = {}
    count_success = 0
    count_failed = 0
    #breakpoint()
    for file in files:
        #print(file)
        #breakpoint()
        if re.search("perm",file) is not None:
            continue
        x += 1
        if x%500 == 0: print (x)
        df = pd.read_csv(file, index_col=0)
        #print(df.head())
        nsnps = int(len(df))
        if nsnps==0:
            continue
        line = str(file).split("/")
        gene = str(line[-1]).split(".")[0]
        chrom = df['chrom'].values[0]
        #print(gene)
        pval = df['pv'].values
        #pval[np.isnan(pval)]=1
        pval[pd.isnull(pval)]=1
        for i in range(nsnps):
            try:
                #import pdb; pdb.set_trace()
                temp={}
                temp['gene'] = gene
                temp['n_snps'] = nsnps
                temp['chrom'] = chrom
                #print(nsnps)
                temp['pv_raw'] = df['pv'].values[i]
                temp['snpID'] = df['variant'].values[i]
                #FWER adjusted (gene-level) pvalue
                temp['pv'] = nsnps*temp['pv_raw']
                if temp['pv']>1: temp['pv'] = 1
                if temp['pv']<0: temp['pv'] = 0
                count_success+=1
            
            except:
                count_failed+=1
                continue
        
            for key in temp.keys():
                smartAppend(table,key,temp[key])
    
    #import pdb; pdb.set_trace()
    for key in table.keys():
        table[key] = sp.array(table[key])
    print (count_success)
    
    df = pd.DataFrame.from_dict(table)
    import os
    outfile = "summary.csv"
    myp = os.path.join(path_results,outfile)
    df.to_csv(myp)

