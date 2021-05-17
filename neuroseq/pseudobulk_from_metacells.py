import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from scipy import sparse


sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="absolute path of the directory containing the metacells", type=str)
args = parser.parse_args()

files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
# Select files in the dir containing donor-specific metacells; e.g. FPP_d11_donors.h5, DA_d52_untr_donors.h5
r = re.compile(".+_d[0-9]{2}(_tr|_untr)?_donors.h5")
metacell_files = list(filter(r.match, files))
# Get the cell-type names
ct = re.compile("^[A-Za-z]+([0-9]|_[A-Z]+([a-z]+([0-9]?))?)?")
celltypes = list(set([re.match(ct, metacell_files[i]).group(0) for i in range(len(metacell_files))]))


columns_to_keep = ['donor_id', 'celltype', 'time_point', 'pool_id', 'treatment', 'Condition', 'leiden_res_1_euclidean', 'leiden_res_18_euclidean', 'leiden_res_34_euclidean', 'leiden_res_4_euclidean']

res = {'leiden_res_1_euclidean':"1", 'leiden_res_18_euclidean':"18", 'leiden_res_34_euclidean':"34", 'leiden_res_4_euclidean':"4"}

for k,v in res.items():
    
    print(" ------- Resolution: {} --------\n\n".format(k))
    init = True
    for ctp in celltypes:
        print("------- {} cells -------\n".format(ctp))
        ct_files = list(filter(re.compile(ctp).match, metacell_files))

        for f in ct_files:
            myfile = sc.read(os.path.join(args.data_dir, f))
            myfile.obs = myfile.obs.xs(columns_to_keep, axis=1)
            con = myfile.obs.Condition.unique()
            print("------- Condition: {} -------".format(con))
            i=1
            for d in myfile.obs.donor_id.unique():
                print("Donor {} out of {}".format(i, myfile.obs.donor_id.nunique()))
                ct_donor = myfile[myfile.obs.donor_id == d]
                cells = list(ct_donor.obs.groupby(by=k).groups.values())
                clusters = list(ct_donor.obs.groupby(by=k).groups.keys())
                if init:
                    # Mean gene expression across cells in the pseudocell
                    pseudobulk = ct_donor[cells[0]].X.mean(axis=0)
                    metadata = ct_donor[cells[0]].obs.xs(['donor_id', 'celltype', 'time_point', 'treatment', 'Condition', k],axis=1).drop_duplicates().reset_index(drop=True)
                    init=False
                    for c in cells[1:]:
                        pseudobulk = np.vstack([pseudobulk, ct_donor[c].X.mean(axis=0)])
                        metadata = metadata.append(ct_donor[c].obs.xs(['donor_id', 'celltype', 'time_point', 'treatment', 'Condition', k],axis=1).drop_duplicates().reset_index(drop=True)).reset_index(drop=True)
                    # Save the number of cells and the IDs of the cells that make-up each pseudocell:
                    metadata = pd.concat([metadata, pd.DataFrame({"Number_of_cells":[len(c) for c in cells], "cell_ids":cells}, index=metadata.index)], axis=1)
                else:
                    pseudobulk = np.vstack([pseudobulk, ct_donor[cells[0]].X.mean(axis=0)])
                    tmp = ct_donor[cells[0]].obs.xs(['donor_id', 'celltype', 'time_point', 'treatment', 'Condition', k],axis=1).drop_duplicates().reset_index(drop=True)
                    for c in cells[1:]:
                        pseudobulk = np.vstack([pseudobulk, ct_donor[c].X.mean(axis=0)])
                        tmp = tmp.append(ct_donor[c].obs.xs(['donor_id', 'celltype', 'time_point', 'treatment', 'Condition', k],axis=1).drop_duplicates().reset_index(drop=True)).reset_index(drop=True)
                    tmp = pd.concat([tmp, pd.DataFrame({"Number_of_cells":[len(c) for c in cells], "cell_ids":cells}, index=tmp.index)], axis=1)
                    metadata = metadata.append(tmp).reset_index(drop=True)
                    
                i+=1

                
    pseudobulk_res = AnnData(X=pseudobulk, obs=metadata, var=myfile.var.xs(myfile.var.columns[:4], axis=1))
    pseudobulk_res.var.columns = ['MITO', 'RIBO', 'ERCC', 'gene_ids-0']
    pseudobulk_res.X = sparse.csr_matrix(pseudobulk_res.X)
    # Need to explicitly save the list of cell_ids as str:
    pseudobulk_res.obs.cell_ids = [pseudobulk_res.obs.cell_ids.values[i].tolist() for i in range(pseudobulk_res.obs.shape[0])]
    pseudobulk_res.obs.cell_ids = [str(pseudobulk_res.obs.cell_ids.values[i]) for i in range(pseudobulk_res.obs.shape[0])]

    pseudobulk_res.write(os.path.join(args.data_dir, "Pseudobulk_per_donor_all_cells_all_conditions_Leiden_res"+str(v)+".h5"))
