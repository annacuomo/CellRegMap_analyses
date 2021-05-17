import os
import argparse
import re
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy import sparse


sc.settings.verbosity = 3     
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


parser = argparse.ArgumentParser()
parser.add_argument('--count_matrix', help="absolute path of the .tsv file containing the gene expression counts", type=str, required=True)
parser.add_argument('--metadata', help="absolute path of the .tsv file containing the cell info", type=str, required=True)
parser.add_argument('--fig_dir', help="absolute path of the directory to save the figures", type=str, required=True)
parser.add_argument('--output_dir', help="absolute path of the directory to save the pseudocell and pseudobulk files", type=str, required=True)
parser.add_argument('--hvgs', help="make pseudocells based on the given number of HVGs; will use all genes if no number is given")

args = parser.parse_args()


if os.path.isdir(os.path.join(ars.fig_dir, "Pseudocells")):
    sc.settings.figdir = os.path.join(ars.fig_dir, "Pseudocells")
else:
    os.makedirs(os.path.join(ars.fig_dir, "Pseudocells"))
    sc.settings.figdir = os.path.join(ars.fig_dir, "Pseudocells")
            
        
if os.path.isdir(ars.output_dir):
    pass
else:
    os.makedirs(ars.output_dir)
        

_, ext_counts = os.path.splitext(args.count_matrix)
assert ext_counts in [".csv", ".tsv"], "Please specify a gene expression matrix either in .tsv or .csv format"
log_norm_counts = pd.read_csv(args.count_matrix, index_col=0, sep="\t" if ext_counts == ".tsv" else ",")

_, ext_meta = os.path.splitext(args.metadata)
assert ext_meta in [".csv", ".tsv"], "Please specify a cell metadata file either in .tsv or .csv format"
metadata = pd.read_csv(args.metadata, sep="\t" if ext_counts == ".tsv" else ",")

gene_names = pd.DataFrame({"HGNC_name":[log_norm_counts.index.str.split("_")[i][1] for i in range(log_norm_counts.shape[0])], "ENSEMBL_id":[log_norm_counts.index.str.split("_")[i][0] for i in range(log_norm_counts.shape[0])]}, index=log_norm_counts.index)

adata = AnnData(X=log_norm_counts.values.T, obs=metadata, var=gene_names)

# Simple batch correction
adata.raw = adata.copy()
sc.pp.regress_out(adata, keys="experiment", n_jobs=None, copy=False)

# # Save the batch-corrected data if you wish
# adata.write(os.path.join(ars.output_dir, "endodiff_lognorm_experiment_regressed.h5"))

# PCA
if args.hvgs:
    sc.pp.highly_variable_genes(adata, n_top_genes=int(args.hvgs), batch_key=None, flavor="seurat")
    sc.tl.pca(adata, n_comps=(int(args.hvgs)-1), zero_center=True, svd_solver='arpack', random_state=32, use_highly_variable=True)
else:
    sc.tl.pca(adata, n_comps=(min(adata.shape)-1), zero_center=True, svd_solver='arpack', random_state=32, use_highly_variable=False)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50, save=".png")    


## Make donor-specific pseudocells
i=0
for d in adata.obs.donor_long_id.unique():
    i+=1
    print("{} out of {} donors".format(i, adata.obs.donor_long_id.nunique())) 
    ddonor = adata[adata.obs.donor_long_id == d].copy()
    assert ddonor.obs.donor_long_id.nunique() == 1
    
    if os.path.isdir(os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures")):
        sc.settings.figdir = os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures")
    else:
        os.makedirs(os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures"))
        sc.settings.figdir = os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures")
            
    sc.pp.neighbors(ddonor, n_neighbors=10, use_rep="X_pca", n_pcs=40, knn=True, method='umap', metric='euclidean', key_added="neighbors_euclidean")
    sc.tl.umap(ddonor, min_dist=0.1, spread=1.0, n_components=2, maxiter=None, alpha=1.0, gamma=1.0, negative_sample_rate=5, init_pos='spectral', random_state=0, a=None, b=None, copy=False, method='umap', neighbors_key="neighbors_euclidean")
    
    res = [0.6, 1, 1.8, 3.4, 4]
    for r in res:
        rr = str(r).replace(".", "")
        sc.tl.leiden(ddonor, resolution=r, neighbors_key="neighbors_euclidean", directed=True, key_added="leiden_res_"+rr+"_euclidean")
        
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18,6), constrained_layout=False)
        sc.pl.umap(ddonor, use_raw=False, color="day", components=['1,2'], size=10, title="", show=False, ax=axs[0])   
        sc.pl.umap(ddonor, use_raw=False, color="pseudo", components=['1,2'], size=10, title="", show=False, ax=axs[1])
        sc.pl.umap(ddonor, use_raw=False, color="leiden_res_"+rr+"_euclidean", components=['1,2'], size=10, title="", show=False, ax=axs[2])
        axs[0].legend(title="Day", loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=15, title_fontsize=16, ncol=1)
        axs[1].legend(title="Pseudotime", loc='upper left', bbox_to_anchor=(0.9, 0), frameon=False, fontsize=15, title_fontsize=16)
        axs[2].legend(title="Pseudocell", loc='upper left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=15, title_fontsize=16, ncol=4)
        [axs[i].xaxis.label.set_fontsize(18) for i in range(len(axs))]
        [axs[i].yaxis.label.set_fontsize(18) for i in range(len(axs))]
        plt.suptitle(" {}  -  res = {}".format(str(d), str(r)), fontsize=20, x=0.45, y=0.97)
        plt.tight_layout()
        plt.savefig(os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures", "UMAP_{}_pseudocells_Leiden_res-{}.png".format(str(d), rr)))       
        
    # Pseudocell stats
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(40,20), constrained_layout=True)
    for r in range(3):
        rr = str(res[r]).replace(".", "")
        clusters_cells = ddonor.obs.groupby(by="leiden_res_"+rr+"_euclidean").apply(lambda x: x.shape[0])
        sns.distplot(clusters_cells, bins='auto', hist_kws={'alpha':0.8, 'edgecolor':"darkblue", "facecolor":"mediumblue"}, kde=False, ax=axs[0,r])
        axs[0,r].grid(axis='both', alpha=0.9)
        axs[0,r].set_xlabel('Number of cells per pseudocell')
        axs[0,r].set_ylabel('Counts')
        axs[0,r].add_artist(AnchoredText('$median={}$\n$max={}$\n$min={}$'.format(int(np.median(clusters_cells)), max(clusters_cells), min(clusters_cells)), loc="upper right", frameon=False, prop={"color":"slategray", "fontfamily":"fantasy", "fontsize":"small", "fontstretch":"expanded", "ma":"right"}))
        axs[0,r].set_title('Leiden res={}'.format(res[r]))
    for r in range(2):
        rr = str(res[r+3]).replace(".", "")
        clusters_cells = ddonor.obs.groupby(by="leiden_res_"+rr+"_euclidean").apply(lambda x: x.shape[0])
        sns.distplot(clusters_cells, bins='auto', hist_kws={'alpha':0.8, 'edgecolor':"darkblue", "facecolor":"mediumblue"}, kde=False, ax=axs[1,r])
        axs[1,r].grid(axis='both', alpha=0.9)
        axs[1,r].set_xlabel('Number of cells per pseudocell')
        axs[1,r].set_ylabel('Counts')
        axs[1,r].add_artist(AnchoredText('$median={}$\n$max={}$\n$min={}$'.format(int(np.median(clusters_cells)), max(clusters_cells), min(clusters_cells)), loc="upper right", frameon=False, prop={"color":"slategray", "fontfamily":"fantasy", "fontsize":"small", "fontstretch":"expanded", "ma":"right"}))
        axs[1,r].set_title('Leiden res={}'.format(res[r+3]))
    fig.delaxes(ax=axs[1,2])
    fig.suptitle(str(d))
    plt.savefig(os.path.join(ars.fig_dir, "Pseudocells", str(d)+"_figures", "Number_of_cells_per_pseudocell_"+str(d)+"_Euclidean_Leiden.png"))
    
    if i==1:
        all_donors = ddonor
    else:
        all_donors = all_donors.concatenate(ddonor, join="outer", batch_key="DONOR", index_unique=None)
        

all_donors.write(os.path.join(args.output_dir, "Metacells_per_donor_across_days.h5"))        

# Plot n_cells per donor vs. n_pseudocells per donor
cells_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.shape[0])
pseudocells_06_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.leiden_res_06_euclidean.nunique())
pseudocells_1_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.leiden_res_1_euclidean.nunique())
pseudocells_18_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.leiden_res_18_euclidean.nunique())
pseudocells_34_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.leiden_res_34_euclidean.nunique())
pseudocells_4_donors = all_donors.obs.groupby(by="donor_long_id").apply(lambda x: x.leiden_res_4_euclidean.nunique())
    
plt.figure(figsize=(10,8))
plt.scatter(x=cells_donors, y=pseudocells_06_donors)
plt.scatter(x=cells_donors, y=pseudocells_1_donors)
plt.scatter(x=cells_donors, y=pseudocells_18_donors)
plt.scatter(x=cells_donors, y=pseudocells_34_donors)
plt.scatter(x=cells_donors, y=pseudocells_4_donors)
plt.xlabel('Cells per donor', size=16)
plt.ylabel('Pseudocells per donor', size=17)
plt.legend(["0.6", "1.0", "1.8", "3.4", "4.0"], title="Leiden res", loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=14, title_fontsize=16)
plt.tight_layout() 
plt.savefig(os.path.join(ars.fig_dir, "Pseudocells", "Cells_vs_Pseudocells_per_donor_Leiden.png"))

# Plot n_cells per donor vs. avg n_cells per pseudo-cell per donor 
avg_cells_pseudocell_06 = all_donors.obs.groupby(by=["donor_long_id","leiden_res_06_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
avg_cells_pseudocell_1 = all_donors.obs.groupby(by=["donor_long_id","leiden_res_1_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
avg_cells_pseudocell_18 = all_donors.obs.groupby(by=["donor_long_id","leiden_res_18_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
avg_cells_pseudocell_34 = all_donors.obs.groupby(by=["donor_long_id","leiden_res_34_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
avg_cells_pseudocell_4 = all_donors.obs.groupby(by=["donor_long_id","leiden_res_4_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
        
plt.figure(figsize=(10,8))
plt.scatter(x=cells_donors, y=avg_cells_pseudocell_06)
plt.scatter(x=cells_donors, y=avg_cells_pseudocell_1)
plt.scatter(x=cells_donors, y=avg_cells_pseudocell_18)
plt.scatter(x=cells_donors, y=avg_cells_pseudocell_34)
plt.scatter(x=cells_donors, y=avg_cells_pseudocell_4)
plt.xlabel('Cells per donor', size=16)
plt.ylabel('Average cells per pseudocell per donor', size=17)
plt.legend(["0.6", "1.0", "1.8", "3.4", "4.0"], title="Leiden res", loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(ars.fig_dir, "Pseudocells", "Cells_vs_Avg-Cells-Pseudocell_per_donor_Leiden.png"))

                        
### Pseudobulk expression ##

res_euclidean = {"leiden_res_06_euclidean":"06", "leiden_res_1_euclidean":"1", "leiden_res_18_euclidean":"18", "leiden_res_34_euclidean":"34", "leiden_res_4_euclidean":"4"}
columns_to_keep = ['cell_name', 'plate_id', 'plate_well_id', 'well_id', 'well_type', 'pseudo', 'samp_type', 'sample_id', 'experiment', 'day', 'donor', 'donor_short_id', 'donor_long_id', 'leiden_res_06_euclidean', 'leiden_res_1_euclidean', 'leiden_res_18_euclidean', 'leiden_res_34_euclidean', 'leiden_res_4_euclidean']


for k,v in res_euclidean.items():
    
    print(" ------- Resolution: {} --------\n\n".format(k))

    # Cells in each donor-specific pseudocell:
    cells = list(all_donors.obs.groupby(by=[k, "donor_long_id"]).groups.values())
            
    # Mean gene expression across cells in the first donor-specific pseudocell
    pseudobulk = all_donors[cells[0]].X.mean(axis=0)
    metainfo = all_donors[cells[0]].obs.xs(['donor_short_id', 'donor_long_id', k],axis=1).drop_duplicates().reset_index(drop=True)
    # Save the IDs, the day, the pseudotime and the experiment of the cells that make-up each pseudocell:
    metainfo = metainfo.assign(cell_day = [all_donors.obs.loc[cells[0], "day"].values.tolist()], cell_pseudo = [all_donors.obs.loc[cells[0], "pseudo"].values.tolist()], cell_experiment = [all_donors.obs.loc[cells[0], "experiment"].values.tolist()], cell_ids = [cells[0].values.tolist()])
    # Mean gene expression across cells in the rest of the pseudocells
    for c in cells[1:]:
        pseudobulk = np.vstack([pseudobulk, all_donors[c].X.mean(axis=0)])
        tmp = all_donors[c].obs.xs(['donor_short_id', 'donor_long_id', k],axis=1).drop_duplicates().reset_index(drop=True).assign(cell_day = [all_donors.obs.loc[c, "day"].values.tolist()], cell_pseudo = [all_donors.obs.loc[c, "pseudo"].values.tolist()], cell_experiment = [all_donors.obs.loc[c, "experiment"].values.tolist()], cell_ids = [c.values.tolist()])
        metainfo = metainfo.append(tmp).reset_index(drop=True)
    # Save the number of cells that make-up each pseudocell:
    metainfo = pd.concat([metainfo, pd.DataFrame({"number_of_cells":[len(c) for c in cells]}, index=metainfo.index)], axis=1)
    
    # Create and save AnnData object
    pseudobulk_res = AnnData(X=pseudobulk, obs=metainfo, var=all_donors.var)
    pseudobulk_res.X = sparse.csr_matrix(pseudobulk_res.X)
    # Need to explicitly save the list of cell_* as str:
    for col in pseudobulk_res.obs.columns:
        if isinstance(pseudobulk_res.obs[col][0], list):
            pseudobulk_res.obs[col] = [str(pseudobulk_res.obs[col].values[i]) for i in range(pseudobulk_res.obs.shape[0])]
   
    pseudobulk_res.write(os.path.join(args.output_dir, "Pseudobulk_per_donor_across_days_Leiden_Euclidean_res"+str(v)+".h5"))
                     
                        
