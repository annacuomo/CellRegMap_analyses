import sys
import os
import re
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy import sparse




def plot_QC(adata, title, fig_home_dir):
    
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sc.settings.figdir = os.path.join(fig_home_dir, "figures/QC_figures/")
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    adata.var["MITO"] = adata.var_names.str.startswith("MT-") 
    adata.var["RIBO"] = adata.var_names.str.startswith(("RPS","RPL"))
    adata.var["ERCC"] = adata.var_names.str.startswith("ERCC")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["MITO", "RIBO", "ERCC"], percent_top=None, log1p=True, inplace=True)
   
    sc.pl.violin(adata, keys=['n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_MITO','pct_counts_RIBO', 'pct_counts_ERCC'], use_raw=False, scale="count", jitter=False, groupby = "celltype", rotation = 45, xlabel=title, save = "_QC_plots_"+title.replace(" ", "-")+".png")

    sc.pl.scatter(adata, x='total_counts', y='total_counts_MITO', color="celltype", size=6, title=title, save = "_plot_MITO_"+title.replace(" ", "-")+".png")
    sc.pl.scatter(adata, x='total_counts', y='total_counts_ERCC', color="celltype", size=6, title=title, save = "_plot_ERCC_"+title.replace(" ", "-")+".png")
    sc.pl.scatter(adata, x='total_counts', y='total_counts_RIBO', color="celltype", size=6, title=title, save = "_plot_RIBO_"+title.replace(" ", "-")+".png")

    tmp = sns.distplot(adata.obs['log1p_total_counts'], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_counts_per_barcode_"+title.replace(" ", "-")+"_log1p.png"))
    tmp.get_figure().clf()
    tmp = sns.distplot(adata.obs['total_counts'], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_counts_per_barcode_"+title.replace(" ", "-")+".png"))
    tmp.get_figure().clf()
    tmp = sns.distplot(adata.obs['total_counts'][adata.obs['total_counts']<10000], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_counts_per_barcode_lower_tail_"+title.replace(" ", "-")+".png"))
    tmp.get_figure().clf()
    tmp = sns.distplot(adata.obs["n_genes_by_counts"], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_genes_per_barcode_"+title.replace(" ", "-")+".png"))
    tmp.get_figure().clf()
    tmp = sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"]<2000], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_genes_per_barcode_lower_tail_"+title.replace(" ", "-")+".png"))
    tmp.get_figure().clf()
    tmp = sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"]>6000], kde=False).set_title(title)
    tmp.get_figure().savefig(os.path.join(fig_home_dir, "figures/QC_figures/QC_n_genes_per_barcode_upper_tail_"+title.replace(" ", "-")+".png"))
    tmp.get_figure().clf()
    
    sc.pl.scatter(adata, x="log1p_total_counts", y="n_genes_by_counts", size=6, color="pct_counts_MITO", title=title, save="_QC_combined_"+title.replace(" ", "-")+"log1p.png")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", size=6, color="pct_counts_MITO", title=title, save="_QC_combined_"+title.replace(" ", "-")+".png")
    sc.pl.scatter(adata[adata.obs.loc[(adata.obs.total_counts<10000) & (adata.obs.n_genes_by_counts <3500)].index], x="total_counts", y="n_genes_by_counts", size=6, color="pct_counts_MITO", title=title, save="_QC_combined_zoom-in_"+title.replace(" ", "-")+".png")





sc.settings.verbosity = 3 
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

wd = os.path.abspath(os.getcwd())


d11 = sc.read(wd+"/Data/Neuroseq/raw_D11.h5") # 253381 cells x 32738 genes
d30 = sc.read(wd+"/Data/Neuroseq/raw_D30.h5") # 250923 cells x 32738 genes
d52 = sc.read(wd+"/Data/Neuroseq/raw_D52.h5")
d52_untr = d52[d52.obs.treatment == "NONE"] # 303856 cells x 32738 genes
d52_tr = d52[d52.obs.treatment == "ROT"]  # 219238 cells x 32738 genes


## Filter cells based on QC plots

plot_QC(d11, "D11", wd)
sc.pp.filter_cells(d11, min_genes=1500) # 10426 cells removed
sc.pp.filter_cells(d11, min_counts=5000) # 1300 cells removed

plot_QC(d30, "D30", wd)
d30 = d30[d30.obs.loc[~((d30.obs.n_genes_by_counts < 1000) & (d30.obs.log1p_total_counts > 8) & (d30.obs.log1p_total_counts <=9))].index]
d30 = d30[d30.obs.loc[~((d30.obs.n_genes_by_counts < 1200) & (d30.obs.log1p_total_counts > 8.2))].index]
d30 = d30[d30.obs.loc[~((d30.obs.n_genes_by_counts < 1800) & (d30.obs.log1p_total_counts > 8.7))].index] # 98 cells removed in total

plot_QC(d52_untr, "D52 untreated", wd)
sc.pp.filter_cells(d52_untr, min_genes=1200) # 651 cells removed
d52_untr = d52_untr[d52_untr.obs.loc[~((d52_untr.obs.pct_counts_MITO > 40) & (d52_untr.obs.n_genes_by_counts<2000) & (d52_untr.obs.total_counts < 10000))].index] # 53 cells removed

plot_QC(d52_tr, "D52 treated", wd)
sc.pp.filter_cells(d52_tr, min_genes=1200) # 247 cells removed
d52_tr = d52_tr[d52_tr.obs.loc[~((d52_tr.obs.pct_counts_MITO >= 40) & (d52_tr.obs.total_counts<10000) & (d52_tr.obs.n_genes_by_counts < 2000))].index] # 19 cells removed


all_days = d11.concatenate(d30, d52_untr, d52_tr, join='inner', batch_key='Condition', batch_categories=["d11","d30","d52_untr","d52_tr"], index_unique="_") # (1,014,604 cells x 32,738 genes)

all_days.write(wd+"/Data/Neuroseq/All_conditions_QCed_raw.h5")


for ct in all_days.obs.celltype.unique():
    print("\n--------- Cell type: {} --------".format(ct))
    ct_all = all_days[all_days.obs.celltype == ct]  
    
    if os.path.isdir(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures")):
        sc.settings.figdir = wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"
    else:
        os.makedirs(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures"))
        sc.settings.figdir = wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"
    
    # Do normalisation across all donors and conditions
    sc.pp.normalize_total(ct_all, target_sum=1e6, exclude_highly_expressed=True, max_fraction=0.10, key_added="Norm_factors", inplace=True) # ['POMC', 'SST', 'CGA', 'TMSB4X', 'MALAT1', 'TTR'] genes excluded

    # Log2 transform
    sc.pp.log1p(ct_all, base=2)

    # Batch correction for pool_id with Harmony
    ct_all.X = ct_all.X.todense()
    sc.tl.pca(ct_all, n_comps=50, zero_center=True, svd_solver='arpack', random_state=32, use_highly_variable=False)
    sce.pp.harmony_integrate(ct_all, key="pool_id", basis='X_pca', adjusted_basis='X_pca_harmony')

    # Donor representation per condition
    nd = ct_all.obs.donor_id.nunique()
    my_colors = cm.get_cmap('RdYlBu', nd)
    
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20,20), constrained_layout=True)
    tmp = pd.crosstab(ct_all.obs['Condition'], ct_all.obs['donor_id'], normalize='index')
    tmp.plot.bar(stacked=True, ax=axs, colormap=my_colors)
    axs.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=22, title_fontsize=24, ncol=6)
    axs.title.set_fontsize(24)
    axs.xaxis.label.set_fontsize(24)
    [t.label.set_fontsize(24) for t in axs.xaxis.get_major_ticks()]
    [t.label.set_rotation("horizontal") for t in axs.xaxis.get_major_ticks()]
    [t.label.set_fontsize(24) for t in axs.yaxis.get_major_ticks()]
    plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"+str(ct)+"_Donor_representation_per_condition.png", bbox_inches='tight')

    # Check also the pool
    donors_pools = pd.DataFrame.from_dict(ct_all.obs.groupby(by="donor_id").apply(lambda x: x.pool_id.nunique()).to_dict(), orient="index").rename(columns={0:"counts"})
    
    fig, ax = plt.subplots(ncols=1, figsize=(5,5), constrained_layout=True, subplot_kw=dict(aspect="equal"))
    ax.pie(x=[donors_pools.loc[donors_pools.counts == c].shape[0] for c in donors_pools.counts.unique()], labels = donors_pools.counts.unique(), autopct='%1.1f%%', colors=["mediumblue", "r", "y"], wedgeprops = {'linewidth': 5}, labeldistance=0.35, radius=1.7, textprops=dict(color="w", weight="bold", va="top", ma="center")),
    fig.suptitle("Number of pools per donor")
    plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/Number_of_pools_per_donor_"+str(ct)+"-cells.png")


    for con in ct_all.obs.Condition.unique():
        print("------- Condition: {} -------".format(con))
        ct_con = ct_all[ct_all.obs.Condition == con]
        
        if os.path.isdir(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con))):
            sc.settings.figdir = os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con))
        else:
            os.makedirs(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con)))
            sc.settings.figdir = os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con))
           
        # Rank donors according to number of cells
        donors_cells = {}
        for d in ct_con.obs.donor_id.unique():
            donors_cells[d] = ct_con.obs.loc[ct_con.obs.donor_id == d].shape[0]
    
        fig, axs = plt.subplots(ncols=1, figsize=(5,5), constrained_layout=True)
        sns.distplot(list(donors_cells.values()), bins='auto', hist_kws={'alpha':0.8, 'edgecolor':"darkblue", "facecolor":"mediumblue"}, kde=False, ax=axs)
        plt.grid(axis='both', alpha=0.5)
        plt.xlabel('Cells')
        plt.ylabel('Counts')
        if bool(re.search("52", str(con))):
            plt.title("Number of "+str(ct)+" cells per donor in "+str(con).replace("_", " ")+"eated")
        else:
            plt.title("Number of "+str(ct)+" cells per donor in "+str(con))
        axs.add_artist(AnchoredText('$median={}$\n$max={}$\n$min={}$'.format(int(np.median(list(donors_cells.values()))), max(list(donors_cells.values())), min(list(donors_cells.values()))), loc="upper right", frameon=False, prop={"color":"slategray", "fontfamily":"fantasy", "fontsize":"small", "fontstretch":"expanded", "ma":"right"}))
        plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"+str(con)+"/Hist_number_of_"+str(ct)+"-cells_per_donor_"+str(con)+".png")
    
    
        # Remove donors with less than 20 cells in the given condition
        (np.array(list(donors_cells.values())) < 20).sum()
        for k,v in donors_cells.items():
            if v<20:
                ct_con = ct_con[ct_con.obs.loc[ct_con.obs.donor_id !=k].index]

        i=1
        for d in ct_con.obs.donor_id.unique():
            print("{} out of {} donors".format(i, ct_con.obs.donor_id.nunique())) 
            ct_donor = ct_con[ct_con.obs.donor_id == d]
            assert ct_donor.obs.donor_id.nunique() == 1
    
            if os.path.isdir(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con), str(d)+"_figures")):
                sc.settings.figdir = os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con), str(d)+"_figures")
            else:
                os.makedirs(os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con), str(d)+"_figures"))
                sc.settings.figdir = os.path.join(wd, "figures", "figures_per_donor", str(ct)+"_figures", str(con), str(d)+"_figures")
            
            sc.pp.neighbors(ct_donor, n_neighbors=10, use_rep="X_pca_harmony", n_pcs=50, knn=True, method='umap', metric='euclidean', key_added="neighbors_euclidean")
            sc.tl.umap(ct_donor, min_dist=0.1, spread=1.0, n_components=2, maxiter=None, alpha=1.0, gamma=1.0, negative_sample_rate=5, init_pos='spectral', random_state=0, a=None, b=None, copy=False, method='umap', neighbors_key="neighbors_euclidean")
    
            res = [1, 1.8, 3.4, 4]
            for r in res:
                rr = str(r).replace(".", "")
                sc.tl.leiden(ct_donor, resolution=r, neighbors_key="neighbors_euclidean", directed=False, key_added="leiden_res_"+rr+"_euclidean")
                sc.pl.umap(ct_donor, color="leiden_res_"+rr+"_euclidean", size=14, save="_Harmony_"+str(ct)+"-cells_"+str(d)+"_"+str(con)+"_Leiden-res"+rr+"-Euclidean.png") 
        
            fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(40,20), constrained_layout=True)
            for r in range(4):
                rr = str(res[r]).replace(".", "")
                clusters_cells = ct_donor.obs.groupby(by="leiden_res_"+rr+"_euclidean").apply(lambda x: x.shape[0])
                sns.distplot(clusters_cells, bins='auto', hist_kws={'alpha':0.8, 'edgecolor':"darkblue", "facecolor":"mediumblue"}, kde=False, ax=axs[r])
                axs[r].grid(axis='both', alpha=0.9)
                axs[r].set_xlabel('Number of cells per pseudocell')
                axs[r].set_ylabel('Counts')
                axs[r].add_artist(AnchoredText('$median={}$\n$max={}$\n$min={}$'.format(int(np.median(clusters_cells)), max(clusters_cells), min(clusters_cells)), loc="upper right", frameon=False, prop={"color":"slategray", "fontfamily":"fantasy", "fontsize":"small", "fontstretch":"expanded", "ma":"right"}))
                axs[r].set_title('Leiden res={}'.format(res[r]))
            fig.suptitle('{} - {}'.format(d, con))
            plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"+str(con)+"/"+str(d)+"_figures"+"/Number_of_cells_per_pseudocell_"+str(ct)+"_"+str(con)+"_"+str(d)+"_Euclidean_Leiden.png")
    
            if i==1:
                all_donors_ct = ct_donor
            else:
                all_donors_ct = all_donors_ct.concatenate(ct_donor, join="inner", batch_key="DONOR")
        
            print(all_donors_ct.shape)
            i+=1
        
        all_donors_ct.write(os.path.join(wd, "Data/Neuroseq/Metacells_per_donor", str(ct)+"_"+str(con)+"_donors.h5"))
        
        # Plot n_cells per donor vs. n_pseudocells per donor
        cells_donors = all_donors_ct.obs.groupby(by="donor_id").apply(lambda x: x.shape[0])
        pseudocells_1_donors = all_donors_ct.obs.groupby(by="donor_id").apply(lambda x: x.leiden_res_1_euclidean.nunique())
        pseudocells_18_donors = all_donors_ct.obs.groupby(by="donor_id").apply(lambda x: x.leiden_res_18_euclidean.nunique())
        pseudocells_34_donors = all_donors_ct.obs.groupby(by="donor_id").apply(lambda x: x.leiden_res_34_euclidean.nunique())
        pseudocells_4_donors = all_donors_ct.obs.groupby(by="donor_id").apply(lambda x: x.leiden_res_4_euclidean.nunique())
    
        plt.figure(figsize=(10,8))
        plt.scatter(x=cells_donors, y=pseudocells_1_donors)
        plt.scatter(x=cells_donors, y=pseudocells_18_donors)
        plt.scatter(x=cells_donors, y=pseudocells_34_donors)
        plt.scatter(x=cells_donors, y=pseudocells_4_donors)
        plt.xlabel('Cells per donor', size=16)
        plt.ylabel('Pseudocells per donor', size=17)
        plt.title(" - ".join([str(ct), str(con)]), size=20)
        plt.legend(["1.0", "1.8", "3.4", "4.0"], title="Leiden res", loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=14, title_fontsize=16)
        plt.tight_layout() 
        plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"+str(con)+"/"+str(ct)+"_cells_vs_Pseudocells_per_donor_Leiden_"+str(con)+".png")

        # Plot n_cells per donor vs. avg n_cells per pseudo-cell per donor 
        avg_cells_pseudocell_1 = all_donors_ct.obs.groupby(by=["donor_id","leiden_res_1_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
        avg_cells_pseudocell_18 = all_donors_ct.obs.groupby(by=["donor_id","leiden_res_18_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
        avg_cells_pseudocell_34 = all_donors_ct.obs.groupby(by=["donor_id","leiden_res_34_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
        avg_cells_pseudocell_4 = all_donors_ct.obs.groupby(by=["donor_id","leiden_res_4_euclidean"]).apply(lambda x: x.shape[0]).mean(level=0)
        
        plt.figure(figsize=(10,8))
        plt.scatter(x=cells_donors, y=avg_cells_pseudocell_1)
        plt.scatter(x=cells_donors, y=avg_cells_pseudocell_18)
        plt.scatter(x=cells_donors, y=avg_cells_pseudocell_34)
        plt.scatter(x=cells_donors, y=avg_cells_pseudocell_4)
        plt.xlabel('Cells per donor', size=16)
        plt.ylabel('Average cells per pseudocell per donor', size=17)
        plt.title(" - ".join([str(ct), str(con)]), size=20)
        plt.legend(["1.0", "1.8", "3.4", "4.0"], title="Leiden res", loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=14, title_fontsize=16)
        plt.tight_layout()
        plt.savefig(wd+"/figures/figures_per_donor/"+str(ct)+"_figures/"+str(con)+"/"+str(ct)+"_cells_vs_Avg-Cells-Pseudocell_per_donor_Leiden_"+str(con)+".png")
