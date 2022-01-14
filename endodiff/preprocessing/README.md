## Preprocessing steps (TODO: add workflow figure)

* from count matrix (genes x cells)
  * extract latent representation C (e.g., [MOFA](https://biofam.github.io/MOFA2/), PCA) - scripts to use MOFA [here](../preprocessing/run_MOFA.R) and [here](../preprocessing/save_MOFA_results.ipynb)
    * this may require raw or normalised counts, could require restricting to HVGs only   
    * standardize factors before building C (factors are columns of **C**)
  * phenotype vectors (one gene at a time, cells x 1) - log2(cpm+1) for the entire count matrix, quantile-normalised each **y**

* genotypes and kinship (plink) - follow [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#genotype-file) and [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#kinship-matrix-file) instructions
  * expand genotypes **G** (donors to cells) as in [here](../preprocessing/Expand_genotypes_kinship.ipynb) 
  * decompose K to hK and then expand **hK** (donors to cells) as in [here](../preprocessing/Expand_genotypes_kinship.ipynb) 
    * if no K is available, consider using [this script](../preprocessing/block_diagonal_K.ipynb) to generate hK such that K is block-diagonal. 
    This will account for repeated observations (cells) for donors (but not for releatedness across donors).

* Fixed effect covariates (**W**)
  * if none are available, add a simple intercept as ``W = numpy.ones((n_cells, 1))``
  * if they are well defined at cell-level (e.g., batch) simply add them as colunms of W
  * if they are well defined at donor-level (e.g., age, sex) expand them from donors to cells

y, C, G (expanded), hK (expanded) and W are the inputs for CellRegMap.