## Preprocessing steps

* From count matrix (genes x cells)
  * Extract latent representation C (_e.g._, [MOFA](https://biofam.github.io/MOFA2/), PCA) - scripts to use MOFA [here](run_MOFA.R) and [here](save_MOFA_results.ipynb)
    * this may require raw or normalised counts, could require restricting to HVGs only   
    * standardize factors before building C (factors are columns of **C**)
  * Define single-cell expression "phenotype" vectors (one gene at a time, cells x 1) - log2(cpm+1) for the entire count matrix, quantile-normalised each **y**

* Genotypes and Kinship files (plink) - follow [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#genotype-file) and [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#kinship-matrix-file) instructions
  * Expand genotypes **G** (donors to cells) as in [here](Expand_genotypes_kinship.ipynb) 
  * Decompose K to hK and then expand **hK** (donors to cells) as in [here](Expand_genotypes_kinship.ipynb) 
    * if no K is available, consider using [this script](block_diagonal_K.ipynb) to generate hK such that K is block-diagonal. 
    This will account for repeated observations (cells) for donors (but not for releatedness across donors).

* Fixed effect covariates (**W**)
  * if none are available, add a simple intercept as ``W = numpy.ones((n_cells, 1))``
  * if they are well defined at cell-level (_e.g._, batch) simply add them as colunms of W
  * if they are well defined at donor-level (_e.g._, age, sex) expand them from donors to cells

y, C, G (expanded), hK (expanded) and W are the inputs for CellRegMap.

<!-- ### TODO: add workflow figure

 consider adding pseudocells workflow here

details on normalisation (e.g. SCT), batch correction

details on genotype format(s) -->
