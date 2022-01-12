Add workflow showing preprocessing steps

* from count matrix (genes x cells)
  * extract latent representation C (e.g., MOFA, PCA)
    * this may require raw or normalised counts, could require restricting to HVGs only   
    * standardize factors before building C (factors are columns of C)
  * phenotype vectors (one gene at a time, cells x 1) - log2(cpm+1) for the entire count matrix, quantile-normalised each y

* genotypes and kinship (plink) - follow [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#genotype-file) and [these](https://github.com/single-cell-genetics/limix_qtl/wiki/Inputs#kinship-matrix-file) instructions
 * expand genotypes (donors to cells) as in [here](../preprocessing/Expand_genotypes_kinship.ipynb) 
 * decompose K to hK and then expand hK (donors to cells) as in [here](../preprocessing/Expand_genotypes_kinship.ipynb) 
