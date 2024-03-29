## Usage 

* Step1: run_association (discovery)
  *   Consider all genes (perhaps above a mean expression threshold) and all SNPs in _cis_ (specifying a window, _e.g._, 1Mb)
  *   For example, run [association script](scripts/association_test_for_one_gene.py) for all gene-SNP pairs (in _cis_) - you can use a [runner script](scripts/run_associations.py) to submit one job per gene
  *   Summarise all results (_e.g._, using this [summarising script](scripts/summarise_associations.py))
  *   Perform muliple testing correction

* Intermediate step 1
  * select all SNP-gene pairs at FDR<20%
  * create filter file for those SNP-gene pairs

* Step2: run_interaction (on SNP-gene pairs from filter file)
  * For example, use the [interaction script](scripts/interaction_test_for_10_snp_gene_pairs.py), [runner script](scripts/run_interactions.py) and [summarising script](scripts/summarise_interactions.py) to run this analysis, using the filter file defined above
  * perform multiple testing correction

* Intermediate step 2
  * after multiple testing correction, identify list of context-specific eQTLs (FDR < 5%)

* Step3: estimate betas (for significant GxC eQTLs from step2)
<!--  * [example script] - also add runner and summarising scripts -->


### Note

Step1 can be skipped and substituted by directly creating a filter file based on a set of a priori defined eQTLs we want to investigate (add example script)

<!-- ### (TODO: add workflow image) -->
