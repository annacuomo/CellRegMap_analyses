Add workflow image

* step1: run_association (discovery)
  * select all SNP-gene at FDR<20%
  * create filter file for those SNP-gene pairs

* step2: run_interaction (on SNP-gene pairs from filter file)
  * after multiple testing correction, identify list of context-specific eQTLs (FDR<5%)

* step3: estimate betas (for significant GxC eQTLs from step2)


step1 can be skipped and substituted by directly creating a filter file based on a set of a priori defined eQTLs we want to investigate
