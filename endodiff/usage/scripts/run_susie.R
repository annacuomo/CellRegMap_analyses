library(susieR)
library(ggplot2)

revision_folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"

myfile0 = paste0(revision_folder,"Marcs_results_all.csv")
df0 = read.csv(myfile0, row.names=1)
print(head(df0))

LD_folder = paste0(revision_folder,"LD_matrices_for_susie/")

genes = gsub(".csv","",list.files(LD_folder))

for (gene in genes){

    filename = paste0(revision_folder,"results_w_susie_credible_sets/",gene,".csv")
    if (file.exists(filename)){next}
    
    print(gene)
    ## get LD matrix
    R_file = paste0(LD_folder,gene,".csv")
    R = read.csv(R_file, row.names=1)
    colnames(R) = gsub("X","",colnames(R))
    #print(head(R))  
    print(dim(R))

    ## get Z scores
    df_rel = df0[df0$feature_id == gene,]
    if (nrow(df_rel) != length(unique(df_rel$snp_id))){
        df_rel = df_rel[-which(duplicated(df_rel$snp_id)),]
    }
    df_rel$z = df_rel$beta / df_rel$beta_se
    #print(head(df_rel))
    print(nrow(df_rel))    

    # some SNPs may be missing
    snps1 = unique(as.character(df_rel$snp_id))
    snps2 = unique(as.character(rownames(R)))
    common_snps = snps1[snps1 %in% snps2]

    # reorder
    R = R[common_snps,common_snps]
    R = R[order(rownames(R)),order(colnames(R))]
    df_rel = df_rel[df_rel$snp_id %in% common_snps,]
    df_rel = df_rel[order(df_rel$snp_id),]

    z_scores = df_rel$z
    
    # run Susie using summary stats
    fitted_rss <- susie_rss(z_scores, as.matrix(R), L = 10)

    cs = summary(fitted_rss)$vars
    cs_ordered = cs[order(cs$variable),]
    df_cs = cbind(df_rel,cs_ordered)

    write.csv(df_cs, filename)
}
