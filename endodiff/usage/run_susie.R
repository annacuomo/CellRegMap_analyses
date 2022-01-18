library(susieR)
library(ggplot2)

revision_folder = "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/debug_May2021/REVISION/"

myfile0 = paste0(revision_folder,"Marcs_results_all.csv")
df0 = read.csv(myfile0, row.names=1)

LD_folder = paste0(revision_folder,"LD_matrices_for_susie/")

genes = gsub(".csv","",list.files(LD_folder))

for (gene in genes){

    filename = paste0(revision_folder,"results_w_susie_credible_sets/",gene,".csv")
    if (file.exists(filename)){next}

    ## get LD matrix
    R_file = paste0(LD_folder,gene,".csv")
    R = read.csv(R_file, row.names=1)
    colnames(R) = gsub("X","",colnames(R))

    ## get Z scores
    df_rel = df0[df0$feature_id == gene,]
    df_rel$z = df_rel$beta / df_rel$beta_se

    # reorder
    R = R[order(rownames(R)),order(colnames(R))]
    df_rel = df_rel[order(df_rel$snp_id),]

    z_scores = df_rel$z
    
    # run Susie using summary stats
    fitted_rss <- susie_rss(z_scores, as.matrix(R), L = 10)

    cs = summary(fitted_rss)$vars
    cs_ordered = cs[order(cs$variable),]
    df_cs = cbind(df_rel,cs_ordered)

    write.csv(df_cs, filename)
}