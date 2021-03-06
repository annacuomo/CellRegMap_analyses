"""
Snakefile for DA cells (neuroseq) across conditions for specific genes (colocs) and specific cells (pseudocells with strongest effect sizes)
15 expression PCs as covariates (fixed effects)
1/n to account for different numbers of cells per donor increasing the variance AND kinship to account for same individuals across conditions

Author: Anna Cuomo
Affiliation: EMBL-EBI, Wellcome Sanger Institute, Garvan Institute
Date: Monday 28 February 2022
#Run: snakemake --snakefile ./snakemake.smk --jobs 400 --latency-wait 30 --cluster-config /nfs/leia/research/stegle/acuomo/singlecell_endodiff/singlecell_endodiff/pipeline_snakemakes/cluster.json --cluster 'bsub -q {cluster.queue} -n {cluster.n} -R "rusage[mem={cluster.memory}]" -M {cluster.memory} -o ./DA.o -e ./DA.e' --keep-going --rerun-incomplete
"""

import glob
import os
from subprocess import run
import pandas as pd
import re
from os.path import join

shell.prefix("set -euo pipefail;")

def _multi_arg_start(flag, files):
    flag += " "
    return " ".join(flag + f for f in files)

def _multi_arg_end(flag, files):
    flag = " "+flag
    return " ".join(f + flag for f in files)

def _multi_arg_both_ends(flag1, flag2, files):
    flag1 += " "
    flag2 = " "+flag2
    return " ".join(flag1 + f + flag2 for f in files)

def flattenChunk(chunk):
    return chunk.replace(":", "_").replace("-", "_")

def extendChunk(chunk):
    relChunk = chunk.pop()
    chunkSplitted = relChunk.split("_")
    return chunkSplitted[0]+":"+chunkSplitted[1]+"-"+chunkSplitted[2]

#Variables
chunkFile = '/nfs/leia/research/stegle/mjbonder/ChunkFiles/chr.txt'
genotypeFile = '/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed'
annotationFile = '/nfs/leia/research/stegle/dseaton/hipsci/singlecell_neuroseq/data/metadata/gene_annotation/Homo_sapiens.GRCh37.82.Limix_annotation_gene_level.txt'
phenotypeFile = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/phenotypes.tsv'
covariateFile = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/covariates.tsv'
kinshipFiles = '/hps/nobackup2/stegle/users/acuomo/hipsci_genotype_files/hipsci.wec.gtarray.HumanCoreExome.imputed_phased.20170327.genotypes.norm.renamed.kinship'
noiseTermFile = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/noise_matrix.tsv'
featureVariantFile = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/fvf.tsv'
numberOfPermutations = '1000'
minorAlleleFrequency = '0.05'
hwe = '0.000001'
callRate = '1'
windowSize = '250000'
blockSize = '15000'
outputFolder = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/'

finalQtlRun = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/top_qtl_results_all.txt'
finalQtlRun1 = '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/qtl_results_all.txt'

with open(chunkFile,'r') as f:
    chunks = [x.strip() for x in f.readlines()]

qtlOutput = []
for chunk in chunks:
    #print(chunk)
    processedChunk = flattenChunk(chunk)
    #print(processedChunk)
    processedChunk=expand(outputFolder+'{chunk}.finished',chunk=processedChunk )
    qtlOutput.append(processedChunk)

## flatten these lists
qtlOutput = [filename for elem in qtlOutput for filename in elem]

rule all:
    input:
        qtlOutput, finalQtlRun, finalQtlRun1

rule run_qtl_mapping:
    input:
        af = annotationFile,
        pf = phenotypeFile,
        cf = covariateFile,
        kf = kinshipFiles,
        rf = noiseTermFile,
        fvf = featureVariantFile
    output:
        '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/{chunk}.finished'
    params:
        gen = genotypeFile,
        od = outputFolder,
        np = numberOfPermutations,
        maf = minorAlleleFrequency,
        hwe = hwe,
        cr = callRate,
        w = windowSize,
        bs = blockSize
    run:
        chunkFull = extendChunk({wildcards.chunk})
        shell(
            "singularity exec /hps/nobackup2/stegle/users/acuomo/containers/limix206_qtl.simg python /hps/nobackup2/stegle/users/acuomo/tools/hipsci_pipeline/limix_QTL_pipeline/run_QTL_analysis.py "
            " --plink {params.gen} "
            " -af {input.af} "
            " -pf {input.pf} "
            " -cf {input.cf} "
            " -od {params.od} "
            " -fvf {input.fvf} "
            " -rf {input.kf},{input.rf} "
            " -gr {chunkFull} "
            " -np {params.np} "
            " -maf {params.maf} "
            " -hwe {params.hwe} "
            " -cr {params.cr} "
            " -c -gm standardize "
            " -w {params.w} "
            " --block_size {params.bs} ")
        shell("touch {output}")


rule aggregate_qtl_results:
    input:
        IF = outputFolder,
        OF = outputFolder,
        finalFiles = qtlOutput
    output:
        '/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/top_qtl_results_all.txt'
    run:
        shell(
            "/nfs/software/stegle/users/acuomo/conda-envs/limix_qtl/bin/python /hps/nobackup2/stegle/users/mjbonder/tools2/hipsci_pipeline/post-processing_QTL/minimal_postprocess.py "
            " -id {input.IF} "
            " -od {input.OF} "
            " -sfo -tfb ")

rule aggregate_qtl_results_all:
    input:
        IF = outputFolder,
        OF = outputFolder,
        finalFiles = qtlOutput
    output:
        "/hps/nobackup2/stegle/users/acuomo/all_scripts/struct_LMM2/sc_neuroseq/May2021/genetic_effect/MOFA10/flip_signs/input_files_ABHD12B-14_51328222_C_T_top20quantile/results/qtl_results_all.txt"
    run:
        shell(
            "/nfs/software/stegle/users/acuomo/conda-envs/limix_qtl/bin/python /hps/nobackup2/stegle/users/mjbonder/tools2/hipsci_pipeline/post-processing_QTL/minimal_postprocess.py "
            " -id {input.IF} "
            " -od {input.OF} "
            " -sfo -mrp 1 ")
