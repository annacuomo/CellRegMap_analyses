snakemake --rerun-incomplete --cluster "bsub -n {threads} -R rusage[mem={resources.mem_mb}] -e lsf_reports/T-%J.err -o lsf_reports/T-%J.out -q short"  -j 25 #-T 2
