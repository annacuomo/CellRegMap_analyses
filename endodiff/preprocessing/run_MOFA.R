library(rhdf5)
library(Rhdf5lib)
library(MOFA2)
library(reticulate)

#use_condaenv("mypy3")
use_python("/nfs/software/stegle/users/acuomo/conda-envs/mypy3/bin/python", required=TRUE)
reticulate::py_config()

sce = readRDS("/hps/nobackup/hipsci/scratch/singlecell_endodiff/data_processed/merged/20180618/sce_merged_afterqc_filt_allexpts.rds")
sce

data = list(scRNAseq = logcounts(sce[rowData(sce)$is_intop500hvg,]))
MOFAobject <- create_mofa(data)

data_opts <- get_default_data_options(MOFAobject)

model_opts <- get_default_model_options(MOFAobject)
model_opts$num_factors <- 10
model_opts$spikeslab_weights <- FALSE

train_opts <- get_default_training_options(MOFAobject)
train_opts$convergence_mode <- "medium"
train_opts$seed <- 42

#########################
## Prepare MOFA object ##
#########################

MOFAobject <- prepare_mofa(MOFAobject,
  data_options = data_opts,
  model_options = model_opts,
  training_options = train_opts
  # stochastic_options = stochastic_opts
)

#####################
## Train the model ##
#####################

outfile <- "/hps/nobackup/stegle/users/acuomo/all_scripts/struct_LMM2/sc_endodiff/mofa_logcounts_model.hdf5"
MOFAmodel <- run_mofa(MOFAobject, outfile)
