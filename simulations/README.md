# Simulation experiments

This directory contains code for reproducing the simulation experiments. To execute the full set of simulations, run the Snakefile in this directory using ``snakemake``.

The workflow is based on the [parameter space exploration](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#parameter-space-exploration) template provided by Snakemake. 
The full set of parameters along with their default values can be found in ``scripts/settings.py``. Specific simulation settings are specified through ``params.csv``. For the simulation oputputs (_P_-values) Snakemake automatically
creates a directory structure reflecting the non-default parameter values to be explored. 

``scripts/simulate.py`` can also be executed directly, in which case all parameters are taken from ``scripts/settings.py``.

## Contents:

- ``scripts/settings.py`` Default simulation parameters.
- ``scripts/sim_utils.py`` Utility functions for the simulation experiments.
- ``scripts/simulate.py`` Main simulation script. 

The notebooks folder contains ``.ipynb`` files to generate the simulation settings as well as code to reproduce the figures.

- ``notebooks/save_params.ipynb`` Create ``params.csv`` used in this paper.
- ``notebooks/test_calibration.ipynb`` Calibration experiments.
- ``notebooks/test_power.ipynb`` Power experiments.
