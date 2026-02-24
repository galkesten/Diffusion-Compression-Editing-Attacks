#!/bin/bash
#
CONDA_HOME=~/miniconda3
CONDA_ENV=ddcm_extentions

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Check if the environment is activated
#echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
#echo "Python path: $(which python)"
#echo "Jupyter path: $(which jupyter)"

# Run Jupyter
/home/amit.vaisman/miniconda3/envs/ddcm_extentions/bin/jupyter lab --notebook-dir=./ --no-browser \
 --ip=$(hostname -I | cut -d " " -f1) --port-retries=100
