#!/usr/bin/sh

source /esat/jonen/tmp/envs/miniconda3/bin/activate pytorch_tmp

echo "path and evironment loaded"
export OMP_NUM_THREADS=1
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

cd /users/visics/wvangans/Documents/repos/nips_github/segmentation/
python linear_finetune.py \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj/linear/linear_deeplab.yml
