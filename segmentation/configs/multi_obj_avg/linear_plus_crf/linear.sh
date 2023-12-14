#!/usr/bin/sh

export OMP_NUM_THREADS=1
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

python linear_finetune.py \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj_avg/linear_plus_crf/linear_deeplab_plus_crf.yml \
    --crf-postprocess
