#!/usr/bin/sh

export OMP_NUM_THREADS=1
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

python multi_gpu.py --multiprocessing-distributed --rank 0 --world-size 1 \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj/deeplab.yml \
    --crf-postprocess

python multi_gpu.py --multiprocessing-distributed --rank 0 --world-size 1 \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj/deeplab.yml \
    --crf-postprocess \
    --crf-train

python multi_gpu.py --multiprocessing-distributed --rank 0 --world-size 1 \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj/deeplab_plus_crf.yml \
    --crf-postprocess
