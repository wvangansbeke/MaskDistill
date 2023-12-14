#!/usr/bin/sh

# Set number of threads to prevent overloading your system
export OMP_NUM_THREADS=1
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

python multi_gpu.py --multiprocessing-distributed --rank 0 --world-size 1 \
    --config_env configs/env.yml \
    --config_exp configs/multi_obj_avg/deeplab.yml \
