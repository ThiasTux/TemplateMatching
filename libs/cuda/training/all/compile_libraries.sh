#!/bin/sh

# Compile libraries for variable template generation
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_training.so wlcss_cuda_training.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_training_2d_enc.so wlcss_cuda_training_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_training_3d_enc.so wlcss_cuda_training_3d_enc.cu
