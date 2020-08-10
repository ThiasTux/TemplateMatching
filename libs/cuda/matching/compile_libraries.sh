#!/bin/sh

# Compile libraries for parameters optimization
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda.so wlcss_cuda.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_2d.so wlcss_cuda_2d.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_3d.so wlcss_cuda_3d.cu