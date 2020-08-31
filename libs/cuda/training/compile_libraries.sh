#!/bin/sh

# Compile libraries for parameters optimization
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_params_training_optimized.so wlcss_cuda_params_training_opt.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_params_training_optimized_2d_enc.so wlcss_cuda_params_training_opt_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_params_training_optimized_3d_enc.so wlcss_cuda_params_training_opt_3d_enc.cu

# Compile libraries for template generation
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_templates_training_optimized.so wlcss_cuda_templates_generation_opt.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_templates_training_optimized_2d_enc.so wlcss_cuda_templates_generation_opt_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o lib_wlcss_cuda_templates_training_optimized_3d_enc.so wlcss_cuda_templates_generation_opt_3d_enc.cu
