#!/bin/sh

# Compile libraries for parameters optimization
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o params/lib_wlcss_cuda_params_training.so params/wlcss_cuda_params_training.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o params/lib_wlcss_cuda_params_training_2d_enc.so params/wlcss_cuda_params_training_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o params/lib_wlcss_cuda_params_training_3d_enc.so params/wlcss_cuda_params_training_3d_enc.cu

# Compile libraries for template generation
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o templates/lib_wlcss_cuda_templates_training.so templates/wlcss_cuda_templates_training.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o templates/lib_wlcss_cuda_templates_training_2d_enc.so templates/wlcss_cuda_templates_training_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o templates/lib_wlcss_cuda_templates_training_3d_enc.so templates/wlcss_cuda_templates_training_3d_enc.cu

# Compile libraries for variable template generation
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o variable_templates/lib_wlcss_cuda_variable_templates_training.so variable_templates/wlcss_cuda_variable_templates_training.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o variable_templates/lib_wlcss_cuda_variable_templates_training_2d_enc.so variable_templates/wlcss_cuda_variable_templates_training_2d_enc.cu
nvcc -Xcompiler -fPIC -shared --ptxas-options=-v -lineinfo -o variable_templates/lib_wlcss_cuda_variable_templates_training_3d_enc.so variable_templates/wlcss_cuda_variable_templates_training_3d_enc.cu

