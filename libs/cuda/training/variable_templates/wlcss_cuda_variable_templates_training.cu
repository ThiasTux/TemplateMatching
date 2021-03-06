#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int32_t *d_mss, *d_mss_offsets, *d_ts, *d_ss, *d_tlen, *d_toffsets, *d_slen, *d_soffsets, *d_params;
int num_templates, num_streams, num_params_sets, h_ts_length, h_ss_length, h_mss_length;

__global__ void wlcss_cuda_kernel(int32_t *d_mss, int32_t *d_mss_offsets, int32_t *d_ts, int32_t *d_ss, int32_t *d_tlen, int32_t *d_toffsets, int32_t *d_slen, int32_t *d_soffsets, int32_t *d_params){

    int32_t params_idx = threadIdx.x;
    int32_t template_idx = blockIdx.x;
    int32_t stream_idx = blockIdx.y;

    int32_t t_len = d_tlen[template_idx];
    int32_t s_len = d_slen[stream_idx];

    int32_t t_offset = d_toffsets[template_idx];
    int32_t s_offset = d_soffsets[stream_idx];

    int32_t d_mss_offset = d_mss_offsets[params_idx*gridDim.x*gridDim.y+template_idx*gridDim.y+stream_idx];
    int32_t *mss = &d_mss[d_mss_offset];

    int32_t *tmp_window = new int32_t[(t_len + 2)]();

    int32_t *t = &d_ts[t_offset];
    int32_t *s = &d_ss[s_offset];

    int32_t reward = d_params[params_idx*3];
    int32_t penalty = d_params[params_idx*3+1];
    int32_t accepteddist = d_params[params_idx*3+2];

    int32_t tmp = 0;

    for(int32_t j=0;j<s_len;j++){
        for(int32_t i=0;i<t_len;i++){
            int32_t distance = abs(s[j]-t[i]);
            if (distance <= accepteddist){
                tmp = tmp_window[i]+reward;
            } else{
                tmp = max(tmp_window[i]-penalty*distance,
                            max(tmp_window[i+1]-penalty*distance,
                                tmp_window[t_len+1]-penalty*distance));
            }
            tmp_window[i] = tmp_window[t_len+1];
            tmp_window[t_len+1] = tmp;
        }
        tmp_window[t_len] = tmp_window[t_len+1];
        mss[j] = tmp_window[t_len+1];
        tmp_window[t_len+1] = 0;
    }
    delete [] tmp_window;
}

extern "C"{
    void wlcss_cuda_init(int32_t *h_mss, int32_t *h_mss_offsets, 
                         int32_t *h_ss, int32_t *h_slen, int32_t *h_soffsets, 
                         int32_t *h_params, int num_ts, int num_ss, int num_ps, int h_ss_len, int h_mss_len){

        num_templates = num_ts;
        num_streams = num_ss;
        num_params_sets = num_ps;
        h_ss_length = h_ss_len;
        h_mss_length = h_mss_len;

        // Allocate memory for streams array
        gpuErrchk( cudaMalloc((void **) &d_ss, h_ss_length * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_ss, h_ss, h_ss_length * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Allocate memory for streams lengths
        gpuErrchk( cudaMalloc((void **) &d_slen, num_streams * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_slen, h_slen, num_streams * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Allocate memory for streams offsets
        gpuErrchk( cudaMalloc((void **) &d_soffsets, num_streams * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_soffsets, h_soffsets, num_streams * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Allocate memory for params array
        gpuErrchk( cudaMalloc((void **) &d_params, num_params_sets * 3 * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_params, h_params, num_params_sets * 3 * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Allocate memory for matching scores
        gpuErrchk( cudaMalloc((void **) &d_mss, h_mss_length * sizeof(int32_t)) );

        // Allocate memory for matching scores offsets
        gpuErrchk( cudaMalloc((void **) &d_mss_offsets, num_streams*num_templates*num_params_sets * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_mss_offsets, h_mss_offsets, num_streams*num_templates*num_params_sets * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Allocate memory for templates lengths
        gpuErrchk( cudaMalloc((void **) &d_tlen, num_templates * sizeof(int32_t)) );
        // Allocate memory for templates offsets
        gpuErrchk( cudaMalloc((void **) &d_toffsets, num_templates * sizeof(int32_t)) );

    }

    void wlcss_cuda(int32_t *h_ts, int32_t *h_tlen, int32_t *h_toffsets, int h_ts_len, int32_t *h_mss){
        
        h_ts_length = h_ts_len;

        /// Allocate memory for templates array
        gpuErrchk( cudaMalloc((void **) &d_ts, h_ts_length * sizeof(int32_t)) );
        gpuErrchk( cudaMemcpy(d_ts, h_ts, h_ts_length * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Initialize memory for templates lengths
        gpuErrchk( cudaMemcpy(d_tlen, h_tlen, num_templates * sizeof(int32_t), cudaMemcpyHostToDevice) );

        // Initialize memory for templates offsets
        gpuErrchk( cudaMemcpy(d_toffsets, h_toffsets, num_templates * sizeof(int32_t), cudaMemcpyHostToDevice) );
        
        // Initialize memory for matching scores
        gpuErrchk( cudaMemcpy(d_mss, h_mss, h_mss_length * sizeof(int32_t), cudaMemcpyHostToDevice) );

        wlcss_cuda_kernel<<<dim3(num_templates, num_streams), num_params_sets>>>(d_mss, d_mss_offsets, d_ts, d_ss, d_tlen, d_toffsets, d_slen, d_soffsets, d_params);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( cudaMemcpy(h_mss, d_mss, h_mss_length * sizeof(int32_t), cudaMemcpyDeviceToHost) );

        cudaFree(d_ts);
    }
    
    void wlcss_freemem(){
        
        cudaFree(d_tlen);
        cudaFree(d_toffsets);
        
        cudaFree(d_ss);
        cudaFree(d_slen);
        cudaFree(d_soffsets);
        
        cudaFree(d_mss);
        cudaFree(d_mss_offsets);
        cudaFree(d_params);
    }
}
