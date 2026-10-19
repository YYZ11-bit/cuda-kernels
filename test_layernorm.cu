#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// 要求：输入 d_in [rows, cols]，gamma [cols]，beta [cols]
// 输出 d_out [rows, cols]
// 对每一行做：out = (x - mean) / sqrt(var + eps) * gamma + beta

__device__ void wrap_reduce(float& sum,float& sq_sum){
    for(int off=16;off>0;off>>=1){
        sum+=__shfl_down_sync(0xffffffff,sum,off);
        sq_sum+=__shfl_down_sync(0xffffffff,sq_sum,off);
    }
}
__global__ void layernorm_kernel(float* d_in,float* d_gamma,float* d_beta,float* d_out,int rows,int cols,float eqs){
    //int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int tid=threadIdx.x;
    int lane=tid&31,wid=tid>>5;
    float sum=0.0f,sq_sum=0.0f;
    // for(int i=tid;i<cols;i+=blockDim.x){
    //     float x=d_in[blockIdx.x*cols+i];
    //     sum+=x;
    //     sq_sum+=x*x;
    // }

    constexpr int ELEMS_PER_THREAD = 16;
    float reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int i = tid + k * blockDim.x;
        float x = (i < cols) ? d_in[blockIdx.x*cols + i] : 0.0f;
        sum += x;
        sq_sum += x * x;
        reg[k] = x;
    }
    __syncthreads();

    wrap_reduce(sum,sq_sum);
    __shared__ float mem_sum[32];
    __shared__ float mem_sqsum[32];
    if(lane==0){mem_sum[wid]=sum; mem_sqsum[wid]=sq_sum;}
    __syncthreads();

    int nw=(blockDim.x+31)>>5;
    sum=(tid<nw)?mem_sum[tid]:0.0f;
    sq_sum=(tid<nw)?mem_sqsum[tid]:0.0f;

    wrap_reduce(sum,sq_sum);
    __shared__ float final_sum,final_sqsum;
    if(tid==0){final_sum=sum; final_sqsum=sq_sum;}
    __syncthreads();

    float mean=final_sum/cols,var=final_sqsum/cols-mean*mean;
    float inv_std=1.0f/sqrt(var+eqs);
    // for(int i=tid;i<cols;i+=blockDim.x){
    //     float x=d_in[blockIdx.x*cols+i];
    //     d_out[blockIdx.x*cols+i]=(x-mean)*inv_std*d_gamma[i]+d_beta[i];
    // }

    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int i = tid + k * blockDim.x;
        if (i < cols) {
            d_out[blockIdx.x*cols + i] = (reg[k] - mean) * inv_std * d_gamma[i] + d_beta[i];
        }
    }
}

void test_layernorm(int rows, int cols) {
    const float eps = 1e-5f;

    // 1. Host 内存
    std::vector<float> h_in(rows * cols);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    std::vector<float> h_out(rows * cols);

    for (int i = 0; i < rows * cols; ++i) {
        h_in[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;  // -1.0 ~ 1.0
    }
    for (int i = 0; i < cols; ++i) {
        h_gamma[i] = 1.0f + (float)(rand() % 100) / 500.0f;  // 1.0 ~ 1.2
        h_beta[i] = (float)(rand() % 100) / 500.0f - 0.1f;    // -0.1 ~ 0.1
    }

    int threadPerBlock=256;
    int blocksPerGrid=rows;
    // 2. Device 内存
    float *d_in, *d_gamma, *d_beta, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        layernorm_kernel<<<blocksPerGrid,threadPerBlock>>>(d_in, d_gamma, d_beta, d_out, rows, cols, eps);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        layernorm_kernel<<<blocksPerGrid,threadPerBlock>>>(d_in, d_gamma, d_beta, d_out, rows, cols, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("LayerNorm rows=%d cols=%d: %.4f ms (avg over %d iters)\n",
           rows, cols, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：逐行对比 CPU 参考实现
    int isCorrect = 1;
    for (int r = 0; r < rows && isCorrect; ++r) {
        // CPU 参考：两遍扫描
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += h_in[r * cols + c];
        }
        float mean = sum / cols;

        float sq_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = h_in[r * cols + c] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / cols;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int c = 0; c < cols; ++c) {
            float expected = (h_in[r * cols + c] - mean) * inv_std * h_gamma[c] + h_beta[c];
            float got = h_out[r * cols + c];
            float diff = std::fabs(got - expected);
            float tol = 1e-3f * std::max(1.0f, std::fabs(expected));
            if (diff > tol) {
                printf("错误！ row=%d col=%d got=%.6f expected=%.6f diff=%.6f\n",
                       r, c, got, expected, diff);
                isCorrect = 0;
                break;
            }
        }
    }
    if (isCorrect) printf("正确！\n");

    // 7. 清理
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // test_layernorm(1, 1024);
    // test_layernorm(1024, 1024);
    test_layernorm(4096, 4096);
    return 0;
}