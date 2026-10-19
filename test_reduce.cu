#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

__device__ float wrap_reduce(float v){
    for(int off=16;off>0;off>>=1){
        v+=__shfl_down_sync(0xffffffff,v,off);
    }
    return v;
}

__global__ void reduce_kernel(const float* d_in, float* d_out, int N) {
    __shared__ float smem[32];
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid=threadIdx.x;
    int lane=tid&31,wid=tid>>5;
    float v = ((idx < N) ? d_in[idx] : 0.0f)
            + ((idx + blockDim.x < N) ? d_in[idx + blockDim.x] : 0.0f);

    v=wrap_reduce(v);
    if(lane==0) smem[wid]=v;
    __syncthreads();
    int nw=(blockDim.x+31)>>5;
    v=(tid<nw)?smem[tid]:0.0f;
    if(wid==0) v=wrap_reduce(v);
    
    if (threadIdx.x == 0) d_out[blockIdx.x] = v;
}

void test_reduce(int N) {
    // 1. Host 内存：全 1 数组，期望结果就是 N
    std::vector<float> h_in(N, 1.0f);
    float h_out = 0.0f;

    // 2. Device 内存
    float* d_in;
    float* d_partial;
    float* d_out;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+2*threadsPerBlock-1)/(2*threadsPerBlock);

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_partial, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        // 第一次启动：N 个元素 -> blocksPerGrid 个部分和
        reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_partial, N);
        // 第二次启动：blocksPerGrid 个部分和 -> 1 个最终结果
        reduce_kernel<<<1, threadsPerBlock>>>(d_partial, d_out, blocksPerGrid);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_partial, N);
        reduce_kernel<<<1, threadsPerBlock>>>(d_partial, d_out, blocksPerGrid);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Reduce N=%d: %.4f ms (avg over %d iters)\n", N, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证
    float expected = (float)N;
    float diff = std::fabs(h_out - expected);
    float tol = 1e-3f * std::max(1.0f, std::fabs(expected));
    if (diff <= tol) {
        printf("正确！ (got %.6f, expected %.6f)\n", h_out, expected);
    } else {
        printf("错误！ (got %.6f, expected %.6f, diff %.6f)\n", h_out, expected, diff);
    }

    // 7. 清理
    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    test_reduce(1 << 6);   
    // test_reduce(1 << 20);   // 约 100 万
    // test_reduce(1 << 24);   // 约 1600 万
    return 0;
}

