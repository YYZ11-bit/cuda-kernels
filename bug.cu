#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

// 完全照抄 LeetGPU 的写法
__device__ float warpReduceSum(float v){
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ float blockReduceSum(float v){
    static __shared__ float warp[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum(v);
    if (lane == 0) warp[wid] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = (threadIdx.x < nw) ? warp[lane] : 0.0f;
    if (wid == 0) v = warpReduceSum(v);
    return v;
}

__global__ void reduceKernel(const float* input, float* output, int N) {
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float v = ((i < N) ? input[i] : 0.0f)
            + ((i + blockDim.x < N) ? input[i + blockDim.x] : 0.0f);
    v = blockReduceSum(v);
    if (threadIdx.x == 0) output[blockIdx.x] = v;
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
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_partial, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_partial, N);
        reduceKernel<<<1, threadsPerBlock>>>(d_partial, d_out, blocksPerGrid);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_partial, N);
        reduceKernel<<<1, threadsPerBlock>>>(d_partial, d_out, blocksPerGrid);
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
    test_reduce(1 << 12);
    // test_reduce(1 << 20);
    // test_reduce(1 << 24);
    return 0;
}