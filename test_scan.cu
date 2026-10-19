#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// 输入：d_in [N]
// 输出：d_out [N]，其中 d_out[i] = sum_{j=0}^{i} d_in[j]（包含前缀和）
__device__ float wrap_reduce(float val){
    for(int off=1;off<=16;off<<=1){
        float tmp=__shfl_up_sync(0xffffffff,val,off);
        if((threadIdx.x&31)>=off) val+=tmp;
    }
    return val;
}

__device__ float block_reduce(float val){
    __shared__ float smem[32];
    int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    val=wrap_reduce(val);
    if(lane==31) smem[wid]=val;
    __syncthreads();

    if(wid==0){
        float c=(lane<(blockDim.x>>5))?smem[lane]:0.0f;
        c=wrap_reduce(c);
        smem[lane]=c;
    }
    __syncthreads();
    if(wid>0) val+=smem[wid-1];
    return val;
}

__global__ void scan_kernel(float* in, float* out, int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    float val=(idx<N)?in[idx]:0.0f;
    val=block_reduce(val);
    out[idx]=val;
}

void test_scan(int N) {
    // 1. Host 内存
    std::vector<float> h_in(N);
    std::vector<float> h_out(N);
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)(rand() % 100) / 10.0f;  // 0.0 ~ 9.9
    }

    // 2. Device 内存
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threadPerBlock=256;
    int blockPerGrid=(N+threadPerBlock-1)/threadPerBlock;
    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        scan_kernel<<<blockPerGrid,threadPerBlock>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        scan_kernel<<<blockPerGrid,threadPerBlock>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Scan N=%d: %.4f ms (avg over %d iters)\n", N, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：逐元素对比
    int block_size = 256;   // 和你的 blockDim.x 一致
    int num_blocks = (N + block_size - 1) / block_size;
    int isCorrect = 1;

    for (int b = 0; b < num_blocks && isCorrect; ++b) {
        int start = b * block_size;
        int end = std::min(start + block_size, N);
        float running = 0.0f;
        for (int i = start; i < end; ++i) {
            running += h_in[i];
            float expected = running;
            float got = h_out[i];
            float diff = std::fabs(got - expected);
            float tol = 1e-3f * std::max(1.0f, std::fabs(expected));
            if (diff > tol) {
                printf("错误！ block=%d i=%d got=%.6f expected=%.6f diff=%.6f\n",
                       b, i, got, expected, diff);
                isCorrect = 0;
                break;
            }
        }
    }
    if (isCorrect) printf("正确！（每个 block 内的前缀和）\n");


    // 7. 清理
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_scan(1024);
    test_scan(1 << 20);   // 约 100 万
    test_scan(1 << 24);   // 约 1600 万
    return 0;
}