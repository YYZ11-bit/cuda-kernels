#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

__device__ void reduce(float& m,float& d){
    for(int offset=16;offset>=1;offset=offset>>1){
        float m2=__shfl_down_sync(0xffffffff,m,offset);
        float d2=__shfl_down_sync(0xffffffff,d,offset);
        float m_new=fmax(m2,m);
        d=d*expf(m-m_new)+d2*expf(m2-m_new);
        m=m_new;
    }
}

__global__ void softmax_kernel(float* d_in, float* d_out, int rows, int col) {
    //int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int tid=threadIdx.x;
    int start=blockIdx.x*col;
    float maxx=-INFINITY;
    float sum=0.0f;

    constexpr int ELEMS_PER_THREAD = 16;
    constexpr int CACHE_SIZE = 8;          // 只缓存前 8 个
    float reg[CACHE_SIZE];
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int i = tid + k * blockDim.x;
        float num = (i < col) ? d_in[start + i] : -INFINITY;
        float max_new = fmaxf(num, maxx);
        sum = sum * expf(maxx - max_new) + expf(num - max_new);
        maxx = max_new;
        if (k < CACHE_SIZE) reg[k] = num;
    }
    reduce(maxx,sum);

    __shared__ float smem_m[32];
    __shared__ float smem_d[32];
    if(tid%32==0){
        smem_m[tid/32]=maxx;
        smem_d[tid/32]=sum;
    }
    __syncthreads();

    int N=blockDim.x/32;
    if(tid<32){
        maxx=(tid<N)?smem_m[tid]:-INFINITY;
        sum=(tid<N)?smem_d[tid]:0.0f;
        reduce(maxx,sum);
    }
    __syncthreads();

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = maxx;
        final_d = sum;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int i = tid + k * blockDim.x;
        if (i < col) {
            float val = (k < CACHE_SIZE) ? reg[k] : d_in[start + i];
            d_out[start + i] = expf(val - final_m) / final_d;
        }
    }
    // for(int i=tid;i<col;i+=blockDim.x){
    //     d_out[start+i]=expf(reg[now++]-final_m)/final_d;
    // }
}

void test_softmax(int rows, int cols) {
    // 1. Host 内存：随机初始化，每行数据不同
    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;  // -5.0 ~ 5.0
    }

    // 2. Device 内存
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = rows;
    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Softmax rows=%d cols=%d: %.4f ms (avg over %d iters)\n",
           rows, cols, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：逐行对比
    int isCorrect = 1;
    for (int r = 0; r < rows && isCorrect; ++r) {
        // CPU 参考：标准两遍 Softmax
        float max_val = -1e30f;
        for (int c = 0; c < cols; ++c) {
            max_val = std::max(max_val, h_in[r * cols + c]);
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum_exp += std::exp(h_in[r * cols + c] - max_val);
        }
        for (int c = 0; c < cols; ++c) {
            float expected = std::exp(h_in[r * cols + c] - max_val) / sum_exp;
            float got = h_out[r * cols + c];
            float diff = std::fabs(got - expected);
            float tol = 1e-4f * std::max(1.0f, std::fabs(expected));
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
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // test_softmax(1, 1024);       // 单行，短
    // test_softmax(1024, 1024);    // 多行，每行 1024
    test_softmax(4096, 4096);    // 大矩阵
    return 0;
}