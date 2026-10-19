#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// 你的 transpose kernel 放这里
// 要求：输入 d_in [rows, cols]，输出 d_out [cols, rows]
// d_out[c * rows + r] = d_in[r * cols + c]
#define TILE 32
__global__ void transpose_kernel(float* d_in, float* d_out, int rows, int cols){
    __shared__ float smem[TILE][TILE+1];
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int y=blockDim.y*blockIdx.y+threadIdx.y;

    if(x<cols&&y<rows) smem[threadIdx.y][threadIdx.x]=d_in[y*cols+x]; 
    __syncthreads();
    x=blockDim.y*blockIdx.y+threadIdx.x;
    y=blockDim.x*blockIdx.x+threadIdx.y;
    if(x<rows&&y<cols) d_out[y*rows+x]=smem[threadIdx.x][threadIdx.y];

}

void test_transpose(int rows, int cols) {
    // 1. Host 内存：随机初始化
    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        h_in[i] = (float)(rand() % 1000) / 100.0f;
    }

    // 2. Device 内存
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadPerBlock(TILE,TILE),blocksPerGrid((cols+TILE-1)/TILE,(rows+TILE-1)/TILE);
    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        transpose_kernel<<<blocksPerGrid,threadPerBlock>>>(d_in, d_out, rows, cols);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        transpose_kernel<<<blocksPerGrid,threadPerBlock>>>(d_in, d_out, rows, cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Transpose rows=%d cols=%d: %.4f ms (avg over %d iters)\n",
           rows, cols, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：d_out[c * rows + r] == h_in[r * cols + c]
    int isCorrect = 1;
    for (int r = 0; r < rows && isCorrect; ++r) {
        for (int c = 0; c < cols; ++c) {
            float expected = h_in[r * cols + c];
            float got = h_out[c * rows + r];
            if (std::fabs(got - expected) > 1e-6f) {
                printf("错误！ row=%d col=%d got=%.6f expected=%.6f\n",
                       r, c, got, expected);
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
    // test_transpose(1024, 1024);
    // test_transpose(4096, 4096);
    test_transpose(2048, 8192);   // 非方阵
    return 0;
}