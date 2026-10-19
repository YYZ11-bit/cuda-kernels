#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<cmath>

using namespace std;

__global__ void vector_add_vec4(const float4* A, const float4* B, float4* C, int N4){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N4){
        float4 a=A[idx];
        float4 b=B[idx];
        float4 c;
        c.x=a.x+b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C[idx] = c;
    }
}

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        C[idx]=A[idx]+B[idx];
    }
}

constexpr int tile=64;
constexpr int threadNumx=16;
constexpr int ratio=tile/threadNumx;
#define FLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])
__global__ void gemm_naive(const float* A, const float* B, float* C, int M,int N,int K){
    __shared__ float As[tile][tile];
    __shared__ float Bs[tile][tile];
    
    int col=blockDim.x*blockIdx.x*ratio;
    int row=blockDim.y*blockIdx.y*ratio;
    int thx=threadIdx.x*ratio;
    int thy=threadIdx.y*ratio;
    
    float sum[ratio][ratio]={0.0f};
    for(int i=0;i<(K+tile-1)/tile;i++){

        for(int r=0;r<ratio;r++){
            // v5:寄存器分块
            for(int c=0;c<ratio;c++){
                int real_r=r+thy;
                int real_c=c+thx;
                As[real_r][real_c]=(row+real_r<M&&i*tile+real_c<K)?A[(row+real_r)*K+i*tile+real_c]:0.0f;
                Bs[real_r][real_c]=(i*tile+real_r<K&&col+real_c<N)?B[(i*tile+real_r)*N+col+real_c]:0.0f;
            }

            // v6:float4
            // int real_r=r+thy;
            // int real_c=thx;
            // float4 Af=FLOAT4(A[(row+real_r)*K+i*tile+real_c]);
            // float4 Bf=FLOAT4(B[(i*tile+real_r)*N+col+real_c]);
            // As[real_r][real_c]=Af.x;
            // As[real_r][real_c+1]=Af.y;
            // As[real_r][real_c+2]=Af.z;
            // As[real_r][real_c+3]=Af.w;
            // Bs[real_r][real_c]=Bf.x;
            // Bs[real_r][real_c+1]=Bf.y;
            // Bs[real_r][real_c+2]=Bf.z;
            // Bs[real_r][real_c+3]=Bf.w;
        }
        __syncthreads();

        for(int k=0;k<tile;k++){
            for(int r=0;r<ratio;r++){
                for(int c=0;c<ratio;c++){
                    int real_r=r+thy;
                    int real_c=c+thx;
                    sum[r][c]+=(As[real_r][k]*Bs[k][real_c]);
                }
            }
            
        }
        
        __syncthreads();
    }
    
    for(int r=0;r<ratio;r++){
        for(int c=0;c<ratio;c++){
            int out_r = row + thy + r;
            int out_c = col + thx + c;
            if (out_r < M && out_c < N) {
                C[out_r * N + out_c] = sum[r][c];
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(threadNumx, threadNumx);
    dim3 blocksPerGrid((N + threadsPerBlock.x * ratio - 1) / (threadsPerBlock.x * ratio),
                       (M + threadsPerBlock.y * ratio - 1) / (threadsPerBlock.y * ratio));
    gemm_naive<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}


void test1D(int N){
    float* A,* B,* C;
    A=(float*)malloc(N*sizeof(float));
    B=(float*)malloc(N*sizeof(float));
    C=(float*)malloc(N*sizeof(float));

    for(int i=0;i<N;i++){
        A[i]=i*1.5+1;
        B[i]=-i*0.5;
    }

    float* A_d,* B_d,* C_d;
    size_t bytes = N * sizeof(float);
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);
    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int N4 = N/4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N4 + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < 10; i++) {
        vector_add_vec4<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<const float4*>(A_d), 
        reinterpret_cast<const float4*>(B_d),
        reinterpret_cast<float4*>(C_d), 
        N/4);
    }
    cudaDeviceSynchronize();

    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        //vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
        vector_add_vec4<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<const float4*>(A_d), 
        reinterpret_cast<const float4*>(B_d),
        reinterpret_cast<float4*>(C_d), 
        N/4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("N=%d, Kernel time: %.4f ms (avg over %d iters)\n", N, ms / iters, iters);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    //验证部分
    int isCorrect=1;
    for(int i=0;i<N;i++){
        float expected = A[i] + B[i];
        float diff = std::fabs(C[i] - expected);
        float tol = 1e-5f * std::max(1.0f, std::fabs(expected));
        if (diff > tol) isCorrect = 0;
    } 

    if(isCorrect) printf("正确！\n");
    else printf("错误！\n");
    
    free(A);
    free(B);
    free(C);
}

void test2D(int M, int N, int K) {
    // Host 内存
    float *A, *B, *C;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));
    C = (float*)malloc(M * N * sizeof(float));

    // 初始化（用简单规律数据，便于调试）
    for (int i = 0; i < M * K; ++i) A[i] = (i % 7) * 0.1f + 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = (i % 5) * 0.2f - 0.5f;

    // Device 内存
    float *A_d, *B_d, *C_d;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    cudaMalloc(&A_d, bytesA);
    cudaMalloc(&B_d, bytesB);
    cudaMalloc(&C_d, bytesC);
    cudaMemcpy(A_d, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytesB, cudaMemcpyHostToDevice);

    // 预热
    for (int i = 0; i < 5; ++i) solve(A_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();

    // 计时
    const int iters = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) solve(A_d, B_d, C_d, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GEMM %d x %d x %d: %.3f ms (avg over %d iters)\n",
           M, N, K, ms / iters, iters);

    // 拷回并验证
    cudaMemcpy(C, C_d, bytesC, cudaMemcpyDeviceToHost);

    int isCorrect = 1;
    for (int row = 0; row < M && isCorrect; ++row) {
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += A[row * K + k] * B[k * N + col];
            }
            float got = C[row * N + col];
            float diff = std::fabs(got - expected);
            float tol = 1e-3f * std::max(1.0f, std::fabs(expected));
            if (diff > tol) { isCorrect = 0; break; }
        }
    }
    printf(isCorrect ? "正确！\n" : "错误！\n");

    // 清理
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    free(A); free(B); free(C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    //test2D(1<<10,1<<10,1<<10);
    test2D(2, 2, 2);       // 应该过
test2D(64, 64, 64);    // 应该过
test2D(65, 65, 65);    // 非 64 倍数
test2D(100, 100, 100); // 非 64 倍数
test2D(127, 127, 127); // 非 64 倍数
test2D(128, 128, 128); // 64 倍数
test2D(1, 1, 1);       // 最小
test2D(2, 64, 64);     // M 小，N 和 K 大
test2D(64, 2, 64);     // N 小
test2D(64, 64, 2);     // K 小
    return 0;
}