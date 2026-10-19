#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

// 输入：d_Q [seq_len, d_head], d_K [seq_len, d_head], d_V [seq_len, d_head]
// 输出：d_O [seq_len, d_head]
// 计算：O = softmax(Q * K^T / sqrt(d_head)) * V
const int TILE=32;
__global__ void flash_attention_kernel(float *d_Q, float*d_K, float*d_V, float*d_O,
    int seq_len, int d_head, float scale){

    constexpr int TILE = 32;
    constexpr int D_HEAD = 64;

    __shared__ float segQ[TILE * D_HEAD];
    __shared__ float segK[TILE * D_HEAD];
    __shared__ float segO[TILE * D_HEAD];
    __shared__ float segV[TILE * D_HEAD];
    __shared__ float mw[TILE][TILE];
    __shared__ float m_final[TILE];
    __shared__ float l_final[TILE];
    __shared__ float m_diff[TILE];
    __shared__ float segP[TILE * TILE];

    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < TILE; idx += blockDim.x * blockDim.y) {
        m_final[idx] = -INFINITY;
        l_final[idx] = 0;
    }
    __syncthreads();

    int q_start=blockIdx.x*TILE;
    for(int i=threadIdx.y*blockDim.x+threadIdx.x;i<TILE*d_head;i+=blockDim.x*blockDim.y){
        int row=i/d_head,col=i%d_head;
        int rg=q_start+row;
        segQ[i]=(rg<seq_len)?d_Q[rg*d_head+col]:0.0f;
        segO[i]=0.0f;
    }
    __syncthreads();

    for(int k=0;k<seq_len/TILE;k++){
        int k_start=k*TILE;
        for(int i=threadIdx.y*blockDim.x+threadIdx.x;i<TILE*d_head;i+=blockDim.x*blockDim.y){
            int row=i/d_head,col=i%d_head;
            int rg=k_start+row;
            segK[row*d_head+col]=(rg<seq_len)?d_K[rg*d_head+col]:0.0f;
            segV[row*d_head+col]=(rg<seq_len)?d_V[rg*d_head+col]:0.0f;
        }
        __syncthreads();

        float s[2][2]={0.0f};
        int row=threadIdx.y*2,col=threadIdx.x*2;
        for(int j=0;j<d_head;j++){
            for(int r=0;r<2;r++){
                for(int c=0;c<2;c++){
                    s[r][c]+=(segQ[(row+r)*d_head+j]*segK[(col+c)*d_head+j]*scale);
                }
            }
        }
        for (int r = 0; r < 2; r++) {
            int i = threadIdx.y * 2 + r;
            for (int c = 0; c < 2; c++) {
                int j = threadIdx.x * 2 + c;
                // 存单个元素的 max 和 exp
                mw[i][j] = s[r][c];              // 原始 score（或已减 max 前的）
                //lw[i][j] = 1.0f;                 // 占位，后面用 exp 算
            }
        }
        __syncthreads(); 

        int idx=threadIdx.y*blockDim.x+threadIdx.x;
        if(idx<TILE){
            m_diff[idx]=m_final[idx];
        }
        if (idx < TILE) {
            float m_new = m_final[idx];
            // 先遍历一遍，找到最终的 m_new
            for (int j = 0; j < TILE; j++) {
                m_new = fmaxf(m_new, mw[idx][j]);
            }
            // 再遍历一遍，用最终的 m_new 更新 l
            float l_new = l_final[idx] * __expf(m_final[idx] - m_new);
            for (int j = 0; j < TILE; j++) {
                l_new += __expf(mw[idx][j] - m_new);
            }
            m_final[idx] = m_new;
            l_final[idx] = l_new;
        }
        __syncthreads();

        for (int r = 0; r < 2; r++) {
            int i = threadIdx.y * 2 + r;
            for (int c = 0; c < 2; c++) {
                int j = threadIdx.x * 2 + c;
                segP[i*TILE+j] = __expf(s[r][c] - m_final[i]);
            }
        }
        __syncthreads();

        for(int i=threadIdx.y*blockDim.x+threadIdx.x;i<TILE*d_head;i+=blockDim.x*blockDim.y){
            int row=i/d_head;
            segO[i]*=exp(m_diff[row]-m_final[row]);
        }
        __syncthreads();

        float o[2][4]={0.0f};
        int real_r=threadIdx.y*2;
        int real_c=threadIdx.x*4;

        for(int j=0;j<TILE;j++){
            for(int r=0;r<2;r++){
                for(int c=0;c<4;c++){
                    o[r][c]+=(segP[(real_r+r)*TILE+j]*segV[j*d_head+real_c+c]);
                }
            }
        }

        for(int r=0;r<2;r++){
            for(int c=0;c<4;c++){
                segO[(real_r+r)*d_head+real_c+c]+=o[r][c];
            }
        }

        __syncthreads();
    }

    for(int i=threadIdx.y*blockDim.x+threadIdx.x;i<TILE*d_head;i+=blockDim.x*blockDim.y){
        int row=i/d_head,col=i%d_head;
        d_O[(q_start+row)*d_head+col]=segO[i]/l_final[row];
    }
}

void test_flash_attention(int seq_len, int d_head) {
    const float scale = 1.0f / sqrtf((float)d_head);

    // 1. Host 内存
    std::vector<float> h_Q(seq_len * d_head);
    std::vector<float> h_K(seq_len * d_head);
    std::vector<float> h_V(seq_len * d_head);
    std::vector<float> h_O(seq_len * d_head);

    for (int i = 0; i < seq_len * d_head; ++i) {
        h_Q[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
        h_K[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
        h_V[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    }

    // 2. Device 内存
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, seq_len * d_head * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_head * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_head * sizeof(float));
    cudaMalloc(&d_O, seq_len * d_head * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), seq_len * d_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), seq_len * d_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), seq_len * d_head * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((seq_len+TILE-1)/TILE,1); 
    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, seq_len, d_head, scale);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        flash_attention_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, seq_len, d_head, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("FlashAttention seq_len=%d d_head=%d: %.4f ms (avg over %d iters)\n",
           seq_len, d_head, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_O.data(), d_O, seq_len * d_head * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：CPU 参考实现
    int isCorrect = 1;
    for (int i = 0; i < seq_len && isCorrect; ++i) {
        // 计算第 i 行的注意力分数
        std::vector<float> scores(seq_len);
        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                dot += h_Q[i * d_head + d] * h_K[j * d_head + d];
            }
            scores[j] = dot * scale;
        }

        // Softmax
        float max_score = -1e30f;
        for (int j = 0; j < seq_len; ++j) {
            max_score = std::max(max_score, scores[j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j < seq_len; ++j) {
            scores[j] /= sum_exp;
        }

        // 计算输出 O[i] = sum_j P[i][j] * V[j]
        for (int d = 0; d < d_head; ++d) {
            float expected = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                expected += scores[j] * h_V[j * d_head + d];
            }
            float got = h_O[i * d_head + d];
            float diff = std::fabs(got - expected);
            float tol = 1e-2f * std::max(1.0f, std::fabs(expected));
            if (diff > tol) {
                printf("错误！ i=%d d=%d got=%.6f expected=%.6f diff=%.6f\n",
                       i, d, got, expected, diff);
                isCorrect = 0;
                break;
            }
        }
    }
    if (isCorrect) printf("正确！\n");

    // 7. 清理
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // test_flash_attention(128, 64);
    // test_flash_attention(512, 64);
    test_flash_attention(1024, 64);
    return 0;
}