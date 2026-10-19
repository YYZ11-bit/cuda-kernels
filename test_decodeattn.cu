#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

// 你的 decode_attention kernel 放这里
// 输入：d_Q [1, d_head], d_K [seq_len, d_head], d_V [seq_len, d_head]
// 输出：d_O [1, d_head]
// 计算：O = softmax(Q * K^T / sqrt(d_head)) * V
__device__ void warp_reduce_ml(float& m,float& l){
    for(int off=16;off>=1;off>>=1){
        float m_down=__shfl_down_sync(0xffffffff,m,off);
        float l_down=__shfl_down_sync(0xffffffff,l,off);
        float m_new=fmaxf(m,m_down);
        l=l*exp(m-m_new)+l_down*exp(m_down-m_new);
        m=m_new;
    }
}

__global__ void decode_attention_stage1(
    float* d_Q, float* d_K, float* d_V, float* d_O_partial,
    float* m_out, float* l_out,
    int seq_len, int d_head, float scale)
{
    constexpr int TILE = 32;   // 每个 block 处理的 key 数
    constexpr int D_HEAD_MAX = 128;
    __shared__ float Q_sh[D_HEAD_MAX];   // D_HEAD_MAX = 128 或 256
    __shared__ float m_block[32];
    __shared__ float l_block[32];
    __shared__ float m_final_smem, l_final_smem;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    // 加载 Q 到共享内存
    for (int i = tid; i < d_head; i += blockDim.x) {
        Q_sh[i] = d_Q[i];
    }
    __syncthreads();

    // 每个线程负责一个 key
    int key_idx = blockIdx.x * TILE + tid;
    float m = -INFINITY;
    float l = 0.0f;
    float o_local = 0.0f;   // 每个线程负责一个 head 维度的部分 O

    if (key_idx < seq_len) {
        // 算 score
        float score = 0.0f;
        for (int i = 0; i < d_head; i++) {
            score += Q_sh[i] * d_K[key_idx * d_head + i];
        }
        score *= scale;
        m = score;
        l = 1.0f;
    }

    // 这个 block 内所有 key 的 max 和 sum
    // 但每个线程只负责一个 key，需要跨线程归约
    // 这里用 warp_reduce_ml 做归约
    warp_reduce_ml(m, l);
    if (lane == 0) { m_block[wid] = m; l_block[wid] = l; }
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (wid == 0) {
        m = (lane < num_warps) ? m_block[lane] : -INFINITY;
        l = (lane < num_warps) ? l_block[lane] : 0.0f;
        warp_reduce_ml(m, l);
        if (lane == 0) {
            m_final_smem = m;
            l_final_smem = l;
        }
    }
    __syncthreads();

    float m_final = m_final_smem;
    float l_final = l_final_smem;

    // 每个 block 的局部 max 和 sum 写到全局
    if (tid == 0) {
        m_out[blockIdx.x] = m_final;
        l_out[blockIdx.x] = l_final;
    }

    // 算局部 O：每个线程负责一个 head 维度
    // 但这里每个线程之前负责一个 key，现在要负责一个 head 维度
    // 需要重新分配线程
    for (int d = tid; d < d_head; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < TILE; j++) {
            int key = blockIdx.x * TILE + j;
            if (key < seq_len) {
                float score = 0.0f;
                for (int i = 0; i < d_head; i++) {
                    score += Q_sh[i] * d_K[key * d_head + i];
                }
                score *= scale;
                float p = __expf(score - m_final);
                o += p * d_V[key * d_head + d];
            }
        }
        d_O_partial[blockIdx.x * d_head + d] = o;
    }
}

__global__ void decode_attention_stage2(
    float* d_O_partial, float* m_out, float* l_out, float* d_O,
    int num_blocks, int d_head)
{
    // 先算全局 max 和 sum
    __shared__ float m_global, l_global;

    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int b = 0; b < num_blocks; b++) {
            m = fmaxf(m, m_out[b]);
        }
        float l = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            l += l_out[b] * __expf(m_out[b] - m);
        }
        m_global = m;
        l_global = l;
    }
    __syncthreads();

    // 每个线程负责一个 head 维度
    for (int d = threadIdx.x; d < d_head; d += blockDim.x) {
        float o = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            o += d_O_partial[b * d_head + d] * __expf(m_out[b] - m_global);
        }
        d_O[d] = o / l_global;
    }
}

void test_decode_attention(int seq_len, int d_head) {
    const float scale = 1.0f / sqrtf((float)d_head);
    constexpr int TILE = 32;

    // 1. Host 内存
    std::vector<float> h_Q(d_head);
    std::vector<float> h_K(seq_len * d_head);
    std::vector<float> h_V(seq_len * d_head);
    std::vector<float> h_O(d_head);

    for (int i = 0; i < d_head; ++i) {
        h_Q[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    }
    for (int i = 0; i < seq_len * d_head; ++i) {
        h_K[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
        h_V[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    }

    // 2. Device 内存
    int num_blocks = (seq_len + TILE - 1) / TILE;

    float *d_Q, *d_K, *d_V, *d_O;
    float *d_O_partial;
    float *m_out, *l_out;

    cudaMalloc(&d_Q, d_head * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_head * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_head * sizeof(float));
    cudaMalloc(&d_O, d_head * sizeof(float));
    cudaMalloc(&d_O_partial, num_blocks * d_head * sizeof(float));
    cudaMalloc(&m_out, num_blocks * sizeof(float));
    cudaMalloc(&l_out, num_blocks * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), d_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), seq_len * d_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), seq_len * d_head * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 预热
    for (int i = 0; i < 5; ++i) {
        decode_attention_stage1<<<num_blocks, 256>>>(
            d_Q, d_K, d_V, d_O_partial, m_out, l_out, seq_len, d_head, scale);
        decode_attention_stage2<<<1, 256>>>(
            d_O_partial, m_out, l_out, d_O, num_blocks, d_head);
    }
    cudaDeviceSynchronize();

    // 4. 计时
    const int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        decode_attention_stage1<<<num_blocks, 256>>>(
            d_Q, d_K, d_V, d_O_partial, m_out, l_out, seq_len, d_head, scale);
        decode_attention_stage2<<<1, 256>>>(
            d_O_partial, m_out, l_out, d_O, num_blocks, d_head);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Decode Attention seq_len=%d d_head=%d: %.4f ms (avg over %d iters)\n",
           seq_len, d_head, ms / iters, iters);

    // 5. 拷回结果
    cudaMemcpy(h_O.data(), d_O, d_head * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证：CPU 参考实现
    // 6.1 算 scores[j] = Q · K[j] * scale
    std::vector<float> scores(seq_len);
    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            dot += h_Q[d] * h_K[j * d_head + d];
        }
        scores[j] = dot * scale;
    }

    // 6.2 Softmax
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

    // 6.3 O[d] = sum_j P[j] * V[j][d]
    int isCorrect = 1;
    for (int d = 0; d < d_head; ++d) {
        float expected = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            expected += scores[j] * h_V[j * d_head + d];
        }
        float got = h_O[d];
        float diff = std::fabs(got - expected);
        float tol = 1e-2f * std::max(1.0f, std::fabs(expected));
        if (diff > tol) {
            printf("错误！ d=%d got=%.6f expected=%.6f diff=%.6f\n",
                   d, got, expected, diff);
            isCorrect = 0;
            break;
        }
    }
    if (isCorrect) printf("正确！\n");

    // 7. 清理
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_O_partial);
    cudaFree(m_out);
    cudaFree(l_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_decode_attention(128, 64);
    test_decode_attention(512, 64);
    test_decode_attention(1024, 64);
    test_decode_attention(4096, 128);   // 长序列，大 head
    return 0;
}