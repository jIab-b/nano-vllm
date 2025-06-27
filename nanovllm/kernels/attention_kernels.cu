#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "attention_kernels.h"
#include <ATen/ATen.h>

namespace cg = cooperative_groups;

// Define constants for tiling dimensions. These can be tuned for different GPU architectures.
constexpr int TILE_SIZE_Q = 16;
constexpr int TILE_SIZE_K = 16;
constexpr int BLOCK_THREADS = 128;

// --- KERNEL IMPLEMENTATION: Paged Attention (Decode) ---
__global__ void paged_attention_kernel(
    at::Half* __restrict__ out,      // [batch_size, num_heads, head_dim]
    const at::Half* __restrict__ q,  // [batch_size, num_heads, head_dim]
    const at::Half* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const at::Half* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_table, // [batch_size, max_blocks_per_seq]
    const int* __restrict__ context_lens, // [batch_size]
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_len = context_lens[batch_idx];
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Shared memory for the query vector and accumulator
    __shared__ at::Half q_tile[TILE_SIZE_Q];
    __shared__ float acc_tile[TILE_SIZE_Q];

    // Load query into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_tile[i] = q[batch_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    __syncthreads();

    // Initialize accumulator, max_score, and sum_exp
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        acc_tile[i] = 0.0f;
    }
    __syncthreads();

    // Loop over the key/value sequence in tiles
    for (int k_start = 0; k_start < seq_len; k_start += TILE_SIZE_K) {
        __shared__ at::Half k_tile[TILE_SIZE_K * TILE_SIZE_Q]; // K_tile is TILE_SIZE_K x head_dim

        // Load K tile from paged cache into shared memory
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
            int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                int block_idx = block_table[batch_idx * (seq_len / block_size + 1) + token_idx / block_size];
                int block_offset = token_idx % block_size;
                k_tile[ (i/head_dim)*head_dim + dim_idx ] = k_cache[block_idx * block_size * num_kv_heads * head_dim + block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        // Compute scores for the tile
        __shared__ float scores[TILE_SIZE_K];
        for (int i = threadIdx.x; i < TILE_SIZE_K; i += blockDim.x) {
            float score = 0.0f;
            for (int j = 0; j < head_dim; ++j) {
                score += __half2float(q_tile[j]) * __half2float(k_tile[i * head_dim + j]);
            }
            scores[i] = score * scale;
        }
        __syncthreads();

        // Online softmax update
        float old_max = max_score;
        for (int i = 0; i < TILE_SIZE_K; ++i) {
            if (k_start + i < seq_len && scores[i] > max_score) {
                max_score = scores[i];
            }
        }
        if (old_max > -FLT_MAX) {
             sum_exp *= expf(old_max - max_score);
             for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
                acc_tile[i] *= expf(old_max - max_score);
            }
        }
       
        __shared__ at::Half v_tile[TILE_SIZE_K * TILE_SIZE_Q];
        // Load V tile
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
             int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                int block_idx = block_table[batch_idx * (seq_len / block_size + 1) + token_idx / block_size];
                int block_offset = token_idx % block_size;
                v_tile[ (i/head_dim)*head_dim + dim_idx ] = v_cache[block_idx * block_size * num_kv_heads * head_dim + block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        // Accumulate output
        for (int i = 0; i < TILE_SIZE_K; ++i) {
            if (k_start + i < seq_len) {
                float current_exp = expf(scores[i] - max_score);
                sum_exp += current_exp;
                for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                    acc_tile[j] += current_exp * __half2float(v_tile[i * head_dim + j]);
                }
            }
        }
        __syncthreads();
    }

    // Write final output
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        out[batch_idx * num_heads * head_dim + head_idx * head_dim + i] = __float2half(acc_tile[i] / sum_exp);
    }
}


// --- KERNEL IMPLEMENTATION: Variable Length Attention (Prefill) ---
__global__ void variable_length_attention_kernel(
    at::Half* __restrict__ out,      // [total_tokens, num_heads, head_dim]
    const at::Half* __restrict__ q,  // [total_tokens, num_heads, head_dim]
    const at::Half* __restrict__ k,  // [total_tokens, num_kv_heads, head_dim]
    const at::Half* __restrict__ v,  // [total_tokens, num_kv_heads, head_dim]
    const int* __restrict__ cu_seqlens, // [batch_size + 1]
    float scale,
    int total_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim) {

    const int q_idx = blockIdx.x * TILE_SIZE_Q + threadIdx.y;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Find sequence boundaries for the current query token
    int seq_idx = 0;
    while (q_idx >= cu_seqlens[seq_idx + 1]) {
        seq_idx++;
    }
    const int seq_start = cu_seqlens[seq_idx];
    const int seq_len = cu_seqlens[seq_idx + 1] - seq_start;

    if (q_idx >= total_tokens) return;

    // Shared memory for Q tile and accumulator
    __shared__ at::Half q_tile[TILE_SIZE_Q * TILE_SIZE_Q];
    __shared__ float acc_tile[TILE_SIZE_Q * TILE_SIZE_Q];

    // Load Q tile
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_tile[threadIdx.y * head_dim + i] = q[q_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    
    // Init accumulator
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        acc_tile[threadIdx.y * head_dim + i] = 0.0f;
    }
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    __syncthreads();

    // Loop over key/value sequence in tiles
    for (int k_start = 0; k_start < seq_len; k_start += TILE_SIZE_K) {
        __shared__ at::Half k_tile[TILE_SIZE_K * TILE_SIZE_Q];
        __shared__ at::Half v_tile[TILE_SIZE_K * TILE_SIZE_Q];

        // Load K and V tiles
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
            int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                k_tile[(i/head_dim)*head_dim + dim_idx] = k[(seq_start + token_idx) * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
                v_tile[(i/head_dim)*head_dim + dim_idx] = v[(seq_start + token_idx) * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        // Compute scores
        for (int k_local_idx = 0; k_local_idx < TILE_SIZE_K; ++k_local_idx) {
            int k_global_idx = k_start + k_local_idx;
            if (k_global_idx > (q_idx - seq_start)) continue; // Causal mask

            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score += __half2float(q_tile[threadIdx.y * head_dim + i]) * __half2float(k_tile[k_local_idx * head_dim + i]);
            }
            score *= scale;

            // Online softmax
            if (score > max_score) {
                float old_max = max_score;
                max_score = score;
                sum_exp *= expf(old_max - max_score);
                for (int i = 0; i < head_dim; ++i) {
                    acc_tile[threadIdx.y * head_dim + i] *= expf(old_max - max_score);
                }
            }
            float current_exp = expf(score - max_score);
            sum_exp += current_exp;
            for (int i = 0; i < head_dim; ++i) {
                acc_tile[threadIdx.y * head_dim + i] += current_exp * __half2float(v_tile[k_local_idx * head_dim + i]);
            }
        }
        __syncthreads();
    }

    // Write final output
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        out[q_idx * num_heads * head_dim + head_idx * head_dim + i] = __float2half(acc_tile[threadIdx.y * head_dim + i] / sum_exp);
    }
}


// --- HOST LAUNCHER: Paged Attention ---
torch::Tensor paged_attention(
    const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
    const torch::Tensor& block_table, const torch::Tensor& context_lens, float scale) {
    
    auto out = torch::empty_like(q);
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int num_kv_heads = k_cache.size(2);
    const int head_dim = q.size(2);
    const int block_size = k_cache.size(1);

    dim3 grid(batch_size, num_heads);
    dim3 block(BLOCK_THREADS);

    paged_attention_kernel<<<grid, block>>>(
        out.data_ptr<at::Half>(), q.data_ptr<at::Half>(), k_cache.data_ptr<at::Half>(), v_cache.data_ptr<at::Half>(),
        block_table.data_ptr<int>(), context_lens.data_ptr<int>(), scale,
        num_heads, num_kv_heads, head_dim, block_size);
    return out;
}

// --- HOST LAUNCHER: Variable Length Attention ---
torch::Tensor variable_length_attention(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& cu_seqlens, const torch::Tensor& block_table, int max_seqlen, float scale) {

    auto out = torch::empty_like(q);
    const int total_tokens = q.size(0);
    const int num_heads = q.size(1);
    const int num_kv_heads = k.size(1);
    const int head_dim = q.size(2);

    dim3 grid( (total_tokens + TILE_SIZE_Q - 1) / TILE_SIZE_Q, num_heads);
    dim3 block(BLOCK_THREADS, TILE_SIZE_Q);

    variable_length_attention_kernel<<<grid, block>>>(
        out.data_ptr<at::Half>(), q.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(),
        cu_seqlens.data_ptr<int>(), scale, total_tokens, num_heads, num_kv_heads, head_dim);
    return out;
}