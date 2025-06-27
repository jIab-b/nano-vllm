#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "attention_kernels.h"
#include <ATen/ATen.h>

namespace cg = cooperative_groups;

// Define constants for tiling dimensions.
constexpr int TILE_SIZE_Q = 16;
constexpr int TILE_SIZE_K = 16;
constexpr int BLOCK_THREADS = 128;

namespace { // Use an anonymous namespace to limit symbol visibility and prevent linkage errors.

// --- KERNEL IMPLEMENTATION: Paged Attention (Decode) ---
template <typename T>
__global__ void paged_attention_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_len = context_lens[batch_idx];
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    extern __shared__ char smem[];
    T* q_tile = reinterpret_cast<T*>(smem);
    float* acc_tile = reinterpret_cast<float*>(smem + head_dim * sizeof(T));
    T* k_tile = reinterpret_cast<T*>(smem + head_dim * sizeof(T) + head_dim * sizeof(float));
    T* v_tile = reinterpret_cast<T*>(smem + head_dim * sizeof(T) + head_dim * sizeof(float) + TILE_SIZE_K * head_dim * sizeof(T));

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_tile[i] = q[batch_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    __syncthreads();

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        acc_tile[i] = 0.0f;
    }
    __syncthreads();

    for (int k_start = 0; k_start < seq_len; k_start += TILE_SIZE_K) {
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
            int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                int block_table_idx = batch_idx * max_blocks_per_seq + token_idx / block_size;
                int block_idx = block_table[block_table_idx];
                int block_offset = token_idx % block_size;
                k_tile[ (i/head_dim)*head_dim + dim_idx ] = k_cache[block_idx * block_size * num_kv_heads * head_dim + block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        __shared__ float scores[TILE_SIZE_K];
        for (int i = threadIdx.x; i < TILE_SIZE_K; i += blockDim.x) {
            if (k_start + i < seq_len) {
                float score = 0.0f;
                for (int j = 0; j < head_dim; ++j) {
                    score += static_cast<float>(q_tile[j]) * static_cast<float>(k_tile[i * head_dim + j]);
                }
                scores[i] = score * scale;
            }
        }
        __syncthreads();

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
       
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
             int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                int block_table_idx = batch_idx * max_blocks_per_seq + token_idx / block_size;
                int block_idx = block_table[block_table_idx];
                int block_offset = token_idx % block_size;
                v_tile[ (i/head_dim)*head_dim + dim_idx ] = v_cache[block_idx * block_size * num_kv_heads * head_dim + block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE_K; ++i) {
            if (k_start + i < seq_len) {
                float current_exp = expf(scores[i] - max_score);
                sum_exp += current_exp;
                for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
                    acc_tile[j] += current_exp * static_cast<float>(v_tile[i * head_dim + j]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        out[batch_idx * num_heads * head_dim + head_idx * head_dim + i] = static_cast<T>(acc_tile[i] / sum_exp);
    }
}

// --- KERNEL IMPLEMENTATION: Variable Length Attention (Prefill) ---
template <typename T>
__global__ void variable_length_attention_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const int* __restrict__ cu_seqlens,
    float scale,
    int total_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim) {

    const int q_idx_global = blockIdx.x * TILE_SIZE_Q + threadIdx.y;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int seq_idx = 0;
    while (q_idx_global >= cu_seqlens[seq_idx + 1]) {
        seq_idx++;
    }
    const int seq_start = cu_seqlens[seq_idx];
    const int seq_len = cu_seqlens[seq_idx + 1] - seq_start;
    const int q_idx_local = q_idx_global - seq_start;

    if (q_idx_global >= total_tokens) return;

    extern __shared__ char smem[];
    // Shared memory is scoped per thread block.
    // We process TILE_SIZE_Q queries per block, so we need space for all of them.
    T* q_tile = reinterpret_cast<T*>(smem);
    float* acc_tile = reinterpret_cast<float*>(smem + TILE_SIZE_Q * head_dim * sizeof(T));
    T* k_tile = reinterpret_cast<T*>(smem + TILE_SIZE_Q * head_dim * sizeof(T) + TILE_SIZE_Q * head_dim * sizeof(float));
    T* v_tile = reinterpret_cast<T*>(k_tile + TILE_SIZE_K * head_dim);

    // Each row of threads (threadIdx.y) handles one query.
    // Load Q for the current query into the correct slice of shared memory.
    T* current_q = &q_tile[threadIdx.y * head_dim];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        current_q[i] = q[q_idx_global * num_heads * head_dim + head_idx * head_dim + i];
    }
    
    float* current_acc = &acc_tile[threadIdx.y * head_dim];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        current_acc[i] = 0.0f;
    }
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    __syncthreads();

    for (int k_start = 0; k_start < seq_len; k_start += TILE_SIZE_K) {
        for (int i = threadIdx.x; i < TILE_SIZE_K * head_dim; i += blockDim.x) {
            int token_idx = k_start + (i / head_dim);
            int dim_idx = i % head_dim;
            if (token_idx < seq_len) {
                k_tile[(i/head_dim)*head_dim + dim_idx] = k[(seq_start + token_idx) * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
                v_tile[(i/head_dim)*head_dim + dim_idx] = v[(seq_start + token_idx) * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
            }
        }
        __syncthreads();

        for (int k_local_idx = 0; k_local_idx < TILE_SIZE_K; ++k_local_idx) {
            int k_global_idx = k_start + k_local_idx;
            if (k_global_idx > q_idx_local) continue;

            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score += static_cast<float>(current_q[i]) * static_cast<float>(k_tile[k_local_idx * head_dim + i]);
            }
            score *= scale;

            if (score > max_score) {
                float old_max = max_score;
                max_score = score;
                sum_exp *= expf(old_max - max_score);
                for (int i = 0; i < head_dim; ++i) {
                    current_acc[i] *= expf(old_max - max_score);
                }
            }
            float current_exp = expf(score - max_score);
            sum_exp += current_exp;
            for (int i = 0; i < head_dim; ++i) {
                current_acc[i] += current_exp * static_cast<float>(v_tile[k_local_idx * head_dim + i]);
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        out[q_idx_global * num_heads * head_dim + head_idx * head_dim + i] = static_cast<T>(current_acc[i] / sum_exp);
    }
}

// --- HOST LAUNCHER: Paged Attention ---
template <typename T>
torch::Tensor paged_attention_template(
    const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
    const torch::Tensor& block_table, const torch::Tensor& context_lens, float scale) {
    
    auto out = torch::empty_like(q, q.options().dtype(q.scalar_type()));
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int num_kv_heads = k_cache.size(2);
    const int head_dim = q.size(2);
    const int block_size = k_cache.size(1);
    const int max_blocks_per_seq = block_table.size(1);

    dim3 grid(batch_size, num_heads);
    dim3 block(BLOCK_THREADS);
    
    size_t shared_mem_size = head_dim * sizeof(T) +         // q_tile
                             head_dim * sizeof(float) +       // acc_tile
                             TILE_SIZE_K * head_dim * sizeof(T) + // k_tile
                             TILE_SIZE_K * head_dim * sizeof(T);  // v_tile

    paged_attention_kernel<T><<<grid, block, shared_mem_size>>>(
        out.data_ptr<T>(), q.data_ptr<T>(), k_cache.data_ptr<T>(), v_cache.data_ptr<T>(),
        block_table.data_ptr<int>(), context_lens.data_ptr<int>(), scale,
        num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq);
    return out;
}

// --- HOST LAUNCHER: Variable Length Attention ---
template <typename T>
torch::Tensor variable_length_attention_template(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& cu_seqlens, float scale) {

    auto out = torch::empty_like(q, q.options().dtype(q.scalar_type()));
    const int total_tokens = q.size(0);
    const int num_heads = q.size(1);
    const int num_kv_heads = k.size(1);
    const int head_dim = q.size(2);

    dim3 grid( (total_tokens + TILE_SIZE_Q - 1) / TILE_SIZE_Q, num_heads);
    // Use a 2D block where threadIdx.y maps to the query in the tile
    constexpr int BLOCK_DIM_X = 32;
    dim3 block(BLOCK_DIM_X, TILE_SIZE_Q);

    size_t shared_mem_size = (TILE_SIZE_Q * head_dim * sizeof(T)) + // q_tile
                             (TILE_SIZE_Q * head_dim * sizeof(float)) + // acc_tile
                             (TILE_SIZE_K * head_dim * sizeof(T)) + // k_tile
                             (TILE_SIZE_K * head_dim * sizeof(T));  // v_tile

    variable_length_attention_kernel<T><<<grid, block, shared_mem_size>>>(
        out.data_ptr<T>(), q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
        cu_seqlens.data_ptr<int>(), scale, total_tokens, num_heads, num_kv_heads, head_dim);
    return out;
}

} // anonymous namespace

// Dispatcher functions that will be bound to Python
torch::Tensor paged_attention(
    const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
    const torch::Tensor& block_table, const torch::Tensor& context_lens, float scale) {
    switch (q.scalar_type()) {
        case at::ScalarType::Half:
            return paged_attention_template<at::Half>(q, k_cache, v_cache, block_table, context_lens, scale);
        case at::ScalarType::Float:
            return paged_attention_template<float>(q, k_cache, v_cache, block_table, context_lens, scale);
        default:
            TORCH_CHECK(false, "Unsupported tensor type for paged_attention");
    }
}

torch::Tensor variable_length_attention(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& cu_seqlens, float scale) {
    switch (q.scalar_type()) {
        case at::ScalarType::Half:
            return variable_length_attention_template<at::Half>(q, k, v, cu_seqlens, scale);
        case at::ScalarType::Float:
            return variable_length_attention_template<float>(q, k, v, cu_seqlens, scale);
        default:
            TORCH_CHECK(false, "Unsupported tensor type for variable_length_attention");
    }
}