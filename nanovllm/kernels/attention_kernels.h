#pragma once

#include <torch/extension.h>

// Forward declaration for the custom variable-length attention kernel (prefill phase)
torch::Tensor variable_length_attention(
    const torch::Tensor& q,         // Query tensor
    const torch::Tensor& k,         // Key tensor
    const torch::Tensor& v,         // Value tensor
    const torch::Tensor& cu_seqlens, // Cumulative sequence lengths
    const torch::Tensor& block_table, // Block table for prefix caching
    int max_seqlen,                 // Maximum sequence length in the batch
    float scale                     // Softmax scale factor
);

// Forward declaration for the custom paged attention kernel (decode phase)
torch::Tensor paged_attention(
    const torch::Tensor& q,           // Query tensor
    const torch::Tensor& k_cache,     // Key cache (paged)
    const torch::Tensor& v_cache,     // Value cache (paged)
    const torch::Tensor& block_table, // Block table mapping sequences to blocks
    const torch::Tensor& context_lens, // Actual lengths of sequences in the batch
    float scale                       // Softmax scale factor
);