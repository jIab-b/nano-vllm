#include <torch/extension.h>
#include "attention_kernels.h"

// Macro to check that a tensor is a CUDA tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

// Macro to check that a tensor is a CUDA tensor and contiguous
#define CHECK_INPUT(x) CHECK_CUDA(x); TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Python binding for the variable-length attention (prefill)
torch::Tensor py_variable_length_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& block_table,
    int max_seqlen,
    float scale) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cu_seqlens);
    CHECK_INPUT(block_table);

    return variable_length_attention(q, k, v, cu_seqlens, block_table, max_seqlen, scale);
}

// Python binding for the paged attention (decode)
torch::Tensor py_paged_attention(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& context_lens,
    float scale) {

    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_table);
    CHECK_INPUT(context_lens);

    return paged_attention(q, k_cache, v_cache, block_table, context_lens, scale);
}


// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("variable_length_attention", &py_variable_length_attention, "Variable-length (prefill) attention with PagedAttention support");
    m.def("paged_attention", &py_paged_attention, "Paged attention (decode) for single-token generation");
}