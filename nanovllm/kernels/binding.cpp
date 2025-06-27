#include <torch/extension.h>
#include "attention_kernels.h"

namespace py = pybind11;

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("variable_length_attention", &variable_length_attention, "Variable-length (prefill) attention with PagedAttention support",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("cu_seqlens"),
          py::arg("scale"));
          
    m.def("paged_attention", &paged_attention, "Paged attention (decode) for single-token generation",
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_table"),
          py::arg("context_lens"),
          py::arg("scale"));
}