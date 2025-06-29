from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

_ATTENTION_BACKEND = None

def set_attention_backend(backend: str):
    """
    Explicitly sets the attention backend.
    Used for debugging and benchmarking.
    """
    global _ATTENTION_BACKEND
    if backend not in ["pytorch", "custom"]:
        raise ValueError(f"Unsupported attention backend: {backend}")
    _ATTENTION_BACKEND = backend

def get_attention_backend():
    """
    Detects the fastest available attention backend.
    Prioritizes custom CUDA kernels, then falls back to PyTorch.
    Caches the result to avoid repeated checks.
    """
    global _ATTENTION_BACKEND
    if _ATTENTION_BACKEND is not None:
        return _ATTENTION_BACKEND

    # 1. Try to import the custom kernels
    try:
        import custom_attention_kernels
        _ATTENTION_BACKEND = "custom"
        print("INFO: Using custom attention backend.")
        return _ATTENTION_BACKEND
    except ImportError:
        pass

    # 2. Fallback to PyTorch
    _ATTENTION_BACKEND = "pytorch"
    print("WARNING: Custom attention kernels not found. Falling back to slower PyTorch backend.")
    return _ATTENTION_BACKEND
