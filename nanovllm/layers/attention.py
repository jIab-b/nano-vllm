import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context, get_attention_backend
from nanovllm.layers.pytorch_attention import pytorch_paged_attention, pytorch_variable_length_attention

# Conditionally import custom kernels
try:
    from custom_attention_kernels import variable_length_attention, paged_attention
    _CUSTOM_KERNELS_AVAILABLE = True
except ImportError:
    variable_length_attention = None
    paged_attention = None
    _CUSTOM_KERNELS_AVAILABLE = False


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.backend = get_attention_backend()
        if self.backend == "custom" and not _CUSTOM_KERNELS_AVAILABLE:
            raise ImportError("Custom attention kernels requested but not found. Please build them first.")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # Prefill stage
            if self.backend == "custom":
                if context.block_tables is not None:  # prefix cache
                    k, v = k_cache, v_cache
                o = variable_length_attention(q, k, v,
                                              cu_seqlens=context.cu_seqlens_q,
                                              scale=self.scale)
            else: # "pytorch" backend
                o = pytorch_variable_length_attention(
                    q, k, v, k_cache, v_cache,
                    cu_seqlens_q=context.cu_seqlens_q,
                    block_tables=context.block_tables,
                    scale=self.scale,
                    num_q_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                )
        else:  # decode
            if self.backend == "custom":
                o = paged_attention(q, k_cache, v_cache,
                                    block_table=context.block_tables,
                                    context_lens=context.context_lens,
                                    scale=self.scale)
            else: # "pytorch" backend
                o = pytorch_paged_attention(q, k_cache, v_cache,
                                            block_table=context.block_tables,
                                            context_lens=context.context_lens,
                                            scale=self.scale,
                                            num_kv_heads=self.num_kv_heads)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
