import torch
import torch.nn.functional as F

def pytorch_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
):
    """
    A slow but robust PyTorch-based implementation of paged attention.
    This function gathers the paged KV cache into a contiguous tensor before
    calling the native scaled_dot_product_attention.
    """
    batch_size, num_q_heads, head_dim = q.shape
    max_len = context_lens.max().item()
    block_size = k_cache.size(1)

    # Create contiguous key/value tensors for PyTorch's SDPA
    k_contiguous = torch.zeros(batch_size, max_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
    v_contiguous = torch.zeros(batch_size, max_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

    # Slow gather operation
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        for j in range(seq_len):
            block_idx = block_table[i, j // block_size].item()
            block_offset = j % block_size
            k_contiguous[i, j] = k_cache[block_idx, block_offset]
            v_contiguous[i, j] = v_cache[block_idx, block_offset]

    # Repeat K/V heads for Grouped-Query Attention (GQA)
    if num_q_heads != num_kv_heads:
        num_groups = num_q_heads // num_kv_heads
        k_contiguous = k_contiguous.repeat_interleave(num_groups, dim=2)
        v_contiguous = v_contiguous.repeat_interleave(num_groups, dim=2)

    # PyTorch's SDPA needs a causal mask
    # Create a mask that is True for positions we want to attend to.
    attn_mask = torch.arange(max_len, device=q.device)[None, :] < context_lens[:, None]
    
    # Reshape for PyTorch's expected format [Batch, Heads, Seq, Dim]
    q = q.unsqueeze(2) # Add sequence dimension for the single query token

    # Use the built-in scaled_dot_product_attention
    # The mask ensures we only attend to valid tokens for each sequence in the batch.
    o = F.scaled_dot_product_attention(q, k_contiguous, v_contiguous, attn_mask=attn_mask.unsqueeze(1).unsqueeze(2), scale=scale)
    
    return o.squeeze(2) # Remove the sequence dimension