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

    # Handle the case where the context length is 0, which occurs during CUDA graph capture
    if max_len == 0:
        return torch.zeros_like(q)
        
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


def pytorch_variable_length_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
):
    """
    A PyTorch-based implementation of variable-length attention for the prefill stage.
    This function handles both cases:
    1.  Standard prefill where sequences are passed as contiguous tensors.
    2.  Prefill with a shared prefix, where the KV cache is scattered.
    """
    total_tokens, _, head_dim = q.shape

    # Handle the case where there are no tokens, which occurs during CUDA graph capture
    if total_tokens == 0:
        return torch.zeros_like(q)
    
    # If block_tables is present, it means we have a scattered KV cache (prefix caching)
    # and we need to gather the keys and values into contiguous tensors.
    if block_tables is not None:
        # This logic is similar to paged_attention, but adapted for variable lengths
        max_len = max(cu_seqlens_q[i] - cu_seqlens_q[i-1] for i in range(1, len(cu_seqlens_q)))
        block_size = k_cache.size(1)
        
        # Create contiguous tensors for the entire batch
        k_contiguous = torch.zeros(total_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
        v_contiguous = torch.zeros(total_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

        # Iterate through each sequence in the batch
        for i in range(len(cu_seqlens_q) - 1):
            start_idx = cu_seqlens_q[i].item()
            end_idx = cu_seqlens_q[i+1].item()
            seq_len = end_idx - start_idx
            
            # Gather K/V for this specific sequence
            for j in range(seq_len):
                block_idx = block_tables[i, j // block_size].item()
                block_offset = j % block_size
                k_contiguous[start_idx + j] = k_cache[block_idx, block_offset]
                v_contiguous[start_idx + j] = v_cache[block_idx, block_offset]
        k, v = k_contiguous, v_contiguous

    # Create a causal mask from the cumulative sequence lengths
    # This is the most critical part for handling variable-length sequences correctly
    token_indices = torch.arange(total_tokens, device=q.device)
    
    # Find the sequence index for each token
    seq_ids = torch.zeros_like(token_indices)
    for i in range(len(cu_seqlens_q) - 1):
        start_idx = cu_seqlens_q[i]
        end_idx = cu_seqlens_q[i+1]
        seq_ids[start_idx:end_idx] = i
        
    # The mask is True where tokens can attend to each other:
    # 1. They must be in the same sequence (seq_ids[:, None] == seq_ids[None, :])
    # 2. The attention must be causal (token_indices[:, None] >= token_indices[None, :])
    attn_mask = (seq_ids[:, None] == seq_ids[None, :]) & (token_indices[:, None] >= token_indices[None, :])
    
    # Repeat K/V heads for Grouped-Query Attention (GQA)
    if num_q_heads != num_kv_heads:
        num_groups = num_q_heads // num_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    # Reshape for PyTorch's expected format [Batch, Heads, Seq, Dim]
    # For prefill, the batch dimension is the total number of tokens
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # Add batch and head dimensions

    # Use the built-in scaled_dot_product_attention
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
    
    return o.squeeze(0) # Remove the batch dimension