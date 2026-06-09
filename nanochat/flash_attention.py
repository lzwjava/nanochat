"""
Unified Flash Attention interface with automatic FA3/FA2/SDPA switching.

Priority order:
  1. Flash Attention 3 (NVIDIA Hopper SM90 only)
  2. Flash Attention 2 (NVIDIA CUDA + AMD ROCm via flash-attn package)
  3. PyTorch SDPA fallback (all platforms)

Exports `flash_attn` module that matches the FA3 API exactly.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 (Hopper), then FA2 (all GPUs including AMD)
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_2():
    """Try to load Flash Attention 2 (works on NVIDIA CUDA + AMD ROCm)."""
    if not torch.cuda.is_available():
        return None
    try:
        import flash_attn
        from flash_attn import flash_attn_func as fa2_func
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        # Test that it actually works (not just importable)
        # Return a simple namespace with the functions we need
        return flash_attn
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

_fa2 = _load_flash_attention_2() if not HAS_FA3 else None
HAS_FA2 = _fa2 is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_impl():
    """Decide which attention implementation to use."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'fa2':
        assert HAS_FA2, "Cannot override to FA2: flash-attn package not installed"
        return 'fa2'
    if _override_impl == 'sdpa':
        return 'sdpa'
    if HAS_FA3:
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
    if HAS_FA2:
        return 'fa2'
    return 'sdpa'

_active_impl = _resolve_impl()
USE_FA3 = _active_impl == 'fa3'
USE_FA2 = _active_impl == 'fa2'

if USE_FA3:
    print("✓ Using Flash Attention 3 (Hopper GPU)")
elif USE_FA2:
    _is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
    _backend = "AMD ROCm" if _is_amd else "NVIDIA CUDA"
    print(f"✓ Using Flash Attention 2 ({_backend})")
else:
    print("! Using PyTorch SDPA fallback (no flash-attn found)")


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# FA2 helper: convert window_size format
# =============================================================================
def _fa2_window_size(window_size):
    """Convert FA3 window_size format (-1,-1) to FA2 format (-1,-1) or (left, right)."""
    # FA2 uses the same (-1, -1) for unlimited
    return (window_size[0], window_size[1]) if window_size[0] >= 0 else (-1, -1)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    if USE_FA2:
        # FA2 API: flash_attn_func(q, k, v, ..., window_size=(left, right))
        # q, k, v: (B, T, H, D) - same as FA3
        # dropout_p=0.0 for training (no dropout in attention)
        return _fa2.flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=causal,
            window_size=_fa2_window_size(window_size),
        )

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    if USE_FA2:
        # FA2 has flash_attn_with_kvcache with similar API
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=_fa2_window_size(window_size),
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
