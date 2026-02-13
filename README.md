# HyperMLP

This repository hosts the implementation of the paper `HyperMLP: An Integrated Perspective for Sequence Modeling`.

We are currently developing a more efficient subsequent architecture for the model. At this stage, we release a cleaned, naive PyTorch implementation that has been refined and structured with the assistance of LLM-based tooling.

We will continue to improve the efficiency and usability of this repository over time.

## Quick Start

```python
import torch
from hyperglu import HyperGLU, HyperGLUConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32

B, T, D, H = 2, 128, 512, 8
x = torch.randn(B, T, D, device=device, dtype=dtype)

cfg = HyperGLUConfig(
    # -------------------------
    # Core model dimensions
    # -------------------------
    n_embd=D,                # hidden size (model width)
    n_head=H,                # number of heads
    block_size=2048,         # max supported sequence length (T <= block_size)
    bias=False,              # whether linear layers use bias

    # -------------------------
    # Attention family
    # -------------------------
    attn_kind="glu",         # "glu" | "relu"
                             # "glu": split heads into 2H and apply softplus(x2)*relu(x1)
                             # "relu": relu(norm(qk)) (HyperMLP-style)

    attn_norm="out_scale",   # "out_scale" | "l2" | "none"
                         # "l2": apply 1/||x|| on the pre-ReLU tensor (normalize then ReLU)
                         # "out_scale": compute the same inv_norm = 1/||x||, but apply it later on the output
                         # In this implementation these two are mathematically equivalent because:
                         #   ReLU(a*x) = a*ReLU(x) for a>=0, and all ops between base and output are linear.
                         # Differences (if any) are only numerical (eps/clamp/rounding).

    # -------------------------
    # Projection sizes
    # -------------------------
    qk_rank=None,            # total Q/K width before head split
                             # None -> defaults to n_embd // 4
                             # must be divisible by:
                             #   2*n_head if attn_kind="glu"
                             #   n_head   if attn_kind="relu"

    vo_rank=None,            # total V/O width
                             # None -> defaults to n_embd
                             # must be divisible by n_head

    # -------------------------
    # Causality & layout
    # -------------------------
    do_masking=True,         # whether to apply causal masking
    aligner_kind="offset",   # "offset" | "identity"
                             # "offset": skewed causal layout (efficient, default)
                             # "identity": left / tril-style layout

    # -------------------------
    # KV causal convolution
    # -------------------------
    use_kv_conv=True,        # apply depthwise causal conv before K/V
    conv_kernel_size=4,      # kernel size for causal conv

    # -------------------------
    # Rotary position embedding
    # -------------------------
    use_rope=False,          # apply RoPE to Q/K
    rope_base=10000.0,       # RoPE theta (only used if use_rope=True)

    # -------------------------
    # Gates (multiplicative)
    # -------------------------
    gate_act="sigmoid",      # "sigmoid" | "none"
    use_qk_gate=True,        # gate applied to Q (dtqk)
    use_vo_gate=True,        # gate applied to output (dtvo)

    # -------------------------
    # Low-rank residual mixing (op1 / op2)
    # -------------------------
    use_op1=True,            # pre-nonlinearity mixing
    op1_rank=16,             # rank of op1 low-rank factor
    op1_per_head=True,       # must be True for fused fast path
    op1_use_diag=True,       # include diagonal term
    op1_shortcut=True,       # include residual connection
    op1_do_masking=True,     # op1 applies causal masking (effective only if do_masking=True)

    use_op2=True,            # post-nonlinearity mixing
    op2_rank=16,             # rank of op2
    op2_per_head=True,
    op2_use_diag=True,
    op2_shortcut=True,
    op2_do_masking=True,

    # -------------------------
    # Implementation selection
    # -------------------------
    impl_default="chunked",  # "dense" | "chunked"
    chunk_T=1024,            # chunk size for chunked forward

    # -------------------------
    # Numeric stability
    # -------------------------
    norm_eps=1e-12,          # epsilon for normalization
)

layer = HyperGLU(cfg).to(device=device, dtype=dtype).eval()
with torch.no_grad():
    y = layer(x)             # (B, T, D)

print(y.shape)
```
