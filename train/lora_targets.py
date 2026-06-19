"""Shared LoRA targeting for the Bernini-R (Wan2.2-A14B) renderer.

The renderer is a dual-expert MoE: `transformer` (high-noise) and
`transformer_2` (low-noise), each a `WanTransformer3DModel`. We attach the same
LoRA config to both. Targets mirror our `video_ref_cross_attn` preset on LTX:
self-attention (attn1, where the in-context concat lives), cross-attention to
text (attn2), and the block FFN.
"""

# diffusers Attention -> to_q/to_k/to_v/to_out(ModuleList[Linear, Dropout])
# WanTransformerBlock.ffn = FeedForward(net=[GEGLU/GELU(proj=Linear), Dropout, Linear])
LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
]

# rank-128 / alpha-64 per our standing identity/head-swap convention.
DEFAULT_RANK = 128
DEFAULT_ALPHA = 64


def build_lora_config(rank: int = DEFAULT_RANK, alpha: int = DEFAULT_ALPHA, dropout: float = 0.0):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )


def inject_lora(transformer, rank: int = DEFAULT_RANK, alpha: int = DEFAULT_ALPHA, dropout: float = 0.0):
    """Inject LoRA in-place; freeze base, return (model, n_trainable)."""
    from peft import inject_adapter_in_model

    for p in transformer.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(build_lora_config(rank, alpha, dropout), transformer)
    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return transformer, n_trainable
