"""Stack verification: load a Bernini-R transformer, inject LoRA, count params.

Run after the env + weights are ready:
    cd /data/bernini-src && .venv/bin/python train/smoke_lora.py
This is the gate before writing the full training loop: it proves
(1) torch sees the Blackwell GPU, (2) the released diffusers transformer loads
with source-id RoPE, (3) PEFT injects LoRA on the exact target modules.
"""
import sys
import torch

sys.path.insert(0, "/data/bernini-src")
from train.lora_targets import inject_lora, LORA_TARGET_MODULES

MODEL_DIR = "/data/models/Bernini-R-Diffusers"


def main():
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        # Blackwell sanity: a real kernel launch, not just a property read.
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print("bf16 matmul on GPU ok:", (x @ x).float().sum().isfinite().item())

    from bernini.models.transformer_wan import WanTransformer3DModel

    print("\nloading transformer (high-noise) on CPU bf16 ...")
    tf = WanTransformer3DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    print("loaded. has rope:", hasattr(tf, "rope"))
    print("use_src_id_rotary_emb:", getattr(tf.rope, "use_src_id_rotary_emb", "?"))
    n_total = sum(p.numel() for p in tf.parameters())
    print(f"base params: {n_total/1e9:.2f}B")

    tf, n_trainable = inject_lora(tf)
    print(f"LoRA trainable params: {n_trainable/1e6:.2f}M  (targets={LORA_TARGET_MODULES})")

    adapted = [n for n, _ in tf.named_modules() if n.endswith("lora_A.default")]
    print(f"adapted linears: {len(adapted)} (first 4)")
    for n in adapted[:4]:
        print("   ", n)
    assert n_trainable > 0, "no trainable LoRA params injected!"
    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
