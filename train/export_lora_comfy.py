"""Split a Bernini-R IC-LoRA checkpoint into ComfyUI high/low-noise LoRA files.

Training saves one file with both experts in diffusers/PEFT naming:
    transformer.<...>.lora_{A,B}.default.weight      (high-noise)
    transformer_2.<...>.lora_{A,B}.default.weight    (low-noise)

ComfyUI's Wan loader expects the **kohya** convention (verified against the
working Wan2.2-Fun-A14B LoRAs in /data/comfyui/models/loras):
    lora_unet__blocks_<N>_<self_attn|cross_attn>_<q|k|v|o>.lora_down.weight
    lora_unet__blocks_<N>_ffn_<0|2>.lora_down.weight
    ... .lora_up.weight  + a scalar .alpha

We emit two files (one per expert):
    <out>_high_noise.safetensors  /  <out>_low_noise.safetensors

    .venv/bin/python train/export_lora_comfy.py --ckpt <lora_step.safetensors>
"""
import argparse
import os

import torch
from safetensors.torch import load_file, save_file

# diffusers module path -> kohya token
_MODULE_MAP = {
    "attn1.to_q": "self_attn_q",
    "attn1.to_k": "self_attn_k",
    "attn1.to_v": "self_attn_v",
    "attn1.to_out.0": "self_attn_o",
    "attn2.to_q": "cross_attn_q",
    "attn2.to_k": "cross_attn_k",
    "attn2.to_v": "cross_attn_v",
    "attn2.to_out.0": "cross_attn_o",
    "ffn.net.0.proj": "ffn_0",
    "ffn.net.2": "ffn_2",
}


def to_kohya(sd, src_prefix, alpha):
    out = {}
    for k, v in sd.items():
        if not k.startswith(src_prefix):
            continue
        rest = k[len(src_prefix):]  # blocks.N.<module>.lora_{A,B}.default.weight
        if ".lora_A.default.weight" in rest:
            mod, suf = rest[: -len(".lora_A.default.weight")], "lora_down.weight"
        elif ".lora_B.default.weight" in rest:
            mod, suf = rest[: -len(".lora_B.default.weight")], "lora_up.weight"
        else:
            continue
        # mod = blocks.N.<diffusers module>
        parts = mod.split(".")
        block = f"blocks_{parts[1]}"
        dmod = ".".join(parts[2:])
        if dmod not in _MODULE_MAP:
            raise KeyError(f"unmapped module: {dmod} (from {k})")
        # ComfyUI's native Wan key_map is `lora_unet_<key.replace('.','_')>`
        # (single underscore). The double-underscore kohya variant some tools
        # emit does NOT match ComfyUI native and silently loads nothing.
        base = f"lora_unet_{block}_{_MODULE_MAP[dmod]}"
        out[f"{base}.{suf}"] = v
        out[f"{base}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefix", default=None, help="default: alongside ckpt")
    ap.add_argument("--alpha", type=float, default=64.0, help="lora_alpha used in training")
    args = ap.parse_args()

    sd = load_file(args.ckpt)
    base = args.out_prefix or os.path.splitext(args.ckpt)[0]

    hi = to_kohya(sd, "transformer.", args.alpha)
    lo = to_kohya(sd, "transformer_2.", args.alpha)
    assert hi and lo, f"missing expert keys (hi={len(hi)} lo={len(lo)})"

    hi_path = f"{base}_high_noise.safetensors"
    lo_path = f"{base}_low_noise.safetensors"
    save_file(hi, hi_path)
    save_file(lo, lo_path)
    print(f"[export] high-noise: {len(hi)} tensors -> {hi_path}")
    print(f"[export] low-noise : {len(lo)} tensors -> {lo_path}")
    for k in list(hi)[:4]:
        print("  ", k)


if __name__ == "__main__":
    main()
