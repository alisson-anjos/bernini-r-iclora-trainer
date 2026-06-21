"""IC-LoRA Reference Conditioning (Bernini-R) — generic ComfyUI inference node.

Task-agnostic: attach a guide video and any number of reference images to the
conditioning as Bernini in-context latents, each with its source_id RoPE phase:

    guide video  -> source_id 1   (the v2v base / kept content)
    reference[i] -> source_id 2,3,...   (what you inject — identity, object, style)
    denoised output -> source_id 0   (the empty latent)

Sizes follow the inputs (snapped to /16). This is the *inference* counterpart of
the trainer — pair it with the matching IC-LoRA on a Bernini-R high/low base.

Apache-2.0 (this file is our own code). It only SETS `context_latents` on the
conditioning; something must CONSUME them during sampling — the Bernini WanModel
patch. Provide it via **ComfyUI-RH-Bernini** (GPL-3.0, install separately) or a
ComfyUI build with native Bernini support. We do NOT bundle that patch here to
keep this repo Apache-licensed.

Drop this file into a ComfyUI custom-nodes package and merge NODE_CLASS_MAPPINGS.
"""
import logging

import torch

import comfy.model_management
import comfy.utils
import node_helpers

log = logging.getLogger("ICLoRA.ReferenceConditioning")
STRIDE = 16


def _snap(v, stride=STRIDE):
    return max(stride, round(v / stride) * stride)


def _snap_frames(n):
    return max(1, ((max(1, int(n)) - 1) // 4) * 4 + 1)


def _encode_native(vae, frames):
    h, w = frames.shape[1], frames.shape[2]
    nh, nw = _snap(h), _snap(w)
    if (nh, nw) != (h, w):
        frames = comfy.utils.common_upscale(
            frames[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
        ).movedim(1, -1)
    return vae.encode(frames[:, :, :, :3]), nh, nw


class ICLoRAReferenceConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "length": ("INT", {"default": 73, "min": 1, "max": 1000, "step": 4,
                                   "tooltip": "Frame count (snapped to 4k+1)."}),
                "amplify_reference": ("BOOLEAN", {"default": True, "tooltip":
                    "ON: references go on positive only (guide stays on both) so CFG amplifies "
                    "them. Use CFG ~3-5, no step-distill/Lightning LoRA (CFG 1.0 = no amplification)."}),
            },
            "optional": {
                "guide_video": ("IMAGE", {"tooltip": "v2v base / kept content -> source_id 1. Output size = this."}),
                "reference_images": ("IMAGE", {"tooltip": "One or more references -> source_id 2,3,... (batched IMAGE)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "debug")
    FUNCTION = "execute"
    CATEGORY = "IC-LoRA/video"
    DESCRIPTION = (
        "Generic Bernini-R in-context conditioning: guide video (source_id 1) + reference "
        "image(s) (source_id 2,3,...). Pair with the matching IC-LoRA on a Bernini-R base.\n\n"
        "⚠️ Step-distill / Lightning LoRAs force CFG ~1.0, which disables CFG guidance — "
        "references won't be amplified. Use CFG ~3-5 without them.\n\n"
        "Requires the Bernini WanModel patch (ComfyUI-RH-Bernini or native ComfyUI support)."
    )

    def execute(self, positive, negative, vae, length, amplify_reference=True,
                guide_video=None, reference_images=None):
        length = _snap_frames(length)
        ctx_pos, refs_only, parts = [], [], []
        gh = gw = None

        if guide_video is not None:
            gl, gh, gw = _encode_native(vae, guide_video[:length])
            ctx_pos.append(gl)
            parts.append(f"guide(sid1) {gw}x{gh} {tuple(gl.shape)}")

        if reference_images is not None:
            for i in range(reference_images.shape[0]):
                rl, rh, rw = _encode_native(vae, reference_images[i:i + 1])
                ctx_pos.append(rl)
                refs_only.append(rl)
                parts.append(f"ref(sid{len(ctx_pos)}) {rw}x{rh} {tuple(rl.shape)}")

        if not ctx_pos:
            raise ValueError("ICLoRAReferenceConditioning: provide guide_video and/or reference_images.")

        # output size follows the guide if present, else the first reference
        if gh is None:
            _, _, _, lh, lw = ctx_pos[0].shape
            gh, gw = lh * 8, lw * 8

        # amplify: references only on positive; guide (if any) on both. No guide ->
        # negative carries nothing so CFG amplifies all references.
        if amplify_reference:
            ctx_neg = [ctx_pos[0]] if guide_video is not None else []
        else:
            ctx_neg = list(ctx_pos)

        positive = node_helpers.conditioning_set_values(positive, {"context_latents": ctx_pos})
        negative = node_helpers.conditioning_set_values(negative, {"context_latents": ctx_neg})

        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, gh // 8, gw // 8],
            device=comfy.model_management.intermediate_device(),
        )
        dbg = ("=== IC-LoRA Reference Conditioning ===\n"
               f"OUTPUT (sid0): {gw}x{gh}, {length}f -> {tuple(latent.shape)}\n"
               + "\n".join("  " + p for p in parts)
               + f"\namplify_reference={amplify_reference} (pos ctx={len(ctx_pos)}, neg ctx={len(ctx_neg)})"
               + ("\n-> CFG ~3-5, no Lightning/step-distill." if amplify_reference else ""))
        log.info("\n" + dbg)
        return (positive, negative, {"samples": latent}, dbg)


class ICLoRALoRADebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "debug")
    FUNCTION = "execute"
    CATEGORY = "IC-LoRA/video"
    DESCRIPTION = "Reports how many LoRA/patch entries are applied to the MODEL. 0 = LoRA not loaded / keys didn't match."

    def execute(self, model):
        patches = getattr(model, "patches", {}) or {}
        n_keys = len(patches)
        block_keys = [k for k in patches if "blocks" in k]
        sample = list(patches.keys())[:10]
        dbg = ("=== IC-LoRA LoRA Debug ===\n"
               f"patched weight keys: {n_keys} (transformer blocks: {len(block_keys)})\n"
               + ("!! ZERO -> LoRA not loaded or keys didn't match.\n" if n_keys == 0 else "")
               + "sample:\n" + "\n".join("  " + k for k in sample))
        log.info("\n" + dbg)
        return (model, dbg)


NODE_CLASS_MAPPINGS = {
    "ICLoRAReferenceConditioning": ICLoRAReferenceConditioning,
    "ICLoRALoRADebug": ICLoRALoRADebug,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ICLoRAReferenceConditioning": "IC-LoRA Reference Conditioning (Bernini-R)",
    "ICLoRALoRADebug": "IC-LoRA LoRA Debug",
}
