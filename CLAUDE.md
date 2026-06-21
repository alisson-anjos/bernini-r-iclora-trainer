# CLAUDE.md — working notes & customization guide

Guidance for Claude Code (and humans) extending this repo. It's a **reference-guided
IC-LoRA trainer for Bernini-R** (Wan2.2-T2V-A14B). Head-swap is the example; the same
code trains any `(guide video, reference image) → target video` edit. Read this before
changing things — it captures what we learned the hard way.

## Mental model (how the task works)
Bernini-R is a **dual-expert MoE**: `diff_dec.transformer` (high-noise) and
`diff_dec.transformer_2` (low-noise). It renders **in-context**: every conditioning
source is a *clean* VAE latent concatenated into ONE self-attention sequence with the
noisy target, disambiguated only by a **per-source RoPE phase** (`source_id`), computed
on the fly (no learned weights):

```
[ guide video (source_id 1) | reference image (source_id 2) | noisy target (source_id 0) ]
```
We **freeze the base** and train **LoRA on attn1 (self) / attn2 (cross) / ffn of both
experts**. Objective: **flow-matching v-target** (`noise - clean`), loss on the **target
tokens only**. The train forward mirrors the VI combo of `GEN_Wanx22.sample()` minus the
guidance — so what you train is what inference runs.

## Code map (`train/`)
- `train_iclora.py` — everything. `Renderer` (load + LoRA inject + **single-expert
  offload** + `train_loss`), `main()` loop (picks an expert per grad-accum group →
  `ensure_expert` → step that expert's optimizer), `save_lora`. Loss is built in packed
  token space; no unpatchify.
- `validate.py` — rv2v generation via the pipeline + **ArcFace** identity metric.
  References are tight head crops, so `_embed` pads before face detection (RetinaFace
  misses borderless crops).
- `export_lora_comfy.py` — diffusers/PEFT keys → **ComfyUI kohya keys**, split into
  high/low files.
- `arcface_monitor.py` — out-of-process ArcFace logger (the in-trainer insightface init
  was flaky; compute from saved val mp4s instead).
- `lora_targets.py`, `smoke_lora.py`, `infer_headswap.py`, `run_headswap_iclora.sh`.

## Gotchas we hit (don't relearn these)
1. **Blackwell needs torch cu128.** The pinned `torch 2.5.1+cu124` predates Blackwell →
   "no kernel image". flash-attn is optional — skip it (SDPA / VeOmni attention).
2. **Two Bernini repo patches are required** (`patches/`): (01) make the Qwen planner
   import optional so the renderer path loads without flash-attn; (02) add gradient
   checkpointing in the transformer block loop (needed to fit memory).
3. **Load the renderer** with `BerniniRendererPipeline.from_pretrained(dir,
   load_ckpt_weights=False, use_src_id_rotary_emb=True)` for the diffusers layout — else
   source-id RoPE is off and weights don't load.
4. **Single-expert offload** is the default: only the routed expert sits on GPU (the
   other + its optimizer park on CPU), expert chosen per grad-accum group to minimize
   swaps. This is what lets high frame counts fit one GPU. `--both_experts` to disable.
   Measured peaks: 121f/256 ≈ 45 GB, 73f/640 ≈ 71 GB.
5. **VRAM at validation** spikes (it generates at train res). 640 long edge rides near a
   96 GB cap and can OOM mid-run — use ~576 or lower the validation res for margin.
6. **Wan VAE** — latents are encoded on the fly; LTX/other precomputed latents are NOT
   reusable.
7. **ArcFace is the real signal**; FM loss is ~flat. Pick the best checkpoint by ArcFace.
8. **Train near your inference target.** A 21f/320 LoRA lost consistency when inferred at
   higher frames/resolution. Retrained at 73f/640. Match training to how you'll run it.
9. **ComfyUI LoRA keys must be single underscore** `lora_unet_blocks_N_self_attn_q...`.
   The double-underscore kohya variant loads NOTHING in ComfyUI native. `export_lora_comfy.py`
   already emits the correct form.
10. **In ComfyUI the base must be Bernini-R** (e.g. `Comfy-Org/Bernini-R` hi/lo), NOT
    vanilla Wan2.2 — the LoRA is a delta on Bernini's in-context fine-tune; vanilla lacks
    the capability and the result is weak.
11. **ComfyUI guidance:** attaching the reference to both positive & negative makes CFG
    cancel it. Put the reference on **positive only** (guide on both) so CFG amplifies
    identity. LightX2V / step-distill force CFG ~1.0 → no amplification → weak swap. Use
    CFG ~3-5 without LightX2V.

## Customizing for YOUR task
The code is task-agnostic given a `(guide video, reference image, target video)` triplet:
1. **Dataset** — build `target/`, `guide/`, `reference/` folders with matching `<vid>`
   names + a JSONL of `{"vid", "caption"}` (see README). `target` is the *edited result*
   you want the model to produce — that's the hard part to source; we synthesized ours.
2. **Trigger / captions** — the trigger word (`head_swap:`) is just convention; use any
   token. Most of our captions were *only* the trigger — identity came from the reference
   + the in-context mechanism, not the text. Rich captions are optional.
3. **More than one reference** — `train_loss` currently assigns guide=1, single ref=2.
   For multiple references, append more clean latents with source_id 3,4,... (mirror the
   `_make_sids` logic in `GEN_Wanx22.sample()` and `packing_vae`).
4. **Knobs** — `--rank/--alpha` (we used 64/64; 128 overfit faster), `--num_frames`,
   `--max_size` (trade against VRAM), `--lr`, `--max_steps`. Watch ArcFace (or your own
   metric) and stop at the plateau (ours ~0.57 at step 2500-3000).
5. **Export** — `export_lora_comfy.py --ckpt <step>.safetensors --alpha <alpha>` → hi/lo
   ComfyUI files. In ComfyUI: Bernini-R base + the matching hi/lo LoRA + CFG 3-5.

## Running the trained LoRA in ComfyUI (you need a node)
The trainer only produces a LoRA. To run it you need a ComfyUI **conditioning node**
that attaches the guide + references as in-context `context_latents` with `source_id` —
stock ComfyUI nodes don't do this. Two parts:
1. **The node (this repo, Apache):** a generic one is included —
   `comfyui/iclora_reference_conditioning.py` (`ICLoRAReferenceConditioning`): guide →
   source_id 1, reference images → source_id 2,3…, plus an `amplify_reference` toggle
   (references on positive only so CFG amplifies them) and a LoRA-debug node. It is
   **task-agnostic** — don't hard-code head-swap here; this repo trains any reference edit.
2. **The patch (NOT in this repo):** something must *consume* `context_latents` during
   sampling — the Bernini WanModel patch. It comes from **ComfyUI-RH-Bernini** (GPL-3.0)
   or a ComfyUI build with native Bernini support. We deliberately don't vendor it here
   because it's GPL and this repo is Apache. (If you maintain a GPL node pack, you *can*
   vendor it there.)

Inference rules: base = **Bernini-R hi/lo** (not vanilla Wan2.2), high LoRA → high model,
low LoRA → low model, **CFG ~3–5**, `amplify_reference` ON, no Lightning/step-distill
LoRA (CFG 1.0 kills the guidance). Verify the LoRA actually loaded (LoRA-debug node:
patched keys > 0; double-underscore keys = loads nothing → re-export).

## When extending with Claude Code
- The trainer is one file (`train_iclora.py`) — `train_loss` is the forward; the `main`
  loop owns the expert offload/optimizer dance. Keep the per-expert optimizer + device
  bookkeeping intact when you touch the loop.
- Validate any change with `python train/train_iclora.py --smoke ...` (3 steps, tiny) and
  watch peak VRAM before launching a long run.
- If results look wrong at inference, check (in order): base is Bernini-R, LoRA keys load
  (HeadSwapLoRADebug or count patches), CFG not stuck at 1.0, reference is a head crop,
  resolution/frames near training.
