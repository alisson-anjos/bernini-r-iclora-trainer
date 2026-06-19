# Bernini-R Head-Swap IC-LoRA Trainer

Train a **head-swap LoRA** on top of the [Bernini-R](https://huggingface.co/ByteDance/Bernini-R-Diffusers)
renderer (Wan2.2-T2V-A14B). Bernini natively concatenates conditioning sources into
one self-attention sequence disambiguated by a per-source RoPE phase (`source_id`),
so this is a *fine-tune, not teach* setup: freeze the base, attach LoRA, and learn
the head-swap from triplets.

```
[ source/guide video (source_id 1) | head reference image (source_id 2) | noisy target (source_id 0) ]
```
Flow-matching v-target, loss on the target tokens only. What you train is exactly
what inference runs (the VI combo of `GEN_Wanx22.sample()` minus guidance).

## What's here
- `train/train_iclora.py` — the trainer (single-expert GPU offload so high frame
  counts fit one GPU; per-expert LoRA + optimizer; flow-matching loss).
- `train/validate.py` — rv2v validation generation + ArcFace identity metric.
- `train/export_lora_comfy.py` — split a checkpoint into ComfyUI high/low-noise
  LoRA files (kohya `lora_unet_blocks_N_...` keys, single underscore).
- `train/arcface_monitor.py` — out-of-process ArcFace curve logger (wandb).
- `train/lora_targets.py`, `train/smoke_lora.py`, `train/run_headswap_iclora.sh`.
- `patches/` — two small edits to the Bernini repo (see Setup step 3).

## Requirements
- 1 GPU with ≥ ~72 GB VRAM for 73f/640 (single-expert offload). Less frames/res → less.
- Python **3.11** (VeOmni needs it).
- A Blackwell GPU needs **torch cu128** (the pinned `torch 2.5.1+cu124` predates Blackwell).

## Setup
**1. Clone Bernini and this repo**
```bash
git clone https://github.com/bytedance/Bernini.git
git clone <this-repo> bernini-headswap-trainer
cp -r bernini-headswap-trainer/train Bernini/train
cd Bernini
```

**2. Environment (Python 3.11)**
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
# Blackwell: cu128. Otherwise use the build matching your GPU.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install diffusers==0.35.2 transformers==4.57.3 accelerate==0.34.2 \
  safetensors einops numpy Pillow tqdm ftfy decord imageio imageio-ffmpeg scipy \
  huggingface_hub peft wandb omegaconf insightface onnxruntime opencv-python-headless
uv pip install --no-deps "git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.10"
uv pip install --no-deps -e .
```
> flash-attn is optional (SDPA/VeOmni attention is used if absent) — skip it on Blackwell.

**3. Apply the two Bernini patches**
```bash
git apply ../bernini-headswap-trainer/patches/01_models_init_optional_planner.patch
git apply ../bernini-headswap-trainer/patches/02_transformer_gradient_checkpointing.patch
```
- `01` makes the Qwen planner import optional so the renderer path works without flash-attn.
- `02` adds gradient checkpointing in the transformer block loop (needed to fit memory).

**4. Download the Bernini-R weights (diffusers format)**
```bash
hf download ByteDance/Bernini-R-Diffusers --local-dir /path/to/Bernini-R-Diffusers
```

## Dataset
Triplets of `target` (the head-swapped result, supervised), `guide` (source video,
kept), `reference` (head/face crop, identity). A JSONL with one object per line:
```json
{"vid": "clip_0001", "video_path": ".../target/clip_0001.mp4", "caption": "head_swap:"}
```
`guide` and `reference` are resolved as `<root>/guide/<vid>.mp4` and `<root>/reference/<vid>.png`.
Captions: the trigger `head_swap:` alone works (most of our data was just that); an
optional natural-language or `FACE:/ACTION:` description can follow.

> Note: Bernini uses the **Wan VAE**, so latents are encoded on the fly — LTX/other
> precomputed latents are not reusable.

## Train
```bash
RUN_NAME=headswap_73f640_r64 NUM_FRAMES=73 MAX_SIZE=640 RANK=64 ALPHA=64 \
  bash train/run_headswap_iclora.sh
```
Or directly:
```bash
python train/train_iclora.py \
  --model_dir /path/to/Bernini-R-Diffusers \
  --data_root /path/to/dataset --jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl --out /path/to/out \
  --rank 64 --alpha 64 --lr 1e-4 --num_frames 73 --max_size 640 \
  --max_steps 10000 --save_every 500 --validate_every 500 --val_n 3 --val_steps 20
```
- **Single-expert offload is the default** (only the routed expert on GPU). Use
  `--both_experts` for low frame counts to keep both resident (faster, more VRAM).
- VRAM refs (single-expert): 121f/256 ≈ 45 GB, 73f/640 ≈ 71 GB. Keep margin —
  validation generation spikes; ~640 long edge rides near a 96 GB cap.
- Watch `val/arcface_mean` (FM loss is ~flat; ArcFace is the real signal).

## Export for ComfyUI
```bash
python train/export_lora_comfy.py --ckpt /path/to/out/lora_step3000.safetensors --alpha 64
# -> lora_step3000_high_noise.safetensors + lora_step3000_low_noise.safetensors
```
In ComfyUI: base = **Bernini-R** high/low (e.g. `Comfy-Org/Bernini-R`, NOT vanilla
Wan2.2); apply the high LoRA to the high-noise model and low to the low-noise model.

## License
Trainer code: Apache-2.0. Built on [ByteDance/Bernini](https://github.com/bytedance/Bernini) (Apache-2.0).
