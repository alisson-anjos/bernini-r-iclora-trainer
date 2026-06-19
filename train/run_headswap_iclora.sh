#!/usr/bin/env bash
# Bernini-R (Wan2.2-A14B) head-swap IC-LoRA training.
# Base frozen, LoRA r128/a64 on both experts, flow-matching v-target on the
# target tokens of [source(sid1) | head-ref(sid2) | noisy-target(sid0)].
#
# VRAM note (RTX PRO 6000, 96GB): smoke at 21f/320px peaked ~76GB with gradient
# checkpointing. Activations scale ~ frames * resolution^2. If you OOM, drop
# NUM_FRAMES (use 4k+1: 21/41/81) or MAX_SIZE first.
set -e
cd /data/bernini-src
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single-expert offload (default) keeps only the routed expert on GPU, so high
# frame counts fit. Measured peaks (single-expert): 121f/256 ~45GB, 73f/640 ~71GB.
# Dataset is ~1024 long edge (mostly portrait); majority of clips are 73 frames.
RUN_NAME="${RUN_NAME:-headswap_73f640_r64_10k}"
NUM_FRAMES="${NUM_FRAMES:-73}"
MAX_SIZE="${MAX_SIZE:-640}"
RANK="${RANK:-64}"
ALPHA="${ALPHA:-64}"

python train/train_iclora.py \
  --model_dir /data/models/Bernini-R-Diffusers \
  --data_root /data/datasets/head_swap_unified \
  --jsonl /data/datasets/head_swap_unified/dataset_train_musubi_videos_train.jsonl \
  --val_jsonl /data/datasets/head_swap_unified/dataset_train_musubi_videos_validation.jsonl \
  --out "/data/training/bernini/${RUN_NAME}" \
  --rank "${RANK}" --alpha "${ALPHA}" --lr 1e-4 \
  --num_frames "${NUM_FRAMES}" --max_size "${MAX_SIZE}" \
  --max_steps 10000 --save_every 500 \
  --validate_every 500 --val_n 3 --val_steps 20
