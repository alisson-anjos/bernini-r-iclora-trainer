"""IC-LoRA head-swap training for the Bernini-R (Wan2.2-A14B) renderer.

The renderer natively concatenates conditioning sources into one self-attention
sequence, disambiguated by a per-source RoPE phase (source_id). We freeze the
base dual-expert transformer and train LoRA adapters on a head-swap triplet:

    [ source video (src_id=1) | head reference image (src_id=2) | noisy target (src_id=0) ]

Flow-matching v-target, loss on the target tokens only. This mirrors the VI
combo of GEN_Wanx22.sample() (minus guidance), so what we train is exactly what
inference runs.
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, "/data/bernini-src")
from bernini.pipeline import BerniniRendererPipeline, _vae_encode
from bernini.data_utils import preprocess_video, preprocess_image, make_divisible
from train.lora_targets import inject_lora

NUM_TRAIN_TIMESTEPS = 1000

# latent [B,C,T,H,W] <-> packed tokens [B, T*(H/2)*(W/2), 4C]  (pt=1, ph=pw=2)
_PACK = "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)"
_UNPACK = "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)"


def to_packed(x, shape):
    return rearrange(x, _PACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def to_spatial(x, shape):
    return rearrange(x, _UNPACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def sample_sigma(shift: float) -> float:
    """Uniform u in (0,1) pushed through the flow-matching shift schedule."""
    u = random.random()
    return shift * u / (1 + (shift - 1) * u)


def _u_boundary(shift: float, sigma_b: float) -> float:
    """u such that shift-mapped sigma == sigma_b (the expert switch boundary)."""
    return sigma_b / (shift - sigma_b * (shift - 1))


def sample_sigma_side(shift: float, u_b: float, high: bool) -> float:
    """Sample sigma on one side of the boundary, preserving the u-uniform law."""
    u = (u_b + (1.0 - u_b) * random.random()) if high else (u_b * random.random())
    return shift * u / (1 + (shift - 1) * u)


def move_optimizer_state(opt, device):
    """Move an optimizer's tensor state to `device` (params move with the module)."""
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_records(jsonl_path, root):
    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d["vid"]
            target = d.get("video_path") or os.path.join(root, "target", f"{vid}.mp4")
            guide = os.path.join(root, "guide", f"{vid}.mp4")
            ref = os.path.join(root, "reference", f"{vid}.png")
            if not (os.path.exists(target) and os.path.exists(guide) and os.path.exists(ref)):
                continue
            recs.append({"vid": vid, "target": target, "guide": guide, "ref": ref,
                         "caption": d["caption"]})
    return recs


class Renderer:
    """Wraps the loaded pipeline and exposes the train forward."""

    NAME_OF = {"transformer_1": "transformer", "transformer_2": "transformer_2"}

    def __init__(self, model_dir, rank, alpha, dropout, device="cuda", single_expert=True):
        self.device = device
        self.single_expert = single_expert
        self.pipe = BerniniRendererPipeline.from_pretrained(
            model_dir, device=device, load_ckpt_weights=False, use_src_id_rotary_emb=True,
        )
        self.model = self.pipe.model              # BerniniRendererModel
        self.diff = self.model.diff_dec           # GEN_Wanx22 (dual expert)
        self.vae = self.pipe.vae                  # AutoencoderKLWan (fp32)
        self.tokenizer = self.pipe.tokenizer
        self.sigma_boundary = float(self.diff.switch_dit_boundary)   # sigma == timestep/1000
        self.boundary = self.sigma_boundary * NUM_TRAIN_TIMESTEPS
        self.shift = float(getattr(self.diff.scheduler, "shift", 3.0))
        self.u_b = _u_boundary(self.shift, self.sigma_boundary)
        self.p_high = 1.0 - self.u_b               # P(step routes to the high-noise expert)

        # inject LoRA on both experts, freeze base. In single-expert mode only the
        # routed expert lives on GPU at a time (the other parks on CPU) so high
        # frame counts fit in one GPU.
        self.lora_params = {}                      # model_id -> [trainable params]
        self.current_expert = None
        for name, mid in (("transformer", "transformer_1"), ("transformer_2", "transformer_2")):
            tf = getattr(self.diff, name, None)
            if tf is None:
                continue
            tf, n = inject_lora(tf, rank, alpha, dropout)
            tf.to(dtype=torch.bfloat16)
            tf.gradient_checkpointing = True
            tf.train()  # base params frozen; train() only toggles ckpt/dropout path
            tf.to("cpu" if single_expert else device)
            self.lora_params[mid] = [p for p in tf.parameters() if p.requires_grad]
            print(f"[lora] {name} ({mid}): {n/1e6:.1f}M trainable")
        self.trainable = [p for grp in self.lora_params.values() for p in grp]
        self.model.t5_text_encoder.to(device)

    def ensure_expert(self, model_id, opts=None):
        """Make `model_id` the GPU-resident expert; park the other on CPU."""
        if not self.single_expert or self.current_expert == model_id:
            self.current_expert = model_id
            return
        if self.current_expert is not None:
            cur = getattr(self.diff, self.NAME_OF[self.current_expert])
            cur.to("cpu")
            if opts and self.current_expert in opts:
                move_optimizer_state(opts[self.current_expert], "cpu")
        tgt = getattr(self.diff, self.NAME_OF[model_id])
        tgt.to(self.device)
        if opts and model_id in opts:
            move_optimizer_state(opts[model_id], self.device)
        self.current_expert = model_id
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_inputs(self, rec, num_frames, max_size):
        # sample() parks the t5 encoder on CPU when it finishes; bring it back.
        self.model.t5_text_encoder.to(self.device)
        self.vae.to(self.device)
        pv_t = preprocess_video(rec["target"], fps=16, max_image_size=max_size,
                                max_image_num=num_frames, device=self.device)
        pv_s = preprocess_video(rec["guide"], fps=16, max_image_size=max_size,
                                max_image_num=num_frames, device=self.device)
        pi_r = preprocess_image(rec["ref"], max_image_size=max_size, device=self.device)
        tgt = _vae_encode(self.vae, pv_t)          # [1,C,T,H,W] normalized
        src = _vae_encode(self.vae, pv_s)
        ref = _vae_encode(self.vae, pi_r)
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        ids, mask = self._tok(rec["caption"])
        cond = self.model.encode_prompt(ids.to(self.device), mask.to(self.device))
        # park the (large) UMT5 encoder on CPU so the GPU is free for the expert
        self.model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        return tgt, src, ref, cond

    def _tok(self, text):
        out = self.tokenizer(text, padding="max_length", max_length=512, truncation=True,
                             add_special_tokens=True, return_attention_mask=True,
                             return_tensors="pt")
        return out.input_ids, out.attention_mask

    def train_loss(self, tgt, src, ref, cond, model_id=None):
        shape = tgt.shape  # [1,C,T,H,W]
        # The loop fixes which expert is GPU-resident; sample sigma on that side.
        if model_id is None:
            sigma = sample_sigma(self.shift)
            model_id = "transformer_1" if sigma >= self.sigma_boundary else "transformer_2"
        else:
            sigma = sample_sigma_side(self.shift, self.u_b, high=(model_id == "transformer_1"))
        timestep = sigma * NUM_TRAIN_TIMESTEPS
        tf = self.diff.transformer if model_id == "transformer_1" else self.diff.transformer_2
        if tf is None:
            tf = self.diff.transformer
            model_id = "transformer_1"

        tgt_packed = to_packed(tgt.float(), shape)            # [1, ntok, 4C]
        noise = torch.randn_like(tgt_packed)
        noisy_packed = (1 - sigma) * tgt_packed + sigma * noise
        target_velocity = (noise - tgt_packed).to(torch.bfloat16)

        dt = tf.dtype
        src_tok, src_rope = tf.patch_vae_latent(src.to(dt), source_id=1.0)
        ref_tok, ref_rope = tf.patch_vae_latent(ref.to(dt), source_id=2.0)
        noisy_spatial = to_spatial(noisy_packed, shape).to(dt)
        noisy_tok, noisy_rope = tf.patch_vae_latent(noisy_spatial, source_id=0.0)

        hidden = torch.cat([src_tok, ref_tok, noisy_tok], dim=1)
        rope = torch.cat([src_rope, ref_rope, noisy_rope], dim=2)
        total = hidden.shape[1]
        noisy_len = noisy_tok.shape[1]
        msk = torch.zeros(total, dtype=torch.bool, device=hidden.device)
        msk[-noisy_len:] = True

        ts = torch.tensor([timestep], device=hidden.device, dtype=torch.float32)
        pred = self.diff.shared_step(
            model_id=model_id, noisy_latents=hidden, timesteps=ts,
            cond_embeds=cond.to(dt), rotary_embs=rope,
            batch_vae_seqlen=[total], batch_text_seqlen=[cond.shape[1]],
        )
        pred_target = pred[:, msk, :]                          # [1, noisy_len, 4C]
        loss = F.mse_loss(pred_target.float(), target_velocity.float())
        return loss, model_id


def save_lora(renderer, path):
    sd = {}
    for name in ("transformer", "transformer_2"):
        tf = getattr(renderer.diff, name, None)
        if tf is None:
            continue
        for k, v in tf.state_dict().items():
            if ".lora_" in k:
                sd[f"{name}.{k}"] = v.detach().cpu()
    from safetensors.torch import save_file
    save_file(sd, path)
    print(f"[save] {len(sd)} lora tensors -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/data/models/Bernini-R-Diffusers")
    ap.add_argument("--data_root", default="/data/datasets/head_swap_unified")
    ap.add_argument("--jsonl", default="/data/datasets/head_swap_unified/dataset_train_musubi_videos_train.jsonl")
    ap.add_argument("--out", default="/data/training/bernini/headswap_iclora")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_frames", type=int, default=41)
    ap.add_argument("--max_size", type=int, default=448)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--val_jsonl", default="/data/datasets/head_swap_unified/dataset_train_musubi_videos_validation.jsonl")
    ap.add_argument("--validate_every", type=int, default=500)
    ap.add_argument("--val_n", type=int, default=4, help="number of validation clips")
    ap.add_argument("--val_steps", type=int, default=30, help="inference steps for validation")
    ap.add_argument("--no_val", action="store_true")
    ap.add_argument("--wandb_project", default="bernini-headswap")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--both_experts", action="store_true",
                    help="keep both experts on GPU (only for low frame counts). Default: single-expert offload.")
    ap.add_argument("--smoke", action="store_true", help="few steps, verbose")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(0); torch.manual_seed(0)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb
            wb.init(project=args.wandb_project,
                    name=args.wandb_name or os.path.basename(args.out.rstrip("/")),
                    config=vars(args))
        except Exception as e:  # offline / not logged in -> keep training
            print(f"[wandb] disabled ({e})")
            wb = None

    recs = load_records(args.jsonl, args.data_root)
    print(f"[data] {len(recs)} usable triplets")
    val_recs = load_records(args.val_jsonl, args.data_root)[: args.val_n] if not args.no_val else []
    print(f"[data] {len(val_recs)} validation triplets")
    if args.smoke:
        recs = recs[:4]
        val_recs = val_recs[:1]
        args.max_steps = 3

    single_expert = not args.both_experts
    r = Renderer(args.model_dir, args.rank, args.alpha, args.dropout, single_expert=single_expert)
    # one optimizer per expert so its state can be parked on CPU with the expert.
    opts = {mid: torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
            for mid, params in r.lora_params.items()}
    print(f"[mode] single_expert={single_expert} | experts={list(r.lora_params)} | P(high)={r.p_high:.2f}")

    val_dir = os.path.join(args.out, "validation")

    def validate(step):
        if not val_recs:
            return
        from train.validate import run_validation
        for name in ("transformer", "transformer_2"):
            tf = getattr(r.diff, name, None)
            if tf is not None:
                tf.eval()
        mean = run_validation(r, val_recs, val_dir, step, num_frames=args.num_frames,
                              max_size=args.max_size, num_inference_steps=args.val_steps)
        for name in ("transformer", "transformer_2"):
            tf = getattr(r.diff, name, None)
            if tf is not None:
                tf.train()
        if single_expert:
            # sampling shuffled the experts across devices; reset so the next
            # train step re-establishes the GPU-resident one.
            for name in ("transformer", "transformer_2"):
                tf = getattr(r.diff, name, None)
                if tf is not None:
                    tf.to("cpu")
            for mid in opts:
                move_optimizer_state(opts[mid], "cpu")
            r.current_expert = None
            torch.cuda.empty_cache()
        if wb is not None:
            import glob
            log = {}
            if mean is not None:
                log["val/arcface_mean"] = mean
            vids = sorted(glob.glob(os.path.join(val_dir, f"step{step}_*.mp4")))
            log.update({f"val/preview_{i}": wb.Video(v, fps=16, format="mp4")
                        for i, v in enumerate(vids)})
            if log:
                wb.log(log, step=step)

    if not args.smoke:
        validate(0)  # baseline before any training

    def pick_expert():
        mid = "transformer_1" if random.random() < r.p_high else "transformer_2"
        return mid if mid in r.lora_params else next(iter(r.lora_params))

    step = 0
    cur_mid = None
    while step < args.max_steps:
        random.shuffle(recs)
        for rec in recs:
            # fix the expert for each grad-accum group -> at most one swap per update
            if step % args.grad_accum == 0:
                cur_mid = pick_expert()
                r.ensure_expert(cur_mid, opts)
            tgt, src, ref, cond = r.encode_inputs(rec, args.num_frames, args.max_size)
            loss, model_id = r.train_loss(tgt, src, ref, cond, model_id=cur_mid)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(r.lora_params[cur_mid], 1.0)
                opts[cur_mid].step(); opts[cur_mid].zero_grad(set_to_none=True)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:5d} | {model_id:13s} | loss {loss.item():.4f} | {rec['vid']} | peakVRAM {mem:.1f}G")
            if wb is not None:
                wb.log({"train/loss": loss.item(), "train/lr": args.lr,
                        "train/expert": 1 if model_id == "transformer_1" else 2,
                        "train/peak_vram_gb": mem}, step=step)
            step += 1
            if step % args.save_every == 0:
                save_lora(r, os.path.join(args.out, f"lora_step{step}.safetensors"))
            if args.validate_every and step % args.validate_every == 0:
                validate(step)
            if step >= args.max_steps:
                break

    save_lora(r, os.path.join(args.out, "lora_final.safetensors"))
    print("DONE_TRAIN")


if __name__ == "__main__":
    main()
