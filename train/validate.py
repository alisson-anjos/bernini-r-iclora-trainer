"""Validation generation + ArcFace identity metric for head-swap IC-LoRA.

Generation reuses the renderer's own rv2v path (BerniniRendererPipeline.__call__):
given a guide video + head reference image + caption, the LoRA-adapted model
renders the swapped clip. The ArcFace metric scores how well the generated face
matches the reference identity (the signal that actually matters — FM loss is
nearly flat for flow-matching).
"""
import os

import numpy as np
import torch

# ArcFace is optional: if insightface isn't installed we still write previews.
try:
    import insightface
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception:
    _HAS_INSIGHT = False

_FACE_APP = None


def _face_app():
    global _FACE_APP
    if _FACE_APP is None and _HAS_INSIGHT:
        # CPU-only: the renderer already occupies most of the GPU during
        # validation, so keep ArcFace off the GPU to avoid OOM/contention.
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


def _embed(img_bgr):
    app = _face_app()
    if app is None:
        return None
    faces = app.get(img_bgr)
    if not faces:
        # Tight head crops (our reference pngs) need margin for RetinaFace to
        # fire; pad with a replicated border and retry.
        import cv2

        h, w = img_bgr.shape[:2]
        padded = cv2.copyMakeBorder(img_bgr, int(h * 0.5), int(h * 0.5),
                                    int(w * 0.5), int(w * 0.5), cv2.BORDER_REPLICATE)
        faces = app.get(padded)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    e = f.normed_embedding
    return e / (np.linalg.norm(e) + 1e-8)


def arcface_similarity(ref_png, frames_rgb):
    """Mean cosine similarity between the reference face and sampled frames.

    `frames_rgb`: np array [T,H,W,C] in [0,1]. Returns float or None.
    """
    app = _face_app()
    if app is None:
        return None
    import cv2

    ref = cv2.imread(ref_png)
    if ref is None:
        return None
    ref_e = _embed(ref)
    if ref_e is None:
        return None
    T = frames_rgb.shape[0]
    idxs = np.linspace(0, T - 1, min(T, 8)).astype(int)
    sims = []
    for i in idxs:
        fr = (frames_rgb[i] * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
        e = _embed(np.ascontiguousarray(fr))
        if e is not None:
            sims.append(float(np.dot(ref_e, e)))
    return float(np.mean(sims)) if sims else None


@torch.no_grad()
def run_validation(renderer, recs, out_dir, step, num_frames=41, max_size=448,
                   num_inference_steps=30, guidance_mode="rv2v"):
    """Generate previews for each val record; return mean arcface (or None).

    Reuses the in-memory LoRA-adapted model. sample() shuffles experts across
    GPU/CPU, so we restore both to the training device afterwards.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = renderer.device
    sims = []
    renderer.model.t5_text_encoder.to(device)
    for rec in recs:
        out_path = os.path.join(out_dir, f"step{step}_{rec['vid']}.mp4")
        video = renderer.pipe(
            prompt=rec["caption"],
            video=rec["guide"],
            image=rec["ref"],
            num_frames=num_frames,
            max_image_size=max_size,
            num_inference_steps=num_inference_steps,
            guidance_mode=guidance_mode,
            output_path=out_path,
            write_output=True,
        )
        if video is not None:
            s = arcface_similarity(rec["ref"], np.asarray(video))
            if s is not None:
                sims.append(s)
                print(f"[val] step{step} {rec['vid']} arcface={s:.3f} -> {out_path}")
            else:
                print(f"[val] step{step} {rec['vid']} (no face / no insightface) -> {out_path}")

    # restore both experts to the training device for the next train step
    for name in ("transformer", "transformer_2"):
        tf = getattr(renderer.diff, name, None)
        if tf is not None:
            tf.to(device, dtype=torch.bfloat16)
    torch.cuda.empty_cache()

    mean = float(np.mean(sims)) if sims else None
    if mean is not None:
        print(f"[val] step{step} MEAN arcface={mean:.3f} over {len(sims)} clips")
        with open(os.path.join(out_dir, "validation_metrics.jsonl"), "a") as f:
            import json
            f.write(json.dumps({"step": step, "arcface_mean": mean, "n": len(sims)}) + "\n")
    return mean
