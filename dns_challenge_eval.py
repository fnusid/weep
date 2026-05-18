import os
import glob
import json
import argparse
from pathlib import Path
import contextlib

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

import sys
sys.path.append("/home/sidharth./codebase/")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper
# from dc_crn import DCCRN
# from models.dpcnn import DPCCN
from models.convtasnet import ConvTasNet
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from tqdm.auto import tqdm
import yaml
from pathlib import Path
from omegaconf import OmegaConf

config_path = Path("/home/sidharth./codebase/wesep/confs/config_tasnet.yaml")

with config_path.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]

hp = OmegaConf.merge(*docs)

# -------------------------
# Checkpoint helpers
# -------------------------
def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "")
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new_state[k2] = v
    return new_state


def load_teacher_single_speaker_model(ckpt_path: str, emb_dim: int, device: torch.device):
    model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    filtered = {}
    for k, v in sd.items():
        if k.startswith("model.") and ("arcface" not in k):
            filtered[k.replace("model.", "", 1)] = v

    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_dual_speaker_model(ckpt_path: str, emb_dim: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = strip_dual_model_weights(ckpt["state_dict"])

    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_enhancer(ckpt_path: str, device: torch.device):
    dpccn = ConvTasNet(**hp.model_args.tse_model)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    dpccn_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            dpccn_sd[k[len("model."):]] = v
    if len(dpccn_sd) == 0:
        for k, v in sd.items():
            if k.startswith("model.model."):
                dpccn_sd[k[len("model.model."):]] = v

    if len(dpccn_sd) == 0:
        raise RuntimeError(f"Couldn't find DPCCN weights in: {ckpt_path}")

    dpccn.load_state_dict(dpccn_sd, strict=False)
    dpccn.to(device).eval()
    return dpccn


# -------------------------
# Utils
# -------------------------
def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(dim=-1)
    an = a.norm(dim=-1) + 1e-8
    bn = b.norm(dim=-1) + 1e-8
    return dot / (an * bn)


def list_audio_files(folder: str):
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C,T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.clamp(-1.0, 1.0)  # [1,T]


def cap_or_pad_1d(wav_bt: torch.Tensor, max_len: int) -> torch.Tensor:
    """wav_bt: [1,T] -> [1,max_len] by cropping or right-padding zeros"""
    T = wav_bt.shape[-1]
    if T > max_len:
        return wav_bt[..., :max_len]
    if T < max_len:
        return F.pad(wav_bt, (0, max_len - T))
    return wav_bt


def maybe_autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


# -------------------------
# Embedding selection
# -------------------------
@torch.no_grad()
def teacher_embed_enroll(teacher, enroll_bt, sr: int, enroll_sec: float):
    L = int(sr * enroll_sec)
    enroll_bt = cap_or_pad_1d(enroll_bt, L)
    return teacher(enroll_bt)  # [1,D]


@torch.no_grad()
def dual_embed_select_chunked(
    dual,
    noisy_bt,          # [1,T] on device
    e_enroll,          # [1,D] on device
    sr: int,
    chunk_threshold_sec: float,
    chunk_sec: float,
    hop_sec: float,
    device: torch.device,
    use_autocast: bool,
):
    """
    Returns:
      e_sel: [1,D] selected embedding
      dbg: dict with pick/cos/margin/chunk_start
    """
    T = noisy_bt.shape[-1]
    thr = int(sr * chunk_threshold_sec)

    # If short enough, do single pass
    if T <= thr:
        with maybe_autocast(use_autocast, device):
            embs = dual(noisy_bt)  # [1,2,D]
        e1, e2 = embs[:, 0, :], embs[:, 1, :]
        c1, c2 = cosine(e1, e_enroll), cosine(e2, e_enroll)
        choose_first = (c1 > c2).unsqueeze(-1)
        e_sel = torch.where(choose_first, e1, e2)
        dbg = {
            "pick": int((c1 > c2).item()),
            "cos1": float(c1.item()),
            "cos2": float(c2.item()),
            "cos_margin": float((torch.max(c1, c2) - torch.min(c1, c2)).item()),
            "chunk_start": 0,
            "best_score": float(torch.max(c1, c2).item()),
            "used_chunking": False,
        }
        return e_sel, dbg

    # Otherwise: chunk over noisy
    L = int(sr * chunk_sec)
    H = int(sr * hop_sec)

    best_score = -1e9
    best = None

    for start in range(0, T, H):
        end = min(start + L, T)
        chunk = noisy_bt[..., start:end]
        chunk = cap_or_pad_1d(chunk, L)

        with maybe_autocast(use_autocast, device):
            embs = dual(chunk)  # [1,2,D]

        e1, e2 = embs[:, 0, :], embs[:, 1, :]
        c1, c2 = cosine(e1, e_enroll), cosine(e2, e_enroll)

        score = float(torch.max(c1, c2).item())
        if score > best_score:
            best_score = score
            best = (start, c1, c2, e1, e2)

        if end == T:
            break

    start, c1, c2, e1, e2 = best
    choose_first = (c1 > c2).unsqueeze(-1)
    e_sel = torch.where(choose_first, e1, e2)

    dbg = {
        "pick": int((c1 > c2).item()),
        "cos1": float(c1.item()),
        "cos2": float(c2.item()),
        "cos_margin": float((torch.max(c1, c2) - torch.min(c1, c2)).item()),
        "chunk_start": int(start),
        "best_score": float(best_score),
        "used_chunking": True,
    }
    return e_sel, dbg


# -------------------------
# Enhancement chunking (Overlap-Add)
# -------------------------
def _get_wav_from_enh_out(enh_out):
    if isinstance(enh_out, (tuple, list)):
        return enh_out[0]
    return enh_out


@torch.no_grad()
def enhance_chunked_ola(
    enhancer,
    noisy_bt: torch.Tensor,   # [1, T] on device
    emb: torch.Tensor,        # [1, D] on device
    sr: int,
    chunk_sec: float,
    hop_sec: float,
    device: torch.device,
    use_autocast: bool,
):
    """
    Chunked enhancement with overlap-add (Hann window).
    Returns: pred_bt [1, T] on CPU float32
    """
    T = noisy_bt.shape[-1]
    L = int(sr * chunk_sec)
    H = int(sr * hop_sec)
    assert L > 0 and H > 0 and H <= L

    win = torch.hann_window(L, periodic=True, device=device).unsqueeze(0)  # [1, L]
    eps = 1e-8

    out_acc = torch.zeros((1, T + L), device=device)
    w_acc = torch.zeros((1, T + L), device=device)

    for start in range(0, T, H):
        end = min(start + L, T)
        x = noisy_bt[..., start:end]
        if x.shape[-1] < L:
            x = F.pad(x, (0, L - x.shape[-1]))

        with maybe_autocast(use_autocast, device):
            y = _get_wav_from_enh_out(enhancer(x, emb))

        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = F.pad(y, (0, L - y.shape[-1]))

        out_acc[..., start:start + L] += y * win
        w_acc[..., start:start + L] += win

        if end == T:
            break

    pred = out_acc[..., :T] / (w_acc[..., :T] + eps)
    return pred.detach().cpu().float()


@torch.no_grad()
def enhance_full_or_chunked(
    enhancer,
    noisy_bt: torch.Tensor,
    emb: torch.Tensor,
    sr: int,
    threshold_sec: float,
    chunk_sec: float,
    hop_sec: float,
    device: torch.device,
    use_autocast: bool,
):
    T = noisy_bt.shape[-1]
    thr = int(sr * threshold_sec)

    if T > thr:
        return enhance_chunked_ola(enhancer, noisy_bt, emb, sr, chunk_sec, hop_sec, device, use_autocast)

    try:
        with maybe_autocast(use_autocast, device):
            enh_out = enhancer(noisy_bt, emb)
        pred = _get_wav_from_enh_out(enh_out)
        return pred[..., :T].detach().cpu().float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return enhance_chunked_ola(enhancer, noisy_bt, emb, sr, chunk_sec, hop_sec, device, use_autocast)


# -------------------------
# DNSMOS P.835
# -------------------------
@torch.no_grad()
def dnsmos_p835_batch(wav_bt: torch.Tensor, fs: int, personalized: bool,
                      device: str = None, num_threads: int = None):
    """
    returns [B] p808, sig, bak, ovrl
    """
    scores = deep_noise_suppression_mean_opinion_score(
        wav_bt,
        fs=fs,
        personalized=personalized,
        device=device,
        num_threads=num_threads,
        cache_session=True,
    )  # [B,4] = [p808, sig, bak, ovrl]
    return scores[:, 0], scores[:, 1], scores[:, 2], scores[:, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns_root", type=str, required=True)
    ap.add_argument("--track", type=str, default="Track1_Headset")
    ap.add_argument("--noisy_subdir", type=str, default="noisy")
    ap.add_argument("--enroll_subdir", type=str, default="enrol")

    ap.add_argument("--dual_ckpt", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--enh_ckpt", type=str, required=True)

    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--dnsmos_personalized", action="store_true")
    ap.add_argument("--dnsmos_device", type=str, default=None)
    ap.add_argument("--dnsmos_num_threads", type=int, default=None)

    ap.add_argument("--out_dir", type=str, default="./dns_p835_out_tasnet")
    ap.add_argument("--save_wavs", action="store_true")

    # Embedding chunking controls
    ap.add_argument("--chunk_threshold_sec", type=float, default=12.0)
    ap.add_argument("--chunk_sec", type=float, default=8.0)
    ap.add_argument("--hop_sec", type=float, default=8.0)
    ap.add_argument("--enroll_sec", type=float, default=10.0)
    ap.add_argument("--dual_autocast_fp16", action="store_true")

    # Enhancement chunking controls
    ap.add_argument("--enh_chunk_threshold_sec", type=float, default=12.0)
    ap.add_argument("--enh_chunk_sec", type=float, default=8.0)
    ap.add_argument("--enh_hop_sec", type=float, default=4.0)
    ap.add_argument("--enh_autocast_fp16", action="store_true")

    # Safety
    ap.add_argument("--skip_on_oom", action="store_true")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    base = Path(args.dns_root) / args.track
    noisy_dir = base / args.noisy_subdir
    enroll_dir = base / args.enroll_subdir
    assert noisy_dir.exists(), f"Missing noisy dir: {noisy_dir}"
    assert enroll_dir.exists(), f"Missing enroll dir: {enroll_dir}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_wavs:
        (out_dir / "enhanced").mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading models...")
    teacher = load_teacher_single_speaker_model(args.teacher_ckpt, args.emb_dim, device)
    dual = load_dual_speaker_model(args.dual_ckpt, args.emb_dim, device)
    enhancer = load_enhancer(args.enh_ckpt, device)

    noisy_files = list_audio_files(str(noisy_dir))
    if len(noisy_files) == 0:
        raise RuntimeError(f"No audio found in {noisy_dir}")

    per_file = []

    with torch.no_grad():
        pbar = tqdm(noisy_files, total=len(noisy_files), desc="DNS eval", unit="file", dynamic_ncols=True)

        for idx, noisy_path in enumerate(pbar):
            fname = Path(noisy_path).name
            enroll_path = enroll_dir / fname
            if not enroll_path.exists():
                continue

            noisy = load_audio_mono(noisy_path, args.sr)             # [1,T]
            enroll = load_audio_mono(str(enroll_path), args.sr)      # [1,T]

            noisy_bt = noisy.to(device)    # [1,T]
            enroll_bt = enroll.to(device)

            # enrollment embedding (cap for safety)
            try:
                e_enroll = teacher_embed_enroll(teacher, enroll_bt, sr=args.sr, enroll_sec=args.enroll_sec)  # [1,D]
            except torch.cuda.OutOfMemoryError:
                if args.skip_on_oom:
                    torch.cuda.empty_cache()
                    continue
                raise

            # dual embedding selection (chunk if long)
            try:
                e_sel, dbg = dual_embed_select_chunked(
                    dual=dual,
                    noisy_bt=noisy_bt,
                    e_enroll=e_enroll,
                    sr=args.sr,
                    chunk_threshold_sec=args.chunk_threshold_sec,
                    chunk_sec=args.chunk_sec,
                    hop_sec=args.hop_sec,
                    device=device,
                    use_autocast=args.dual_autocast_fp16,
                )
            except torch.cuda.OutOfMemoryError:
                if args.skip_on_oom:
                    torch.cuda.empty_cache()
                    continue
                raise

            # enhance full clip (or chunked OLA if long / OOM)
            try:
                pred_bt_cpu = enhance_full_or_chunked(
                    enhancer=enhancer,
                    noisy_bt=noisy_bt,
                    emb=e_sel,
                    sr=args.sr,
                    threshold_sec=args.enh_chunk_threshold_sec,
                    chunk_sec=args.enh_chunk_sec,
                    hop_sec=args.enh_hop_sec,
                    device=device,
                    use_autocast=args.enh_autocast_fp16,
                )  # [1, T] CPU float32
            except torch.cuda.OutOfMemoryError:
                if args.skip_on_oom:
                    torch.cuda.empty_cache()
                    continue
                raise

            pred = pred_bt_cpu.squeeze(0)  # [T] CPU float32
            noi = noisy.squeeze(0).detach().cpu().float()  # [T]

            if args.save_wavs:
                torchaudio.save(str(out_dir / "enhanced" / fname), pred.unsqueeze(0), sample_rate=args.sr)

            # DNSMOS P.835 per file (CPU float32)
            noi_bt = noi.unsqueeze(0)
            pred_bt = pred.unsqueeze(0)

            p808_n, sig_n, bak_n, ovrl_n = dnsmos_p835_batch(
                noi_bt, fs=args.sr, personalized=args.dnsmos_personalized,
                device=args.dnsmos_device, num_threads=args.dnsmos_num_threads
            )
            p808_e, sig_e, bak_e, ovrl_e = dnsmos_p835_batch(
                pred_bt, fs=args.sr, personalized=args.dnsmos_personalized,
                device=args.dnsmos_device, num_threads=args.dnsmos_num_threads
            )

            row = {
                "file": fname,
                "pick": int(dbg["pick"]),
                "cos1": float(dbg["cos1"]),
                "cos2": float(dbg["cos2"]),
                "cos_margin": float(dbg["cos_margin"]),
                "chunk_start": int(dbg["chunk_start"]),
                "used_chunking": bool(dbg["used_chunking"]),
                "noisy": {
                    "P808": float(p808_n[0].item()),
                    "SIG":  float(sig_n[0].item()),
                    "BAK":  float(bak_n[0].item()),
                    "OVRL": float(ovrl_n[0].item()),
                },
                "enh": {
                    "P808": float(p808_e[0].item()),
                    "SIG":  float(sig_e[0].item()),
                    "BAK":  float(bak_e[0].item()),
                    "OVRL": float(ovrl_e[0].item()),
                },
            }
            row["delta"] = {k: row["enh"][k] - row["noisy"][k] for k in ["P808", "SIG", "BAK", "OVRL"]}
            per_file.append(row)

            if len(per_file) > 0:
                pbar.set_postfix({
                    "ΔOVRL": f'{np.mean([r["delta"]["OVRL"] for r in per_file]):.3f}',
                    "ΔSIG":  f'{np.mean([r["delta"]["SIG"]  for r in per_file]):.3f}',
                    "ΔBAK":  f'{np.mean([r["delta"]["BAK"]  for r in per_file]):.3f}',
                    "marg":  f'{np.mean([r["cos_margin"]     for r in per_file]):.3f}',
                })

            if (idx + 1) % 50 == 0:
                with open(out_dir / "per_file_partial.json", "w") as f:
                    json.dump(per_file, f, indent=2)

    # Summary
    def mean(xs): return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "num_files": len(per_file),
        "P835_delta_SIG": mean([r["delta"]["SIG"] for r in per_file]),
        "P835_delta_BAK": mean([r["delta"]["BAK"] for r in per_file]),
        "P835_delta_OVRL": mean([r["delta"]["OVRL"] for r in per_file]),
        "avg_cos_margin": mean([r["cos_margin"] for r in per_file]),
        "chunking_used_frac": mean([1.0 if r["used_chunking"] else 0.0 for r in per_file]),
    }

    with open(out_dir / "per_file.json", "w") as f:
        json.dump(per_file, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE]", json.dumps(summary, indent=2))
    print(f"[DONE] Wrote {out_dir / 'summary.json'} and {out_dir / 'per_file.json'}")


if __name__ == "__main__":
    main()