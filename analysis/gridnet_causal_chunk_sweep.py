import argparse
import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_recording import enhance_full_or_chunked, load_audio_mono, load_embedding_model, load_tse_model


DEFAULT_INPUT = Path("/home/sidcs/codebase/wesep/analysis/real_world_wavs/sid_randbaldman_model_input.wav")
DEFAULT_CKPT = Path(
    "/home/sidcs/model_ckpts/causal_gridnet_2sp_use_attn_True_mrstft_0.1_libri05_08/"
    "epochepoch=102-trainlosstrain_loss=-4.285.ckpt"
)
DEFAULT_OUTDIR = Path("/home/sidcs/codebase/wesep/analysis/real_world_wavs/gridnet_causal_chunk_sweep")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run causal GridNet on one audio file and save outputs for multiple "
            "chunk sizes. Since no clean target is available, the script saves "
            "both slot-conditioned outputs for each chunk setting."
        )
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--tse-ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument(
        "--embedding-ckpt",
        type=Path,
        default=None,
        help="Optional separate dual-embedding checkpoint. Omit for joint checkpoints.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument(
        "--chunk-secs",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 8.0, 16.0, 30.0, 60.0],
        help="Chunk sizes in seconds to evaluate through the chunked OLA path.",
    )
    ap.add_argument(
        "--hop-frac",
        type=float,
        default=1.0,
        help="Hop as a fraction of chunk size. Use 1.0 for no overlap, 0.5 for 50%% overlap.",
    )
    ap.add_argument(
        "--include-full-pass",
        action="store_true",
        help="Also save outputs from a single full forward pass with chunking disabled.",
    )
    ap.add_argument("--autocast-fp16", action="store_true")
    return ap.parse_args()


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.unsqueeze(0).cpu(), sample_rate=sample_rate)


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tse_model, tse_info = load_tse_model("gridnet", None, args.tse_ckpt, device)
    emb_model, emb_info = load_embedding_model(args.embedding_ckpt, args.tse_ckpt, args.emb_dim, device)
    print(
        f"[info] loaded separator via {tse_info['label']}; "
        f"embedding via {emb_info['label']}"
    )

    mixture = load_audio_mono(args.input, args.sr)
    mix_bt = mixture.to(device)
    with torch.no_grad():
        embeddings = emb_model(mix_bt)

    stem = args.input.stem

    if args.include_full_pass:
        for slot_idx in range(2):
            emb = embeddings[:, slot_idx, :]
            pred = enhance_full_or_chunked(
                model_name="gridnet",
                enhancer=tse_model,
                noisy_bt=mix_bt,
                emb=emb,
                sr=args.sr,
                chunk_threshold_sec=1000.0,
                chunk_sec=1000.0,
                hop_sec=1000.0,
                device=device,
                use_autocast=args.autocast_fp16,
            ).squeeze(0)
            out_path = args.out_dir / f"{stem}_fullpass_slot{slot_idx + 1}.wav"
            save_wav(out_path, pred, args.sr)
            print(f"[info] saved {out_path}")

    for chunk_sec in args.chunk_secs:
        hop_sec = chunk_sec * args.hop_frac
        if hop_sec <= 0:
            raise ValueError("hop-frac must produce a positive hop size.")
        for slot_idx in range(2):
            emb = embeddings[:, slot_idx, :]
            pred = enhance_full_or_chunked(
                model_name="gridnet",
                enhancer=tse_model,
                noisy_bt=mix_bt,
                emb=emb,
                sr=args.sr,
                chunk_threshold_sec=0.0,
                chunk_sec=chunk_sec,
                hop_sec=hop_sec,
                device=device,
                use_autocast=args.autocast_fp16,
            ).squeeze(0)
            tag = f"chunk{str(chunk_sec).replace('.', 'p')}_hop{str(hop_sec).replace('.', 'p')}"
            out_path = args.out_dir / f"{stem}_{tag}_slot{slot_idx + 1}.wav"
            save_wav(out_path, pred, args.sr)
            print(f"[info] saved {out_path}")


if __name__ == "__main__":
    main()
