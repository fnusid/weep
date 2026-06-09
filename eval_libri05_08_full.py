import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CODEBASE_ROOT = REPO_ROOT.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

from infer_recording import DEFAULT_CONFIGS, enhance_full_or_chunked, load_embedding_model, load_tse_model
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper


EPS = 1e-8
DEFAULT_TEACHER_CKPT = Path(
    "/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
)
DEFAULT_METADATA = Path(
    "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/"
    "Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_test_mix_both.csv"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate the untouched Libriuni_05_08 test set with full-file TSE, "
            "using clean teacher embeddings to choose the target-conditioned "
            "mixture embedding the same way the repo's test path does."
        )
    )
    ap.add_argument("--model", choices=["dpccn"], required=True)
    ap.add_argument("--tse-ckpt", type=Path, required=True)
    ap.add_argument("--embedding-ckpt", type=Path, default=None)
    ap.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER_CKPT)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--chunk-threshold-sec", type=float, default=20.0)
    ap.add_argument("--chunk-sec", type=float, default=8.0)
    ap.add_argument("--hop-sec", type=float, default=4.0)
    ap.add_argument("--autocast-fp16", action="store_true")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument(
        "--target-mode",
        choices=["both", "random"],
        default="both",
        help="Evaluate both target speakers per mixture, or one random target like the training test step.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def normalize_mixboth_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if cols[:8] != [
        "mixture_ID",
        "mixture_path",
        "source_1_path",
        "speaker_1_ID",
        "source_2_path",
        "speaker_2_ID",
        "noise_path",
        "length",
    ]:
        return df

    sample = df.iloc[0]
    if isinstance(sample["speaker_1_ID"], str) and sample["speaker_1_ID"].endswith(".wav"):
        raw = df.iloc[:, :8].copy()
        raw.columns = [
            "mixture_ID",
            "mixture_path",
            "source_1_path",
            "source_2_path",
            "speaker_1_ID",
            "speaker_2_ID",
            "noise_path",
            "length",
        ]
        return raw
    return df


def load_audio_mono(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.clamp(-1.0, 1.0)


def load_teacher_model(ckpt_path: Path, emb_dim: int, device: torch.device):
    model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["state_dict"]
    filtered = {}
    for key, value in state.items():
        if key.startswith("model.") and "arcface" not in key:
            filtered[key.replace("model.", "", 1)] = value
    model.load_state_dict(filtered, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(dim=-1)
    an = a.norm(dim=-1) + 1e-8
    bn = b.norm(dim=-1) + 1e-8
    return dot / (an * bn)


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    est = est.float()
    ref = ref.float()
    ref_energy = torch.sum(ref * ref) + EPS
    projection = torch.sum(est * ref) * ref / ref_energy
    noise = est - projection
    ratio = (torch.sum(projection * projection) + EPS) / (torch.sum(noise * noise) + EPS)
    return float(10.0 * torch.log10(ratio).item())


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.unsqueeze(0).cpu(), sample_rate=sample_rate)


def format_summary_table(summary: dict) -> str:
    headers = ["Condition", "SI-SDR", "SI-SDRi"]
    rows = [
        ["Mixture", f"{summary['mean_mixture_si_sdr']:.3f}", "-"],
        ["Full", f"{summary['mean_full_si_sdr']:.3f}", f"{summary['mean_full_si_sdri']:.3f}"],
    ]
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt_row(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    return "\n".join([fmt_row(headers), sep, *(fmt_row(row) for row in rows)])


def main():
    args = parse_args()
    rng = torch.Generator().manual_seed(args.seed)
    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    metadata = normalize_mixboth_metadata(pd.read_csv(args.metadata_csv))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config
    if args.model in DEFAULT_CONFIGS:
        config_path = config_path or DEFAULT_CONFIGS[args.model]

    tse_model, tse_info = load_tse_model(args.model, config_path, args.tse_ckpt, device)
    emb_model, emb_info = load_embedding_model(args.embedding_ckpt, args.tse_ckpt, args.emb_dim, device)
    teacher = load_teacher_model(args.teacher_ckpt, args.emb_dim, device)
    print(
        f"[info] loaded separator via {tse_info['label']}; "
        f"embedding via {emb_info['label']}; teacher from {args.teacher_ckpt}"
    )

    rows = []
    mix_scores = []
    full_scores = []
    full_improvements = []
    pred_dir = args.out_dir / "predictions"

    total_targets = len(metadata) * (2 if args.target_mode == "both" else 1)
    pbar = tqdm(metadata.iterrows(), total=len(metadata), desc="Evaluating Libri05_08", unit="mix", dynamic_ncols=True)
    processed_targets = 0

    for _, row in pbar:
        mix_id = row["mixture_ID"]
        sample_rate = 16000
        mixture = load_audio_mono(Path(row["mixture_path"]), sample_rate)
        source1 = load_audio_mono(Path(row["source_1_path"]), sample_rate)
        source2 = load_audio_mono(Path(row["source_2_path"]), sample_rate)

        mix_bt = mixture.to(device)
        with torch.no_grad():
            teacher_emb1 = teacher(source1.to(device))
            teacher_emb2 = teacher(source2.to(device))
            embs = emb_model(mix_bt)
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

        target_indices = [0, 1]
        if args.target_mode == "random":
            target_indices = [int(torch.randint(0, 2, (1,), generator=rng).item())]

        for target_idx in target_indices:
            teacher_target = teacher_emb1 if target_idx == 0 else teacher_emb2
            target_wav = source1 if target_idx == 0 else source2

            cos1 = cosine(e1, teacher_target)
            cos2 = cosine(e2, teacher_target)
            choose_mask = (cos1 > cos2).unsqueeze(-1)
            pred_emb = torch.where(choose_mask, e1, e2)

            enhanced = enhance_full_or_chunked(
                model_name=args.model,
                enhancer=tse_model,
                noisy_bt=mix_bt,
                emb=pred_emb,
                sr=sample_rate,
                chunk_threshold_sec=args.chunk_threshold_sec,
                chunk_sec=args.chunk_sec,
                hop_sec=args.hop_sec,
                device=device,
                use_autocast=args.autocast_fp16,
            ).squeeze(0).cpu()

            min_len = min(enhanced.shape[-1], target_wav.shape[-1], mixture.shape[-1])
            pred_eval = enhanced[:min_len]
            tgt_eval = target_wav.squeeze(0)[:min_len]
            mix_eval = mixture.squeeze(0)[:min_len]

            mix_score = si_sdr(mix_eval, tgt_eval)
            full_score = si_sdr(pred_eval, tgt_eval)
            improvement = full_score - mix_score

            mix_scores.append(mix_score)
            full_scores.append(full_score)
            full_improvements.append(improvement)

            row_name = f"{mix_id}_target{target_idx + 1}"
            if args.save_audio:
                save_wav(pred_dir / f"{row_name}_pred.wav", pred_eval, sample_rate)
                save_wav(pred_dir / f"{row_name}_target.wav", tgt_eval, sample_rate)
                save_wav(pred_dir / f"{row_name}_mix.wav", mix_eval, sample_rate)

            rows.append(
                {
                    "mixture_id": mix_id,
                    "target_index": target_idx + 1,
                    "mixture_si_sdr": mix_score,
                    "full_si_sdr": full_score,
                    "full_si_sdri": improvement,
                }
            )
            processed_targets += 1
            pbar.set_postfix(
                targets=f"{processed_targets}/{total_targets}",
                mix_si_sdr=f"{sum(mix_scores) / len(mix_scores):.3f}",
                full_si_sdr=f"{sum(full_scores) / len(full_scores):.3f}",
                si_sdri=f"{sum(full_improvements) / len(full_improvements):.3f}",
            )

    results_df = pd.DataFrame(rows)
    results_csv = args.out_dir / "per_sample_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "num_examples": len(rows),
        "target_mode": args.target_mode,
        "mean_mixture_si_sdr": float(sum(mix_scores) / max(len(mix_scores), 1)),
        "mean_full_si_sdr": float(sum(full_scores) / max(len(full_scores), 1)),
        "mean_full_si_sdri": float(sum(full_improvements) / max(len(full_improvements), 1)),
        "results_csv": str(results_csv.resolve()),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] wrote results to {results_csv}")
    print(format_summary_table(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
