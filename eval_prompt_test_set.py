import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio

from infer_recording import (
    DEFAULT_CONFIGS,
    enhance_full_or_chunked,
    load_audio_mono,
    load_embedding_model,
    load_tse_model,
)
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper


EPS = 1e-8
DEFAULT_TEACHER_CKPT = Path(
    "/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a prompt-conditioned 2-speaker test set. "
            "Embeddings are extracted from the first prompt segment only, "
            "and enhancement is run on the remaining suffix."
        )
    )
    ap.add_argument("--model", choices=["dpccn"], required=True)
    ap.add_argument("--tse-ckpt", type=Path, required=True)
    ap.add_argument("--embedding-ckpt", type=Path, default=None)
    ap.add_argument("--teacher-ckpt", type=Path, default=DEFAULT_TEACHER_CKPT)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--chunk-threshold-sec", type=float, default=20.0)
    ap.add_argument("--chunk-sec", type=float, default=8.0)
    ap.add_argument("--hop-sec", type=float, default=4.0)
    ap.add_argument("--autocast-fp16", action="store_true")
    ap.add_argument("--save-audio", action="store_true")
    return ap.parse_args()


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


def format_summary_table(summary: dict) -> str:
    headers = ["Condition", "SI-SDR", "SI-SDRi"]
    rows = [
        ["Mixture", f"{summary['mean_mixture_si_sdr']:.3f}", "-"],
        ["Full", f"{summary['mean_full_si_sdr']:.3f}", f"{summary['mean_full_si_sdri']:.3f}"],
        ["Prompt", f"{summary['mean_prompt_si_sdr']:.3f}", f"{summary['mean_prompt_si_sdri']:.3f}"],
    ]
    widths = [
        max(len(headers[col]), max(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    def fmt_row(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    return "\n".join([fmt_row(headers), sep, *(fmt_row(row) for row in rows)])


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    metadata = pd.read_csv(args.metadata_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config
    if args.model in DEFAULT_CONFIGS:
        config_path = config_path or DEFAULT_CONFIGS[args.model]

    tse_model, tse_info = load_tse_model(args.model, config_path, args.tse_ckpt, device)
    emb_model, emb_info = load_embedding_model(args.embedding_ckpt, args.tse_ckpt, args.emb_dim, device)
    teacher = load_teacher_model(args.teacher_ckpt, args.emb_dim, device)
    print(
        f"[info] loaded separator via {tse_info['label']}; "
        f"embedding via {emb_info['label']}; "
        f"teacher from {args.teacher_ckpt}"
    )

    rows = []
    aggregate = {
        "mixture_si_sdr": [],
        "prompt_best_slot_si_sdr": [],
        "prompt_improvement_si_sdr": [],
        "full_best_slot_si_sdr": [],
        "full_improvement_si_sdr": [],
    }

    pred_dir = args.out_dir / "predictions"

    for _, row in metadata.iterrows():
        sample_rate = int(row["sample_rate"])
        prompt_samples = int(row["prompt_samples"])
        target_index = int(row["target_index"])
        mix_id = row["mixture_id"]

        mixture = load_audio_mono(Path(row["mixture_path"]), sample_rate)
        source1 = load_audio_mono(Path(row["source_1_path"]), sample_rate)
        source2 = load_audio_mono(Path(row["source_2_path"]), sample_rate)

        mix_prompt = mixture[:, :prompt_samples].to(device)
        mix_eval = mixture[:, prompt_samples:].to(device)
        mix_full = mixture.to(device)
        target_ref = source1[:, prompt_samples:] if target_index == 1 else source2[:, prompt_samples:]
        baseline = mixture[:, prompt_samples:]

        with torch.no_grad():
            prompt_embeddings = emb_model(mix_prompt)
            full_embeddings = emb_model(mix_full)
            teacher_target = teacher(
                (source1 if target_index == 1 else source2).to(device)
            )

        prompt_e1 = prompt_embeddings[:, 0, :]
        prompt_e2 = prompt_embeddings[:, 1, :]
        prompt_c1 = cosine(prompt_e1, teacher_target)
        prompt_c2 = cosine(prompt_e2, teacher_target)
        prompt_choose = (prompt_c1 > prompt_c2).unsqueeze(-1)
        prompt_emb = torch.where(prompt_choose, prompt_e1, prompt_e2)

        prompt_pred = enhance_full_or_chunked(
            model_name=args.model,
            enhancer=tse_model,
            noisy_bt=mix_eval,
            emb=prompt_emb,
            sr=sample_rate,
            chunk_threshold_sec=args.chunk_threshold_sec,
            chunk_sec=args.chunk_sec,
            hop_sec=args.hop_sec,
            device=device,
            use_autocast=args.autocast_fp16,
        ).squeeze(0).cpu()
        prompt_score = si_sdr(prompt_pred, target_ref.squeeze(0))

        full_e1 = full_embeddings[:, 0, :]
        full_e2 = full_embeddings[:, 1, :]
        full_c1 = cosine(full_e1, teacher_target)
        full_c2 = cosine(full_e2, teacher_target)
        full_choose = (full_c1 > full_c2).unsqueeze(-1)
        full_emb = torch.where(full_choose, full_e1, full_e2)

        full_pred = enhance_full_or_chunked(
            model_name=args.model,
            enhancer=tse_model,
            noisy_bt=mix_full,
            emb=full_emb,
            sr=sample_rate,
            chunk_threshold_sec=args.chunk_threshold_sec,
            chunk_sec=args.chunk_sec,
            hop_sec=args.hop_sec,
            device=device,
            use_autocast=args.autocast_fp16,
        ).squeeze(0).cpu()
        full_score = si_sdr(full_pred[prompt_samples:], target_ref.squeeze(0))

        mixture_score = si_sdr(baseline.squeeze(0), target_ref.squeeze(0))
        prompt_improvement = prompt_score - mixture_score
        full_improvement = full_score - mixture_score

        aggregate["mixture_si_sdr"].append(mixture_score)
        aggregate["prompt_best_slot_si_sdr"].append(prompt_score)
        aggregate["prompt_improvement_si_sdr"].append(prompt_improvement)
        aggregate["full_best_slot_si_sdr"].append(full_score)
        aggregate["full_improvement_si_sdr"].append(full_improvement)

        if args.save_audio:
            save_wav(pred_dir / f"{mix_id}_prompt_pred.wav", prompt_pred, sample_rate)
            save_wav(pred_dir / f"{mix_id}_full_pred.wav", full_pred, sample_rate)
            save_wav(pred_dir / f"{mix_id}_prompt_target.wav", target_ref.squeeze(0), sample_rate)
            save_wav(pred_dir / f"{mix_id}_prompt_baseline.wav", baseline.squeeze(0), sample_rate)
            save_wav(pred_dir / f"{mix_id}_full_target.wav", target_ref.squeeze(0), sample_rate)
            save_wav(pred_dir / f"{mix_id}_full_baseline.wav", baseline.squeeze(0), sample_rate)

        rows.append(
            {
                "mixture_id": mix_id,
                "target_index": target_index,
                "noisy": bool(row["noisy"]),
                "mixture_si_sdr": mixture_score,
                "prompt_cos1": float(prompt_c1.item()),
                "prompt_cos2": float(prompt_c2.item()),
                "prompt_selected_slot": 1 if bool(prompt_choose.item()) else 2,
                "prompt_si_sdr": prompt_score,
                "prompt_si_sdri": prompt_improvement,
                "full_cos1": float(full_c1.item()),
                "full_cos2": float(full_c2.item()),
                "full_selected_slot": 1 if bool(full_choose.item()) else 2,
                "full_si_sdr": full_score,
                "full_si_sdri": full_improvement,
            }
        )

    results_df = pd.DataFrame(rows)
    results_csv = args.out_dir / "per_sample_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "num_examples": len(rows),
        "mean_mixture_si_sdr": float(sum(aggregate["mixture_si_sdr"]) / max(len(rows), 1)),
        "mean_full_si_sdr": float(sum(aggregate["full_best_slot_si_sdr"]) / max(len(rows), 1)),
        "mean_full_si_sdri": float(sum(aggregate["full_improvement_si_sdr"]) / max(len(rows), 1)),
        "mean_prompt_si_sdr": float(sum(aggregate["prompt_best_slot_si_sdr"]) / max(len(rows), 1)),
        "mean_prompt_si_sdri": float(sum(aggregate["prompt_improvement_si_sdr"]) / max(len(rows), 1)),
        "results_csv": str(results_csv.resolve()),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] wrote results to {results_csv}")
    print(format_summary_table(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
