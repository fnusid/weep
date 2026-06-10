import argparse
import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
CODEBASE_ROOT = REPO_ROOT.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dpcnn import DPCCN
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper


DEFAULT_CONFIGS = {
    "dpccn": REPO_ROOT / "confs" / "config_dpcnn.yaml",
}

CAUSAL_GRIDNET_ARGS = {
    "spk_emb_dim": 256,
    "stft_chunk_size": 128,
    "stft_pad_size": 128,
    "stft_back_pad": 128,
    "num_ch": 1,
    "D": 64,
    "L": 4,
    "I": 1,
    "J": 1,
    "B": 3,
    "H": 64,
    "local_atten_len": 50,
    "use_attn": True,
    "masked_attn": True,
    "chunk_causal": True,
    "spectral_masking": True,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run two-speaker TSE inference on one recording using a wesep "
            "enhancer plus a dual-embedding model."
        )
    )
    ap.add_argument("--model", choices=["dpccn", "gridnet"], required=True)
    ap.add_argument("--tse-ckpt", type=Path, required=True)
    ap.add_argument(
        "--embedding-ckpt",
        type=Path,
        default=None,
        help=(
            "Checkpoint for the dual embedding model. If omitted, the script "
            "tries to load `dual_emb_model.*` weights from --tse-ckpt."
        ),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional architecture config. For DPCCN, this should be a YAML with "
            "`model_args.tse_model`. For causal GridNet, JSON/YAML kwargs may be "
            "provided; otherwise built-in causal defaults are used."
        ),
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g. cuda, cuda:0, or cpu.",
    )
    ap.add_argument(
        "--speaker",
        choices=["1", "2", "both"],
        default="both",
        help="Which separated target(s) to save.",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix for saved outputs.",
    )
    ap.add_argument(
        "--chunk-threshold-sec",
        type=float,
        default=20.0,
        help="Use overlap-add inference when the recording exceeds this length.",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=8.0,
        help="Chunk size for overlap-add inference.",
    )
    ap.add_argument(
        "--hop-sec",
        type=float,
        default=4.0,
        help="Hop size for overlap-add inference.",
    )
    ap.add_argument(
        "--autocast-fp16",
        action="store_true",
        help="Enable fp16 autocast on CUDA for the enhancer forward pass.",
    )
    ap.add_argument(
        "--gridnet-local-atten-len",
        type=int,
        default=None,
        help=(
            "Override causal GridNet local attention length at inference time. "
            "Only used when --model gridnet."
        ),
    )
    return ap.parse_args()


def load_yaml_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]
    return OmegaConf.merge(*docs)


def load_generic_mapping(config_path: Path):
    if config_path.suffix.lower() == ".json":
        import json

        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.clamp(-1.0, 1.0)


def maybe_autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def get_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    return ckpt_obj


def strip_prefix(state_dict, prefix: str):
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def filter_dual_embedding_from_standalone(state_dict):
    filtered = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        new_key = key[len("model."):]
        if new_key.startswith("single_sp_model.") or new_key.startswith("arcface_loss."):
            continue
        filtered[new_key] = value
    return filtered


def pick_best_candidate(model, candidates, strict_preference=True):
    best = None
    errors = []

    for label, candidate_state in candidates:
        if not candidate_state:
            continue
        try:
            missing, unexpected = model.load_state_dict(candidate_state, strict=False)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue

        loaded = len(candidate_state) - len(unexpected)
        score = (loaded, -len(missing), -len(unexpected))
        if best is None or score > best["score"]:
            best = {
                "label": label,
                "state": candidate_state,
                "score": score,
                "missing": list(missing),
                "unexpected": list(unexpected),
            }

    if best is None or best["score"][0] <= 0:
        details = "\n".join(errors) if errors else "No candidate state dict matched."
        raise RuntimeError(details)

    strict = strict_preference and not best["missing"] and not best["unexpected"]
    model.load_state_dict(best["state"], strict=strict)
    return best


def build_tse_model(model_name: str, config_path: Path, overrides: dict | None = None):
    if model_name == "dpccn":
        hp = load_yaml_config(config_path)
        model_args = hp.model_args.tse_model
        return DPCCN(**model_args)
    if model_name == "gridnet":
        from models.gridnet_causal_net import Net as CausalGridNet

        if config_path is None:
            model_args = dict(CAUSAL_GRIDNET_ARGS)
        else:
            loaded = load_generic_mapping(config_path)
            if loaded is None:
                model_args = dict(CAUSAL_GRIDNET_ARGS)
            elif "model_args" in loaded and "tse_model" in loaded["model_args"]:
                model_args = dict(loaded["model_args"]["tse_model"])
            else:
                model_args = dict(loaded)
        if overrides:
            model_args.update(overrides)
        return CausalGridNet(**model_args)
    raise ValueError(f"Unsupported model: {model_name}")


def load_tse_model(
    model_name: str,
    config_path: Path,
    ckpt_path: Path,
    device: torch.device,
    overrides: dict | None = None,
):
    model = build_tse_model(model_name, config_path, overrides=overrides)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = get_state_dict(ckpt)

    candidates = [
        ("raw", state_dict),
        ("strip:model.", strip_prefix(state_dict, "model.")),
        ("strip:model.model.", strip_prefix(state_dict, "model.model.")),
        ("strip:tse_model.", strip_prefix(state_dict, "tse_model.")),
    ]
    chosen = pick_best_candidate(model, candidates)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, chosen


def run_separator(model_name: str, enhancer, noisy_bt: torch.Tensor, emb: torch.Tensor):
    if model_name == "gridnet":
        mixture_bt = noisy_bt.unsqueeze(1).contiguous()
        emb = emb.contiguous()
        outputs = enhancer({"mixture": mixture_bt, "embedding": emb}, input_state=None, pad=True)
        estimate = outputs["output"]
        if estimate.ndim == 3 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
        return estimate
    return unwrap_enhancer_output(enhancer(noisy_bt, emb))


def load_embedding_model(embedding_ckpt: Path | None, tse_ckpt: Path, emb_dim: int, device: torch.device):
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    candidates = []

    if embedding_ckpt is not None:
        emb_ckpt = torch.load(str(embedding_ckpt), map_location="cpu")
        emb_state = get_state_dict(emb_ckpt)
        candidates.extend(
            [
                ("embedding:raw", emb_state),
                ("embedding:strip:model.", strip_prefix(emb_state, "model.")),
                ("embedding:standalone-filter", filter_dual_embedding_from_standalone(emb_state)),
                ("embedding:strip:dual_emb_model.", strip_prefix(emb_state, "dual_emb_model.")),
            ]
        )

    tse_loaded = torch.load(str(tse_ckpt), map_location="cpu")
    tse_state = get_state_dict(tse_loaded)
    candidates.extend(
        [
            ("tse:strip:dual_emb_model.", strip_prefix(tse_state, "dual_emb_model.")),
            ("tse:strip:model.dual_emb_model.", strip_prefix(tse_state, "model.dual_emb_model.")),
        ]
    )

    chosen = pick_best_candidate(model, candidates)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, chosen


def unwrap_enhancer_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def enhance_chunked_ola(
    model_name: str,
    enhancer,
    noisy_bt: torch.Tensor,
    emb: torch.Tensor,
    sr: int,
    chunk_sec: float,
    hop_sec: float,
    device: torch.device,
    use_autocast: bool,
):
    total = noisy_bt.shape[-1]
    chunk_len = int(sr * chunk_sec)
    hop_len = int(sr * hop_sec)
    if chunk_len <= 0 or hop_len <= 0 or hop_len > chunk_len:
        raise ValueError("Invalid chunk_sec / hop_sec combination.")

    window = torch.hann_window(chunk_len, periodic=True, device=device).unsqueeze(0)
    out_acc = torch.zeros((1, total + chunk_len), device=device)
    weight_acc = torch.zeros((1, total + chunk_len), device=device)
    eps = 1e-8

    for start in range(0, total, hop_len):
        end = min(start + chunk_len, total)
        chunk = noisy_bt[..., start:end]
        if chunk.shape[-1] < chunk_len:
            chunk = F.pad(chunk, (0, chunk_len - chunk.shape[-1]))

        with maybe_autocast(use_autocast, device):
            enhanced = run_separator(model_name, enhancer, chunk, emb)

        if enhanced.shape[-1] > chunk_len:
            enhanced = enhanced[..., :chunk_len]
        elif enhanced.shape[-1] < chunk_len:
            enhanced = F.pad(enhanced, (0, chunk_len - enhanced.shape[-1]))

        out_acc[..., start:start + chunk_len] += enhanced * window
        weight_acc[..., start:start + chunk_len] += window

        if end == total:
            break

    return (out_acc[..., :total] / (weight_acc[..., :total] + eps)).detach().cpu().float()


@torch.no_grad()
def enhance_full_or_chunked(
    model_name: str,
    enhancer,
    noisy_bt: torch.Tensor,
    emb: torch.Tensor,
    sr: int,
    chunk_threshold_sec: float,
    chunk_sec: float,
    hop_sec: float,
    device: torch.device,
    use_autocast: bool,
):
    threshold = int(sr * chunk_threshold_sec)
    total = noisy_bt.shape[-1]
    if total > threshold:
        return enhance_chunked_ola(
            model_name=model_name,
            enhancer=enhancer,
            noisy_bt=noisy_bt,
            emb=emb,
            sr=sr,
            chunk_sec=chunk_sec,
            hop_sec=hop_sec,
            device=device,
            use_autocast=use_autocast,
        )

    try:
        with maybe_autocast(use_autocast, device):
            enhanced = run_separator(model_name, enhancer, noisy_bt, emb)
        return enhanced[..., :total].detach().cpu().float()
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return enhance_chunked_ola(
            model_name=model_name,
            enhancer=enhancer,
            noisy_bt=noisy_bt,
            emb=emb,
            sr=sr,
            chunk_sec=chunk_sec,
            hop_sec=hop_sec,
            device=device,
            use_autocast=use_autocast,
        )


def save_outputs(outputs, output_dir: Path, base_name: str, sample_rate: int, selection: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_map = {"1": [0], "2": [1], "both": [0, 1]}[selection]
    saved = []
    for idx in save_map:
        out_path = output_dir / f"{base_name}_spk{idx + 1}.wav"
        torchaudio.save(str(out_path), outputs[idx].unsqueeze(0), sample_rate=sample_rate)
        saved.append(out_path)
    return saved


def main():
    args = parse_args()
    config_path = args.config
    if args.model in DEFAULT_CONFIGS:
        config_path = config_path or DEFAULT_CONFIGS[args.model]

    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    gridnet_overrides = {}
    if args.model == "gridnet" and args.gridnet_local_atten_len is not None:
        gridnet_overrides["local_atten_len"] = args.gridnet_local_atten_len

    print(f"[info] loading config: {config_path}")
    tse_model, tse_load = load_tse_model(
        args.model,
        config_path,
        args.tse_ckpt,
        device,
        overrides=gridnet_overrides or None,
    )
    print(
        f"[info] loaded {args.model} checkpoint via {tse_load['label']} "
        f"(missing={len(tse_load['missing'])}, unexpected={len(tse_load['unexpected'])})"
    )
    if gridnet_overrides:
        print(f"[info] applied gridnet overrides: {gridnet_overrides}")

    emb_model, emb_load = load_embedding_model(args.embedding_ckpt, args.tse_ckpt, args.emb_dim, device)
    print(
        f"[info] loaded embedding checkpoint via {emb_load['label']} "
        f"(missing={len(emb_load['missing'])}, unexpected={len(emb_load['unexpected'])})"
    )

    wav = load_audio_mono(args.input, args.sr)
    wav_bt = wav.to(device)
    print(f"[info] input samples={wav_bt.shape[-1]} sr={args.sr}")

    with torch.no_grad():
        embeddings = emb_model(wav_bt)

    outputs = []
    for spk_idx in range(2):
        emb = embeddings[:, spk_idx, :]
        enhanced = enhance_full_or_chunked(
            model_name=args.model,
            enhancer=tse_model,
            noisy_bt=wav_bt,
            emb=emb,
            sr=args.sr,
            chunk_threshold_sec=args.chunk_threshold_sec,
            chunk_sec=args.chunk_sec,
            hop_sec=args.hop_sec,
            device=device,
            use_autocast=args.autocast_fp16,
        )
        outputs.append(enhanced.squeeze(0))

    stem = args.input.stem
    prefix = args.prefix or f"{stem}_{args.model}"
    saved = save_outputs(outputs, args.output_dir, prefix, args.sr, args.speaker)
    for path in saved:
        print(f"[info] saved {path}")


if __name__ == "__main__":
    main()
