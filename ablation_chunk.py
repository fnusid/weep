import math
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
import tqdm

from torch.utils.data import DataLoader
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from omegaconf import OmegaConf

from models.dpcnn import DPCCN
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper
from dataset.dataloader import LibriMixDataModule


# -------------------------
# Paths
# -------------------------
TSE_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt"
EMB_CKPT = "/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
JOINT_TRAINED_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"

CONFIG_PATH = Path("/home/sidcs/codebase/wesep/confs/config_dpcnn.yaml")
OUT_CSV = "/home/sidcs/codebase/wesep/analysis/chunked_tradeoff_sisdr_causal_untrained.csv"


# -------------------------
# Config
# -------------------------
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]
hp = OmegaConf.merge(*docs)


# -------------------------
# Loading helpers
# -------------------------
def load_dpccn_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    dpccn_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            dpccn_sd[k[len("model."):]] = v

    print("DPCCN keys:", len(dpccn_sd))
    model.load_state_dict(dpccn_sd, strict=True)
    return model


def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue

        k2 = k[len("model."):]

        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue

        new_state[k2] = v

    return new_state


def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("dual_emb_model."):
            new_state[k[len("dual_emb_model."):]] = v
    return new_state


def load_dual(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    sd = strip_dual_model_weights(sd)
    model.load_state_dict(sd, strict=True)

    joint_sd = torch.load(JOINT_TRAINED_CKPT, map_location="cpu")["state_dict"]
    joint_sd = joint_trained_model_weights(joint_sd)
    model.load_state_dict(joint_sd, strict=True)

    model.eval()
    return model


# -------------------------
# Metric helpers
# -------------------------
def make_2d(x):
    """
    Ensure [B, T].
    """
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    return x


def make_src(x):
    """
    Ensure [B, 2, T].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected src [B, 2, T], got {x.shape}")
    return x


@torch.no_grad()
def sisdr_value(metric, pred, target):
    """
    pred:   [T]
    target: [T]
    """
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[:min_len].unsqueeze(0)
    target = target[:min_len].unsqueeze(0)
    return metric(pred, target).item()


@torch.no_grad()
def pit_sisdr_2sp(metric, pred_2sp, target_2sp):
    """
    pred_2sp:   [2, T]
    target_2sp: [2, T]

    Returns best average SI-SDR over 2-speaker permutation.
    """
    s00 = sisdr_value(metric, pred_2sp[0], target_2sp[0])
    s11 = sisdr_value(metric, pred_2sp[1], target_2sp[1])
    score_a = 0.5 * (s00 + s11)

    s01 = sisdr_value(metric, pred_2sp[0], target_2sp[1])
    s10 = sisdr_value(metric, pred_2sp[1], target_2sp[0])
    score_b = 0.5 * (s01 + s10)

    return max(score_a, score_b)


@torch.no_grad()
def input_sisdr_2sp(metric, mix, target_2sp):
    """
    SI-SDR of mixture against sources, PIT style.
    mix:       [T]
    target_2sp:[2, T]
    """
    s0 = sisdr_value(metric, mix, target_2sp[0])
    s1 = sisdr_value(metric, mix, target_2sp[1])
    return 0.5 * (s0 + s1)


# -------------------------
# Inference wrapper
# -------------------------
class ChunkedInferenceWrapper(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        hp.model_args.tse_model.causal = True  # ensure causal for chunked inference
        self.tse_model = DPCCN(**hp.model_args.tse_model)
        self.emb_model = SpeakerEncoderDualWrapper(emb_dim=256)

        self.tse_model = load_dpccn_ckpt(self.tse_model, TSE_CKPT)
        self.emb_model = load_dual(self.emb_model, EMB_CKPT)

        self.tse_model.to(self.device).eval()
        self.emb_model.to(self.device).eval()

        self.embedding1 = torch.zeros(1, 256, device=self.device)
        self.embedding2 = torch.zeros(1, 256, device=self.device)

    @torch.no_grad()
    def set_initial_embeddings(self, audio, emb_context_sec, sample_rate=16000):
        """
        audio: [1, T]
        """
        emb_len = int(emb_context_sec * sample_rate)
        emb_chunk = audio[:, :min(audio.shape[-1], emb_len)]

        embs = self.emb_model(emb_chunk)
        e1 = embs[0, 0].unsqueeze(0)
        e2 = embs[0, 1].unsqueeze(0)

        self.embedding1 = e1
        self.embedding2 = e2

    @torch.no_grad()
    def update_embeddings(self, e1, e2, alpha=0.1):
        sim_new = F.cosine_similarity(e1, e2).item()
        if sim_new >= 0.75:
            return

        c11 = F.cosine_similarity(e1, self.embedding1).item()
        c12 = F.cosine_similarity(e1, self.embedding2).item()
        c21 = F.cosine_similarity(e2, self.embedding1).item()
        c22 = F.cosine_similarity(e2, self.embedding2).item()

        if c11 > c12 and c22 > c21:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e1
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e2

        elif c12 > c11 and c21 > c22:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e2
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e1

    @torch.no_grad()
    def update_embeddings_from_context(self, audio, end_sample, emb_context_sec, sample_rate=16000):
        """
        Update using recent rolling context ending at end_sample.
        """
        ctx_len = int(emb_context_sec * sample_rate)
        start = max(0, end_sample - ctx_len)
        ctx = audio[:, start:end_sample]

        if ctx.shape[-1] < sample_rate:
            return

        embs = self.emb_model(ctx)
        e1 = embs[0, 0].unsqueeze(0)
        e2 = embs[0, 1].unsqueeze(0)

        self.update_embeddings(e1, e2)

    @torch.no_grad()
    def out_computer(self, chunk):
        out1, _ = self.tse_model(chunk, self.embedding1)
        out2, _ = self.tse_model(chunk, self.embedding2)
        return torch.cat([out1, out2], dim=0)

    @torch.no_grad()
    def chunked_inference(
        self,
        audio,
        emb_context_sec,
        chunk_ms,
        hop_ms,
        emb_update_interval_sec,
        sample_rate=16000,
        adaptive_update=True,
    ):
        """
        audio: [1, T]
        returns: [2, T]
        """
        audio = audio.to(self.device)

        self.set_initial_embeddings(audio, emb_context_sec, sample_rate)

        chunk_size = int(sample_rate * chunk_ms / 1000)
        hop_size = int(sample_rate * hop_ms / 1000)
        update_interval = int(sample_rate * emb_update_interval_sec)

        if audio.shape[-1] < chunk_size:
            audio = F.pad(audio, (0, chunk_size - audio.shape[-1]))

        num_chunks = math.ceil((audio.shape[-1] - chunk_size) / hop_size) + 1
        total_len = (num_chunks - 1) * hop_size + chunk_size

        if total_len > audio.shape[-1]:
            audio = F.pad(audio, (0, total_len - audio.shape[-1]))

        output = torch.zeros((2, total_len), dtype=audio.dtype, device=self.device)
        weight = torch.zeros((total_len,), dtype=audio.dtype, device=self.device)
        window = torch.hann_window(chunk_size, periodic=True, device=self.device, dtype=audio.dtype)

        last_update_sample = 0

        for i in range(num_chunks):
            start = i * hop_size
            end = start + chunk_size

            if adaptive_update and end - last_update_sample >= update_interval:
                self.update_embeddings_from_context(
                    audio=audio,
                    end_sample=end,
                    emb_context_sec=emb_context_sec,
                    sample_rate=sample_rate,
                )
                last_update_sample = end

            chunk = audio[:, start:end]
            chunk_out = self.out_computer(chunk)

            output[:, start:end] += chunk_out[:, :chunk_size] * window
            weight[start:end] += window

        final_output = output / weight.clamp_min(1e-8).unsqueeze(0)

        return final_output[:, :audio.shape[-1]]


# -------------------------
# Sweep evaluation
# -------------------------
@torch.no_grad()
def evaluate_sweep(
    wrapper,
    val_loader,
    emb_context_secs,
    chunk_sizes_ms,
    emb_update_intervals_sec,
    sample_rate=16000,
    max_batches=None,
    device="cuda",
):
    metric = ScaleInvariantSignalDistortionRatio().to(device)

    results = []

    for emb_sec in emb_context_secs:
        for chunk_ms in chunk_sizes_ms:
            hop_ms = chunk_ms // 2

            for update_sec in emb_update_intervals_sec:
                sisdr_scores = []
                input_sisdr_scores = []

                pbar = tqdm.tqdm(
                    enumerate(val_loader),
                    total=len(val_loader) if max_batches is None else min(len(val_loader), max_batches),
                    desc=f"emb={emb_sec}s chunk={chunk_ms}ms hop={hop_ms}ms upd={update_sec}s",
                    leave=True,
                )

                for batch_idx, batch in pbar:
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    mix, src, labels = batch

                    mix = make_2d(mix).float()
                    src = make_src(src).float()

                    B = mix.shape[0]

                    for b in range(B):
                        mix_b = mix[b:b+1].to(device)          # [1, T]
                        src_b = src[b].to(device)              # [2, T]

                        pred = wrapper.chunked_inference(
                            audio=mix_b,
                            emb_context_sec=emb_sec,
                            chunk_ms=chunk_ms,
                            hop_ms=hop_ms,
                            emb_update_interval_sec=update_sec,
                            sample_rate=sample_rate,
                            adaptive_update=True,
                        )

                        min_len = min(pred.shape[-1], src_b.shape[-1], mix_b.shape[-1])
                        pred = pred[:, :min_len]
                        src_eval = src_b[:, :min_len]
                        mix_eval = mix_b[0, :min_len]

                        sisdr = pit_sisdr_2sp(metric, pred, src_eval)
                        mix_sisdr = input_sisdr_2sp(metric, mix_eval, src_eval)

                        sisdr_scores.append(sisdr)
                        input_sisdr_scores.append(mix_sisdr)

                    mean_sisdr = sum(sisdr_scores) / max(1, len(sisdr_scores))
                    mean_mix = sum(input_sisdr_scores) / max(1, len(input_sisdr_scores))
                    pbar.set_postfix({
                        "SI-SDR": f"{mean_sisdr:.3f}",
                        "SI-SDRi": f"{mean_sisdr - mean_mix:.3f}",
                    })

                mean_sisdr = sum(sisdr_scores) / len(sisdr_scores)
                mean_mix_sisdr = sum(input_sisdr_scores) / len(input_sisdr_scores)
                mean_sisdri = mean_sisdr - mean_mix_sisdr

                row = {
                    "emb_context_sec": emb_sec,
                    "chunk_ms": chunk_ms,
                    "hop_ms": hop_ms,
                    "emb_update_interval_sec": update_sec,
                    "SI_SDR": mean_sisdr,
                    "Mix_SI_SDR": mean_mix_sisdr,
                    "SI_SDRi": mean_sisdri,
                    "num_examples": len(sisdr_scores),
                }

                print(row)
                results.append(row)

                with open(OUT_CSV, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)

    return results


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "/home/sidcs/datasets/LibriMix/LibriMix"
    speaker_map_path = (
        "/home/sidcs/datasets/LibriMix/LibriMix/"
        "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    )

    batch_size = 4
    num_workers = 8
    num_speakers = 2
    sample_rate = 16000

    dm = LibriMixDataModule(
        data_root=data_root,
        speaker_map_path=speaker_map_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
    )

    dm.setup()
    val_loader: DataLoader = dm.test_dataloader()

    wrapper = ChunkedInferenceWrapper(device=device)

    emb_context_secs = [5, 10, 20, 30]
    chunk_sizes_ms = [2000, 5000, 10000, 20000]
    emb_update_intervals_sec = [2, 5, 10]

    results = evaluate_sweep(
        wrapper=wrapper,
        val_loader=val_loader,
        emb_context_secs=emb_context_secs,
        chunk_sizes_ms=chunk_sizes_ms,
        emb_update_intervals_sec=emb_update_intervals_sec,
        sample_rate=sample_rate,
        max_batches=10,   # set None for full test set
        device=device,
    )

    print(f"Saved results to: {OUT_CSV}")