import queue
import time
import math
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from models.dpcnn import DPCCN
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper


# =========================
# Paths
# =========================
TSE_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt"
EMB_CKPT = "/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
JOINT_TRAINED_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"

CONFIG_PATH = Path("/home/sidcs/codebase/wesep/confs/config_dpcnn.yaml")


# =========================
# Config
# =========================
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]
hp = OmegaConf.merge(*docs)


# =========================
# Checkpoint loading
# =========================
def load_dpccn_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    dpccn_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            dpccn_sd[k[len("model."):]] = v

    print("[load] DPCCN keys:", len(dpccn_sd))
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

    return model


# =========================
# Real-time demo
# =========================
class RealTimeTSEDemo(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        chunk_sec=5.0,
        hop_sec=2.5,
        emb_context_sec=10.0,
        emb_update_interval_sec=10.0,
        block_sec=0.1,
        device=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.hop_sec = hop_sec
        self.emb_context_sec = emb_context_sec
        self.emb_update_interval_sec = emb_update_interval_sec
        self.block_sec = block_sec

        self.chunk_size = int(sample_rate * chunk_sec)
        self.hop_size = int(sample_rate * hop_sec)
        self.emb_context_size = int(sample_rate * emb_context_sec)
        self.emb_update_interval = int(sample_rate * emb_update_interval_sec)
        self.block_size = int(sample_rate * block_sec)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print("[init] device:", self.device)
        print("[init] chunk:", self.chunk_sec, "sec")
        print("[init] hop:", self.hop_sec, "sec")
        print("[init] embedding context:", self.emb_context_sec, "sec")

        self.tse_model = DPCCN(**hp.model_args.tse_model)
        self.emb_model = SpeakerEncoderDualWrapper(emb_dim=256)

        self.tse_model = load_dpccn_ckpt(self.tse_model, TSE_CKPT)
        self.emb_model = load_dual(self.emb_model, EMB_CKPT)

        self.tse_model.to(self.device).eval()
        self.emb_model.to(self.device).eval()

        self.embedding1 = torch.zeros(1, 256, device=self.device)
        self.embedding2 = torch.zeros(1, 256, device=self.device)

        self.selected_slot = 1

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_seen = 0

        self.next_process_end = self.chunk_size
        self.last_emb_update_sample = 0

        self.ola_base = 0
        self.ola_audio = np.zeros(0, dtype=np.float32)
        self.ola_weight = np.zeros(0, dtype=np.float32)
        self.emit_cursor = 0

        self.window = np.hanning(self.chunk_size).astype(np.float32)

    # -------------------------
    # Audio callback
    # -------------------------
    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print("[audio status]", status)

        mic_block = indata[:, 0].copy().astype(np.float32)
        self.input_queue.put(mic_block)

        try:
            out_block = self.output_queue.get_nowait()
        except queue.Empty:
            out_block = np.zeros(frames, dtype=np.float32)

        if len(out_block) < frames:
            padded = np.zeros(frames, dtype=np.float32)
            padded[:len(out_block)] = out_block
            out_block = padded

        outdata[:, 0] = out_block[:frames]

    # -------------------------
    # Embeddings
    # -------------------------
    @torch.no_grad()
    def compute_dual_embeddings(self, audio_np):
        wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        embs = self.emb_model(wav)

        e1 = embs[0, 0].unsqueeze(0)
        e2 = embs[0, 1].unsqueeze(0)

        return e1, e2

    @torch.no_grad()
    def initialize_embeddings(self):
        ctx = self.audio_buffer[-self.emb_context_size:]
        e1, e2 = self.compute_dual_embeddings(ctx)

        self.embedding1 = e1
        self.embedding2 = e2

        sim = F.cosine_similarity(e1, e2).item()
        print(f"[emb] initialized. cos(e1,e2)={sim:.3f}")

    @torch.no_grad()
    def update_embeddings(self):
        if len(self.audio_buffer) < self.emb_context_size:
            return

        ctx = self.audio_buffer[-self.emb_context_size:]
        e1, e2 = self.compute_dual_embeddings(ctx)

        sim_new = F.cosine_similarity(e1, e2).item()
        if sim_new >= 0.75:
            print(f"[emb] skipped update, slots too similar: {sim_new:.3f}")
            return

        c11 = F.cosine_similarity(e1, self.embedding1).item()
        c12 = F.cosine_similarity(e1, self.embedding2).item()
        c21 = F.cosine_similarity(e2, self.embedding1).item()
        c22 = F.cosine_similarity(e2, self.embedding2).item()

        alpha = 0.1

        if c11 > c12 and c22 > c21:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e1
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e2
            print(f"[emb] update same order | c11={c11:.3f}, c22={c22:.3f}")

        elif c12 > c11 and c21 > c22:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e2
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e1
            print(f"[emb] update swapped | c12={c12:.3f}, c21={c21:.3f}")

        else:
            print("[emb] skipped ambiguous assignment")

    # -------------------------
    # TSE
    # -------------------------
    @torch.no_grad()
    def enhance_chunk(self, chunk_np):
        wav = torch.from_numpy(chunk_np).float().unsqueeze(0).to(self.device)

        emb = self.embedding1 if self.selected_slot == 1 else self.embedding2

        out, _ = self.tse_model(wav, emb)
        out = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return out

    # -------------------------
    # OLA output
    # -------------------------
    def ensure_ola_capacity(self, abs_end):
        needed = abs_end - self.ola_base
        if needed <= len(self.ola_audio):
            return

        extra = needed - len(self.ola_audio)
        self.ola_audio = np.pad(self.ola_audio, (0, extra))
        self.ola_weight = np.pad(self.ola_weight, (0, extra))

    def add_chunk_to_ola(self, chunk_start_abs, enhanced_np):
        chunk_end_abs = chunk_start_abs + self.chunk_size
        self.ensure_ola_capacity(chunk_end_abs)

        rel_start = chunk_start_abs - self.ola_base
        rel_end = rel_start + self.chunk_size

        self.ola_audio[rel_start:rel_end] += enhanced_np * self.window
        self.ola_weight[rel_start:rel_end] += self.window

    def emit_available_audio(self, emit_until_abs):
        if emit_until_abs <= self.emit_cursor:
            return

        rel_start = self.emit_cursor - self.ola_base
        rel_end = emit_until_abs - self.ola_base

        if rel_end > len(self.ola_audio):
            return

        audio = self.ola_audio[rel_start:rel_end]
        weight = self.ola_weight[rel_start:rel_end]

        emitted = audio / np.maximum(weight, 1e-8)
        emitted = np.clip(emitted, -1.0, 1.0).astype(np.float32)

        for i in range(0, len(emitted), self.block_size):
            self.output_queue.put(emitted[i:i + self.block_size])

        self.emit_cursor = emit_until_abs

        trim = self.emit_cursor - self.ola_base
        if trim > 0:
            self.ola_audio = self.ola_audio[trim:]
            self.ola_weight = self.ola_weight[trim:]
            self.ola_base = self.emit_cursor

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        print("\nStarting mic stream.")
        print("Initial embedding collection will take", self.emb_context_sec, "seconds.")
        print("Use headphones to avoid feedback.\n")

        stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        )

        initialized = False
        last_print = time.time()

        with stream:
            while True:
                block = self.input_queue.get()

                self.audio_buffer = np.concatenate([self.audio_buffer, block])
                self.total_samples_seen += len(block)

                # Keep buffer bounded
                max_keep = max(self.emb_context_size, self.chunk_size * 3)
                if len(self.audio_buffer) > max_keep:
                    self.audio_buffer = self.audio_buffer[-max_keep:]

                # Initialize embeddings
                if not initialized and self.total_samples_seen >= self.emb_context_size:
                    self.initialize_embeddings()

                    print("\nChoose target speaker slot:")
                    print("1 = embedding slot 1")
                    print("2 = embedding slot 2")
                    choice = input("Enter 1 or 2: ").strip()

                    if choice == "2":
                        self.selected_slot = 2
                    else:
                        self.selected_slot = 1

                    print(f"[select] using speaker slot {self.selected_slot}")
                    initialized = True

                    self.next_process_end = self.total_samples_seen

                if not initialized:
                    now = time.time()
                    if now - last_print > 1.0:
                        remain = max(0, self.emb_context_size - self.total_samples_seen) / self.sample_rate
                        print(f"[warmup] {remain:.1f}s remaining")
                        last_print = now
                    continue

                # Process every hop
                while self.total_samples_seen >= self.next_process_end:
                    chunk_end_abs = self.next_process_end
                    chunk_start_abs = chunk_end_abs - self.chunk_size

                    if chunk_start_abs < 0:
                        self.next_process_end += self.hop_size
                        continue

                    # Map absolute indices to local rolling buffer
                    local_end = len(self.audio_buffer) - (self.total_samples_seen - chunk_end_abs)
                    local_start = local_end - self.chunk_size

                    if local_start < 0 or local_end > len(self.audio_buffer):
                        break

                    chunk = self.audio_buffer[local_start:local_end]

                    # update embeddings every N seconds
                    if chunk_end_abs - self.last_emb_update_sample >= self.emb_update_interval:
                        self.update_embeddings()
                        self.last_emb_update_sample = chunk_end_abs

                    enhanced = self.enhance_chunk(chunk)

                    self.add_chunk_to_ola(chunk_start_abs, enhanced)

                    # emit one hop worth of finalized output
                    emit_until = chunk_start_abs + self.hop_size
                    self.emit_available_audio(emit_until)

                    print(
                        f"[tse] processed {chunk_start_abs / self.sample_rate:.1f}s"
                        f"–{chunk_end_abs / self.sample_rate:.1f}s | slot={self.selected_slot}"
                    )

                    self.next_process_end += self.hop_size


if __name__ == "__main__":
    demo = RealTimeTSEDemo(
        sample_rate=16000,
        chunk_sec=5.0,
        hop_sec=2.5,
        emb_context_sec=10.0,
        emb_update_interval_sec=10.0,
        block_sec=0.1,
    )

    demo.run()