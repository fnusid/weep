import argparse
import queue
import time
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

sys.path.append("/home/sidcs/codebase/")

from models.gridnet_causal_net import Net as CausalGridNet
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper


CAUSAL_GRIDNET_ARGS = {
    "spk_emb_dim": 256,
    # "stft_chunk_size": 128,
    "stft_chunk_size": 256,
    "stft_pad_size": 128,
    "stft_back_pad": 128,
    "num_ch": 1,
    "D": 64,
    "L": 0,
    "I": 1,
    "J": 1,
    "B": 3,
    "H": 64,
    "local_atten_len": 50,
    "use_attn": False,
    "chunk_causal": True,
    "spectral_masking": True,
}

TSE_CKPT = "/home/sidcs/model_ckpts/causal_gridnet_joint_training_hardpairs_silence/epochepoch=09-trainlosstrain_loss=-1.636.ckpt"
EMB_CKPT = "/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
SAVE_PATH = "/home/sidcs/codebase/wesep/realtime_causal_gridnet_output.wav"


def print_audio_devices():
    print(sd.query_devices())


def find_device_id(name_substring, kind="output"):
    name_substring = name_substring.lower()
    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        if name_substring not in dev["name"].lower():
            continue
        if kind == "input" and dev["max_input_channels"] > 0:
            return idx
        if kind == "output" and dev["max_output_channels"] > 0:
            return idx

    raise ValueError(f"No {kind} device matching '{name_substring}' found.")


def strip_separator_weights(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state[key[len("model."):]] = value
    return new_state


def strip_dual_model_weights(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        trimmed = key[len("model."):]
        if trimmed.startswith("single_sp_model.") or trimmed.startswith("arcface_loss."):
            continue
        new_state[trimmed] = value
    return new_state


def strip_joint_dual_override(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith("dual_emb_model."):
            new_state[key[len("dual_emb_model."):]] = value
    return new_state


def extract_silence_proto(state_dict):
    silence_key = "dual_emb_loss.loss_fn.silence_proto"
    if silence_key not in state_dict:
        return None
    silence_proto = state_dict[silence_key].detach().float()
    return F.normalize(silence_proto, dim=0, eps=1e-8)


def load_causal_gridnet_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_separator_weights(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    print("[load] separator keys:", len(state_dict))
    return model


def load_dual_model(model, emb_ckpt_path, joint_ckpt_path):
    if emb_ckpt_path is not None:
        emb_state = torch.load(emb_ckpt_path, map_location="cpu")["state_dict"]
        emb_state = strip_dual_model_weights(emb_state)
        model.load_state_dict(emb_state, strict=True)
        print("[load] base dual-embedding keys:", len(emb_state))

    joint_ckpt = torch.load(joint_ckpt_path, map_location="cpu")
    joint_state = strip_joint_dual_override(joint_ckpt["state_dict"])
    if len(joint_state) > 0:
        model.load_state_dict(joint_state, strict=True)
        print("[load] joint dual-embedding override keys:", len(joint_state))
    else:
        print("[load] no dual-embedding override found in joint checkpoint")

    silence_proto = extract_silence_proto(joint_ckpt["state_dict"])
    if silence_proto is None:
        print("[load] no silence prototype found in joint checkpoint")
    else:
        print("[load] loaded learned silence prototype")

    return model, silence_proto


class RealTimeCausalGridNetDemo(nn.Module):
    def __init__(
        self,
        tse_ckpt=TSE_CKPT,
        emb_ckpt=EMB_CKPT,
        sample_rate=16000,
        emb_context_sec=10.0,
        emb_update_interval_sec=10.0,
        disable_embedding_updates=True,
        block_hops=1,
        output_gain=1.0,
        device=None,
        input_device=None,
        output_device=None,
        bypass_enhancement=False,
        save_path=SAVE_PATH,
        auto_select_non_silence=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.emb_context_sec = emb_context_sec
        self.emb_update_interval_sec = emb_update_interval_sec
        self.disable_embedding_updates = disable_embedding_updates
        self.block_hops = block_hops
        self.output_gain = output_gain
        self.bypass_enhancement = bypass_enhancement
        self.save_path = save_path
        self.input_device = input_device
        self.output_device = output_device
        self.auto_select_non_silence = auto_select_non_silence

        self.chunk_size = CAUSAL_GRIDNET_ARGS["stft_chunk_size"]
        self.back_pad = CAUSAL_GRIDNET_ARGS["stft_back_pad"]
        self.pad_size = CAUSAL_GRIDNET_ARGS["stft_pad_size"]
        self.window_size = self.back_pad + self.chunk_size + self.pad_size
        self.block_size = self.chunk_size * self.block_hops
        self.emb_context_size = int(sample_rate * emb_context_sec)
        self.emb_update_interval = int(sample_rate * emb_update_interval_sec)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print("[init] torch device:", self.device)
        print("[init] hop:", self.chunk_size / self.sample_rate, "sec")
        print("[init] analysis window:", self.window_size / self.sample_rate, "sec")
        print("[init] playback block:", self.block_size / self.sample_rate, "sec")
        print("[init] embedding context:", self.emb_context_sec, "sec")
        print("[init] embedding updates disabled:", self.disable_embedding_updates)
        print("[init] output gain:", self.output_gain)
        print("[init] bypass enhancement:", self.bypass_enhancement)

        self.tse_model = CausalGridNet(**CAUSAL_GRIDNET_ARGS)
        self.emb_model = SpeakerEncoderDualWrapper(emb_dim=256)

        self.tse_model = load_causal_gridnet_ckpt(self.tse_model, tse_ckpt)
        emb_ckpt_path = emb_ckpt if emb_ckpt not in [None, "", "none", "None"] else None
        self.emb_model, self.silence_proto = load_dual_model(self.emb_model, emb_ckpt_path, tse_ckpt)

        self.tse_model.to(self.device).eval()
        self.emb_model.to(self.device).eval()
        if self.silence_proto is not None:
            self.silence_proto = self.silence_proto.to(self.device)

        self.embedding1 = torch.zeros(1, 256, device=self.device)
        self.embedding2 = torch.zeros(1, 256, device=self.device)
        self.selected_slot = 1

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.saved_output = []

        self.raw_history = np.zeros(0, dtype=np.float32)
        self.pending_input = np.zeros(0, dtype=np.float32)
        self.total_samples_seen = 0
        self.last_emb_update_sample = 0

        self.model_state = None
        self.model_input_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.last_debug_print = time.time()

    def reset_stream_state(self):
        self.model_state = self.tse_model.init_buffers(batch_size=1, device=self.device)
        self.model_input_buffer = np.zeros(self.window_size, dtype=np.float32)

    def clear_runtime_buffers(self):
        self.pending_input = np.zeros(0, dtype=np.float32)
        self.output_queue = queue.Queue()
        self.saved_output = []

    def resolve_audio_devices(self):
        input_dev = self.input_device
        output_dev = self.output_device

        if input_dev is None:
            input_dev = sd.default.device[0]
        if output_dev is None:
            output_dev = sd.default.device[1]

        if isinstance(input_dev, str):
            input_dev = find_device_id(input_dev, kind="input")
        if isinstance(output_dev, str):
            output_dev = find_device_id(output_dev, kind="output")

        print("[audio] input device:", input_dev)
        print("[audio] output device:", output_dev)
        return input_dev, output_dev

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            pass

        mic_block = indata[:, 0].copy().astype(np.float32)
        self.input_queue.put(mic_block)

        latest = None
        while True:
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            out_block = np.zeros(frames, dtype=np.float32)
        else:
            out_block = latest

        if len(out_block) < frames:
            padded = np.zeros(frames, dtype=np.float32)
            padded[:len(out_block)] = out_block
            out_block = padded
        elif len(out_block) > frames:
            out_block = out_block[:frames]

        outdata[:, 0] = out_block

    @torch.no_grad()
    def compute_dual_embeddings(self, audio_np):
        wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        embs = self.emb_model(wav)
        return embs[:, 0, :], embs[:, 1, :]

    @torch.no_grad()
    def _slot_silence_scores(self, e1, e2):
        if self.silence_proto is None:
            return None, None
        s1 = F.cosine_similarity(e1, self.silence_proto.unsqueeze(0)).item()
        s2 = F.cosine_similarity(e2, self.silence_proto.unsqueeze(0)).item()
        return s1, s2

    @torch.no_grad()
    def initialize_embeddings(self):
        ctx = self.raw_history[-self.emb_context_size:]
        e1, e2 = self.compute_dual_embeddings(ctx)
        self.embedding1 = e1
        self.embedding2 = e2
        sim = F.cosine_similarity(e1, e2).item()
        print(f"[emb] initialized. cos(e1,e2)={sim:.3f}")
        sil1, sil2 = self._slot_silence_scores(e1, e2)
        if sil1 is not None:
            print(f"[emb] slot-silence cosine: slot1={sil1:.3f}, slot2={sil2:.3f}")

    @torch.no_grad()
    def update_embeddings(self):
        if self.disable_embedding_updates:
            return

        if len(self.raw_history) < self.emb_context_size:
            return

        ctx = self.raw_history[-self.emb_context_size:]
        e1, e2 = self.compute_dual_embeddings(ctx)
        sim_new = F.cosine_similarity(e1, e2).item()
        if sim_new >= 0.75:
            return

        c11 = F.cosine_similarity(e1, self.embedding1).item()
        c12 = F.cosine_similarity(e1, self.embedding2).item()
        c21 = F.cosine_similarity(e2, self.embedding1).item()
        c22 = F.cosine_similarity(e2, self.embedding2).item()
        alpha = 0.1

        if c11 > c12 and c22 > c21:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e1
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e2
        elif c12 > c11 and c21 > c22:
            self.embedding1 = (1 - alpha) * self.embedding1 + alpha * e2
            self.embedding2 = (1 - alpha) * self.embedding2 + alpha * e1

    def choose_slot(self):
        sil1, sil2 = self._slot_silence_scores(self.embedding1, self.embedding2)
        recommended_slot = 1
        if sil1 is not None and sil2 is not None:
            recommended_slot = 1 if sil1 < sil2 else 2
            print(f"[select] silence cosine slot1={sil1:.3f}, slot2={sil2:.3f}")
            print(f"[select] recommended speech slot: {recommended_slot}")

        if self.auto_select_non_silence:
            self.selected_slot = recommended_slot
            print(f"[select] auto-selected slot {self.selected_slot}")
            return

        print("\nChoose target speaker slot:")
        print("1 = embedding slot 1")
        print("2 = embedding slot 2")
        if sil1 is not None and sil2 is not None:
            print(f"Press Enter to use recommended slot {recommended_slot}")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "":
            self.selected_slot = recommended_slot
        else:
            self.selected_slot = 2 if choice == "2" else 1
        print(f"[select] using speaker slot {self.selected_slot}")

    @torch.no_grad()
    def process_hop(self, hop_np):
        if self.bypass_enhancement:
            return hop_np.copy()

        self.model_input_buffer[:-self.chunk_size] = self.model_input_buffer[self.chunk_size:]
        self.model_input_buffer[-self.chunk_size:] = hop_np

        x = torch.from_numpy(self.model_input_buffer).float().unsqueeze(0).unsqueeze(0).to(self.device)
        emb = self.embedding1 if self.selected_slot == 1 else self.embedding2

        out = self.tse_model(
            {"mixture": x, "embedding": emb},
            input_state=self.model_state,
            pad=False,
        )
        self.model_state = out["next_state"]

        y = out["output"][0, 0].detach().cpu().numpy().astype(np.float32)
        y = np.clip(y * self.output_gain, -1.0, 1.0)

        now = time.time()
        if now - self.last_debug_print > 1.0:
            rms = float(np.sqrt(np.mean(y**2)))
            print(f"[audio] output rms={rms:.4f}")
            self.last_debug_print = now

        return y

    def queue_output_audio(self, audio_np):
        audio_np = audio_np.astype(np.float32)

        if len(audio_np) != self.block_size:
            padded = np.zeros(self.block_size, dtype=np.float32)
            padded[:min(len(audio_np), self.block_size)] = audio_np[:self.block_size]
            audio_np = padded

        self.saved_output.append(audio_np.copy())
        self.output_queue.put(audio_np)

    def save_output(self):
        if len(self.saved_output) == 0:
            print("[save] no output saved")
            return

        y = np.concatenate(self.saved_output).astype(np.float32)
        y = torch.from_numpy(y).float().unsqueeze(0)
        torchaudio.save(self.save_path, y, self.sample_rate)
        print(f"[save] saved enhanced output to {self.save_path}")

    def run(self):
        print("\nStarting mic stream.")
        print("Initial embedding collection will take", self.emb_context_sec, "seconds.")
        print("Playback starts only after warmup and speaker selection.")
        print("Use headphones to avoid feedback.")
        print("Press Ctrl+C to stop and save output.\n")

        input_dev, output_dev = self.resolve_audio_devices()

        stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=(input_dev, output_dev),
            channels=(1, 1),
            dtype="float32",
            latency="low",
            callback=self.audio_callback,
        )

        initialized = False
        last_print = time.time()

        with stream:
            while True:
                block = self.input_queue.get()
                self.pending_input = np.concatenate([self.pending_input, block])
                self.raw_history = np.concatenate([self.raw_history, block])
                self.total_samples_seen += len(block)

                max_keep = max(self.emb_context_size, self.window_size * 8)
                if len(self.raw_history) > max_keep:
                    self.raw_history = self.raw_history[-max_keep:]

                max_pending = self.chunk_size * 4
                if len(self.pending_input) > max_pending:
                    self.pending_input = self.pending_input[-max_pending:]

                if not initialized and self.total_samples_seen >= self.emb_context_size:
                    self.initialize_embeddings()
                    self.choose_slot()

                    self.reset_stream_state()
                    self.clear_runtime_buffers()
                    initialized = True
                    self.last_emb_update_sample = self.total_samples_seen
                    print("[stream] warmup audio discarded, playback starts from fresh post-selection audio")

                if not initialized:
                    now = time.time()
                    if now - last_print > 1.0:
                        remain = max(0, self.emb_context_size - self.total_samples_seen) / self.sample_rate
                        print(f"[warmup] {remain:.1f}s remaining")
                        last_print = now
                    continue

                while len(self.pending_input) >= self.chunk_size:
                    hop = self.pending_input[:self.chunk_size]
                    self.pending_input = self.pending_input[self.chunk_size:]

                    if self.total_samples_seen - self.last_emb_update_sample >= self.emb_update_interval:
                        self.update_embeddings()
                        self.last_emb_update_sample = self.total_samples_seen

                    enhanced = self.process_hop(hop)
                    self.queue_output_audio(enhanced)


def build_argparser():
    parser = argparse.ArgumentParser(description="Realtime causal TF-GridNet TSE demo")
    parser.add_argument("--tse-ckpt", default=TSE_CKPT)
    parser.add_argument("--emb-ckpt", default=EMB_CKPT)
    parser.add_argument("--save-path", default=SAVE_PATH)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--emb-context-sec", type=float, default=10.0)
    parser.add_argument("--emb-update-interval-sec", type=float, default=10.0)
    parser.add_argument("--disable-embedding-updates", action="store_true", default=True)
    parser.add_argument("--block-hops", type=int, default=1)
    parser.add_argument("--output-gain", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--bypass-enhancement", action="store_true")
    parser.add_argument("--auto-select-non-silence", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.list_devices:
        print_audio_devices()
        raise SystemExit(0)

    demo = RealTimeCausalGridNetDemo(
        tse_ckpt=args.tse_ckpt,
        emb_ckpt=args.emb_ckpt,
        sample_rate=args.sample_rate,
        emb_context_sec=args.emb_context_sec,
        emb_update_interval_sec=args.emb_update_interval_sec,
        disable_embedding_updates=args.disable_embedding_updates,
        block_hops=args.block_hops,
        output_gain=args.output_gain,
        device=args.device,
        input_device=args.input_device,
        output_device=args.output_device,
        bypass_enhancement=args.bypass_enhancement,
        save_path=args.save_path,
        auto_select_non_silence=args.auto_select_non_silence,
    )

    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n[stop] stopping demo")
        demo.save_output()
