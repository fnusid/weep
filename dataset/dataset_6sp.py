import os
import json
import numpy as np
import soundfile as sf
import webrtcvad
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


'''
On-the-fly LibriSpeech + WHAM! mixtures with up to `max_speakers` speakers.

train: LibriSpeech/train-clean-360 + wham_noise/tr  (new random mixtures every epoch)
test : LibriSpeech/test-clean      + wham_noise/tt  (fixed mixtures, seeded by index)

Per mixture:
  1. n_spk ~ U{min_speakers..max_speakers}, distinct speakers, one random utterance each
  2. crop   : random crop of length ~ U[min_src_len, segment] s taken inside one speech region
              (WebRTC VAD, pauses < vad_max_gap merged), so leading/trailing silence and long
              pauses are never included; shorter if the region is shorter
  3. gain   : scale crop to RMS level ~ U[level_range] dBFS
  4. offset : place crop at a random offset inside the segment
  5. noise  : random WHAM! crop/channel, scaled to SNR ~ U[snr_range] dB w.r.t. the speech sum
  6. if the mixture peak exceeds 0.99, mixture and sources are rescaled together

Returns (mixture [T], sources [max_speakers, T], labels [max_speakers]).
Unused slots have all-zero sources and label -1, so mixture == sources.sum(0) + noise.
'''

EPS = 1e-8
#set seed
import random
random.seed(42)

def _index_librispeech(split_dir):
    """{speaker_id: [flac paths]} for a LibriSpeech split laid out as spk/chapter/*.flac"""
    spk2files = {}
    for spk in sorted(os.listdir(split_dir)):
        spk_dir = os.path.join(split_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        files = []
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            files += sorted(os.path.join(chap_dir, f) for f in os.listdir(chap_dir) if f.endswith(".flac"))
        if files:
            spk2files[spk] = files
    return spk2files


def _read_random_crop(path, length, rng):
    """Read a random crop of `length` samples (or the whole file if shorter) -> [n, C] float32"""
    with sf.SoundFile(path) as f:
        n = f.frames
        length = min(length, n)
        start = int(rng.integers(0, n - length + 1))
        f.seek(start)
        return f.read(length, dtype="float32", always_2d=True)


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + EPS)


def _speech_regions(x, sr, vad, frame_ms=30, max_gap=0.3):
    """[(start, end)] sample spans that WebRTC VAD marks as speech,
    with pauses shorter than `max_gap` seconds merged into the surrounding speech"""
    hop = sr * frame_ms // 1000
    n_frames = len(x) // hop
    if n_frames == 0:
        return [(0, len(x))]
    pcm = (np.clip(x[:n_frames * hop], -1.0, 1.0) * 32767).astype(np.int16)
    active = np.array(
        [vad.is_speech(pcm[i * hop:(i + 1) * hop].tobytes(), sr) for i in range(n_frames)], dtype=np.int8
    )
    if not active.any():
        return [(0, len(x))]

    edges = np.diff(np.concatenate([[0], active, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    max_gap_frames = int(max_gap * 1000 / frame_ms)
    regions = []
    for s, e in zip(starts, ends):
        if regions and s - regions[-1][1] < max_gap_frames:
            regions[-1][1] = e
        else:
            regions.append([s, e])
    return [(s * hop, e * hop) for s, e in regions]


class LibriWhamMix(Dataset):
    def __init__(
        self,
        libri_split_dir,
        wham_split_dir,
        speaker_map_path=None,
        num_mixtures=50000,
        min_speakers=1,
        max_speakers=6,
        segment=6.0,
        min_src_len=1.5,
        level_range=(-33.0, -25.0),
        snr_range=(-5.0, 20.0),
        vad_mode=3,
        vad_max_gap=0.3,
        deterministic=False,
        seed=0,
        sampling_rate=16000,
    ):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.segment_samples = int(segment * sampling_rate)
        self.min_src_samples = min(int(min_src_len * sampling_rate), self.segment_samples)
        self.level_range = level_range
        self.snr_range = snr_range
        self.vad_mode = vad_mode
        self.vad_max_gap = vad_max_gap
        self.deterministic = deterministic
        self.seed = seed
        self.sampling_rate = sampling_rate

        self.spk2files = _index_librispeech(libri_split_dir)
        self.speakers = sorted(self.spk2files)
        assert len(self.speakers) >= max_speakers, f"only {len(self.speakers)} speakers in {libri_split_dir}"

        self.noise_files = sorted(
            os.path.join(wham_split_dir, f) for f in os.listdir(wham_split_dir) if f.endswith(".wav")
        )
        assert self.noise_files, f"no wav files in {wham_split_dir}"

        if speaker_map_path is not None:
            with open(speaker_map_path, "r") as f:
                self.speaker_to_index = json.load(f)
        else:
            self.speaker_to_index = {}

    def __len__(self):
        return self.num_mixtures

    def _rng(self, idx):
        if self.deterministic:
            # same mixture for a given idx every epoch / run
            return np.random.default_rng([self.seed, idx])
        # torch RNG is seeded per worker and per epoch by the DataLoader; add the DDP rank
        # so ranks don't produce identical mixtures when seeded identically
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return np.random.default_rng([int(torch.randint(0, 2 ** 62, (1,)).item()), rank])

    def _label(self, spk):
        # train speakers -> class index; unmapped (e.g. test) speakers -> raw LibriSpeech ID
        return self.speaker_to_index.get(spk, int(spk))

    def _speech_crop(self, path, crop_len, rng):
        """Random crop of up to `crop_len` samples lying entirely inside one speech region"""
        utt, _ = sf.read(path, dtype="float32")
        # fresh Vad per utterance: WebRTC VAD adapts its state across calls, which would make
        # the decision depend on previously processed files (and break test determinism)
        vad = webrtcvad.Vad(self.vad_mode)
        regions = _speech_regions(utt, self.sampling_rate, vad, max_gap=self.vad_max_gap)
        lens = np.array([e - s for s, e in regions])
        # prefer regions long enough for min_src_len, sampled proportional to length
        weights = np.where(lens >= self.min_src_samples, lens, 0)
        if weights.sum() == 0:
            k = int(lens.argmax())
        else:
            k = int(rng.choice(len(regions), p=weights / weights.sum()))
        s, e = regions[k]
        crop_len = min(crop_len, e - s)
        start = s + int(rng.integers(0, e - s - crop_len + 1))
        return utt[start:start + crop_len]

    def _noise(self, rng, T):
        path = self.noise_files[int(rng.integers(len(self.noise_files)))]
        noise = _read_random_crop(path, T, rng)
        noise = noise[:, int(rng.integers(noise.shape[1]))]
        if len(noise) < T:
            noise = np.tile(noise, int(np.ceil(T / len(noise))))[:T]
        return noise

    def __getitem__(self, idx):
        rng = self._rng(idx)
        T = self.segment_samples

        n_spk = int(rng.integers(self.min_speakers, self.max_speakers + 1))
        spk_ids = rng.choice(len(self.speakers), size=n_spk, replace=False)

        sources = np.zeros((self.max_speakers, T), dtype=np.float32)
        labels = np.full(self.max_speakers, -1, dtype=np.int64)

        for i, s in enumerate(spk_ids):
            spk = self.speakers[s]
            files = self.spk2files[spk]
            path = files[int(rng.integers(len(files)))]

            crop_len = int(rng.integers(self.min_src_samples, T + 1))
            utt = self._speech_crop(path, crop_len, rng)

            target_db = rng.uniform(*self.level_range)
            utt = utt * (10 ** (target_db / 20) / _rms(utt))

            offset = int(rng.integers(0, T - len(utt) + 1))
            sources[i, offset:offset + len(utt)] = utt
            labels[i] = self._label(spk)

        speech = sources.sum(0)
        noise = self._noise(rng, T)
        snr_db = rng.uniform(*self.snr_range)
        noise = noise * (_rms(speech) / (_rms(noise) * 10 ** (snr_db / 20)))
        mixture = speech + noise

        peak = np.abs(mixture).max()
        if peak > 0.99:
            scale = 0.99 / peak
            mixture *= scale
            sources *= scale

        return (
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(sources),
            torch.from_numpy(labels),
        )


class LibriMixDataModule(pl.LightningDataModule):

    def __init__(self, data_root, speaker_map_path,
                batch_size=32,
                num_workers=0,
                min_speakers=1,
                max_speakers=6,
                segment=15.0,
                min_src_len=1.5,
                train_mixtures_per_epoch=50000,
                test_mixtures=3000,
                train_snr_range=(-5.0, 20.0),
                test_snr_range=(0.0, 15.0),
                level_range=(-33.0, -25.0),
                test_seed=1234,
                sample_rate=16000):

        super().__init__()
        self.data_root = data_root  # .../LibriMix/LibriMix  (contains LibriSpeech/ and wham_noise/)
        self.speaker_map_path = speaker_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_rate = sample_rate

        self.common_kwargs = dict(
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            segment=segment,
            min_src_len=min_src_len,
            level_range=level_range,
            sampling_rate=sample_rate,
        )
        self.train_mixtures_per_epoch = train_mixtures_per_epoch
        self.test_mixtures = test_mixtures
        self.train_snr_range = train_snr_range
        self.test_snr_range = test_snr_range
        self.test_seed = test_seed

        self.persistent_workers = True if self.num_workers > 0 else False

        self.libri_root = os.path.join(self.data_root, "LibriSpeech")
        self.wham_root = os.path.join(self.data_root, "wham_noise")

    def setup(self, stage=None):

        self.train_dataset = LibriWhamMix(
            libri_split_dir=os.path.join(self.libri_root, "train-clean-360"),
            wham_split_dir=os.path.join(self.wham_root, "tr"),
            speaker_map_path=self.speaker_map_path,
            num_mixtures=self.train_mixtures_per_epoch,
            snr_range=self.train_snr_range,
            deterministic=False,
            **self.common_kwargs,
        )
        self.test_dataset = LibriWhamMix(
            libri_split_dir=os.path.join(self.libri_root, "test-clean"),
            wham_split_dir=os.path.join(self.wham_root, "tt"),
            speaker_map_path=None,  # test speakers are unseen -> raw speaker IDs
            num_mixtures=self.test_mixtures,
            snr_range=self.test_snr_range,
            deterministic=True,
            seed=self.test_seed,
            **self.common_kwargs,
        )
        self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # every item is already a fresh random mixture
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )


if __name__ == '__main__':

    data_root = "/tmp/sidcs/turbo/sidcs_backup/datasets/LibriMix/LibriMix"
    speaker_map_path = os.path.join(data_root, "LibriSpeech/train-360_mapping.json")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_samples")
    os.makedirs(save_dir, exist_ok=True)

    dm = LibriMixDataModule(data_root, speaker_map_path, batch_size=4, num_workers=0)
    dm.setup()

    for split, dl in [("train", dm.train_dataloader()), ("test", dm.test_dataloader())]:
        mix, sources, labels = next(iter(dl))
        print(split, mix.shape, sources.shape, labels.tolist())
        for b in range(mix.shape[0]):
            torchaudio.save(os.path.join(save_dir, f"{split}_mix_{b}.wav"), mix[b].unsqueeze(0), 16000)
            for s in range(sources.shape[1]):
                if labels[b, s] >= 0:
                    torchaudio.save(os.path.join(save_dir, f"{split}_mix_{b}_src{s}_spk{labels[b, s].item()}.wav"),
                                    sources[b, s].unsqueeze(0), 16000)
