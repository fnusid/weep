import torch 
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import json
import numpy as np
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence


def mix_noise_with_snr(clean, noise, snr_db):
    """
    clean, noise: torch tensors [T]
    snr_db: desired SNR = clean_power / noise_power (in dB)
    """
    # match lengths
   
    if noise.ndim > 1:
        noise = noise.mean(0)
    if clean.ndim > 1:
        clean = clean.mean(0)
    if len(noise) < len(clean):
        diff = len(clean) - len(noise)
        padded_noise = F.pad(noise, (diff//2, diff - (diff//2)))
        noise = padded_noise
    else:
        noise = noise[:len(clean)]

    # compute powers
    clean_power = clean.pow(2).mean()
    noise_power  = noise.pow(2).mean()

    # scaling for noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / (noise_power + 1e-8))

    return clean + scale * noise

class MyLibri2Mix(Dataset):
    def __init__(self, metadata_path, speaker_map_path, num_speakers=2, sampling_rate=16000, split='train'):
        super().__init__()

        self.metadata = pd.read_csv(metadata_path)
   
        self.noise_prob = 0.8
        if split=='train':
            self.noise_file_path = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/noise_files_embedding_model/freesound_noise_bins.json" #[freesound, sound-bible, wham tr]
        elif split == 'val' or split == 'test':
            self.noise_file_path = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/noise_files_embedding_model/wham_tt_noise_bins.json"

        with open(self.noise_file_path, 'r') as f:
            self.noise_dict = json.load(f)

        


        self.num_speakers = num_speakers

        with open(speaker_map_path, 'r') as f:
            self.speaker_to_index = json.load(f)
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx] #mixture_ID,mixture_path,source_1_path,source_2_path,speaker_1_ID,speaker_2_ID,length

        mix_path = row['mixture_path']

        mix_audio, _ = torchaudio.load(mix_path)

        #add noise with a probability

        if random.random() < self.noise_prob:

            mix_audio_len = mix_audio.shape[-1] // 16000  # seconds

            # Parse bin upper bounds
            bins = list(self.noise_dict.keys())
            uppers = np.array([int(b.split("-")[1]) for b in bins])

            # Find smallest upper bound ≥ mix length
            cand = uppers[uppers > mix_audio_len]
            if len(cand) == 0:
                # fallback to largest bin
                ub = uppers.max()
            else:
                ub = cand[0]

            # Construct candidate keys for that UB
            candidate_keys = []
            if ub in [5, 10]:
                candidate_keys.append(f"{ub-5}-{ub}")
            else:
                candidate_keys.append(f"{ub-10}-{ub}")

            # Now pick a **valid** key from noise_dict
            valid_key = None
            for k in candidate_keys:
                if k in self.noise_dict and len(self.noise_dict[k]) > 0:
                    valid_key = k
                    break

            # If still empty → find nearest NON-EMPTY bin
            if valid_key is None:
                for k in bins:
                    if len(self.noise_dict[k]) > 0:
                        valid_key = k
                        break

            # Final fallback: skip adding noise
            if valid_key is not None:
                noise_file = random.choice(self.noise_dict[valid_key])
                noise_audio, sr = torchaudio.load(noise_file)

                r = random.random()
                if r < 0.4:
                    snr = random.uniform(-5, 5)
                elif r < 0.8:
                    snr = random.uniform(5, 15)
                else:
                    snr = random.uniform(15, 25)

                mix_audio = mix_noise_with_snr(mix_audio, noise_audio, snr)

        source_audios = []
        for i in range(self.num_speakers):
            s_path = row[f"source_{i+1}_path"]
            s_audio,_ = torchaudio.load(s_path)
            source_audios.append(s_audio)
        
        sources_tensor = torch.cat(source_audios, dim = 0) #[B,2,T]

        speaker_indices = []
        for i in range(self.num_speakers):
            speaker_id = str(row[f"speaker_{i+1}_ID"])
            if speaker_id in self.speaker_to_index:
                index = self.speaker_to_index[speaker_id]
            else:
                index = int(speaker_id)


            speaker_indices.append(index)
        
        labels_tensor = torch.tensor(speaker_indices, dtype=torch.long)

        return mix_audio.squeeze(0), sources_tensor, labels_tensor



def librimix_collate(batch):

    mix, source, labels = zip(*batch)
    '''
    mix: [B, T]
    source: [B, 2, T]

    '''
    mix_padded = pad_sequence(mix, batch_first=True, padding_value=0.0)

    #permute the sources to [T, 2]

    sources_permuted = [s.permute(1,0) for s in source]
    sources_padded = pad_sequence(sources_permuted, batch_first=True, padding_value = 0.0)

    sources_padded = sources_padded.permute(0,2,1)

    labels = torch.stack(labels)

    return mix_padded, sources_padded, labels



class LibriMixDataModule(pl.LightningDataModule):

    def __init__(self, data_root, speaker_map_path, 
                batch_size=32,
                num_workers=0,
                num_speakers=2,
                sample_rate=16000):
        
        super().__init__()
        self.data_root = data_root
        self.speaker_map_path = speaker_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_speakers = num_speakers
        self.sampling_rate = sample_rate

        self.persistent_workers = True if self.num_workers > 0 else False

        self.base_data_path = os.path.join(self.data_root, f"3sp/Libri3Mix_ovl50to80/wav16k/min") #/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_train-360_mix_clean.csv, /mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json

        self.metadata_path = os.path.join(self.base_data_path, "metadata")


    def setup(self, stage=None):



        train_meta = os.path.join(self.metadata_path, "mixture_train-360_mix_clean.csv")
        val_meta = os.path.join(self.metadata_path, "mixture_dev_mix_clean.csv")
        test_meta = os.path.join(self.metadata_path, "mixture_test_mix_clean.csv")

        self.train_dataset = MyLibri2Mix(
            metadata_path=train_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            split='train'
        )
        self.val_dataset = MyLibri2Mix(
            metadata_path=val_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            split='val'
        )
        self.test_dataset = MyLibri2Mix(
            metadata_path=test_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers,
            split='test'
        )
        self.fixed_val_indices = list(range(5))  # first 5 samples

    def get_fixed_batch(self):
        """Return a fixed batch of 5 examples."""
        dataset = self.val_dataset
        items = [dataset[i] for i in self.fixed_val_indices]
        mixes   = [i[0] for i in items]    # list of [T]
        sources = [i[1] for i in items]    # list of [2, T]
        labels  = [i[2] for i in items]
        return mixes, sources, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )


if __name__ == '__main__':

    # breakpoint()
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_03_08/Libri2Mix_ovl30to80/wav16k/min/metadata/train360_mapping.json"

    dataset = LibriMixDataModule(
        data_root, speaker_map_path, 
                batch_size=4,
                num_workers=0,
                num_speakers=2,
                sample_rate=16000
    )
    dataset.setup()
    dl = dataset.train_dataloader()
    breakpoint()
    for i, (mix, source, labels) in enumerate(dl):
        breakpoint()
        for b in range(mix.shape[0]):
            torchaudio.save(f"/home/sidharth./codebase/wavlm_dual_embedding/dataset_samples/mix_{b}.wav", mix[b].unsqueeze(0), 16000)
            torchaudio.save(f"/home/sidharth./codebase/wavlm_dual_embedding/dataset_samples/source1_{b}.wav", source[b,0,:].unsqueeze(0), 16000)
            torchaudio.save(f"/home/sidharth./codebase/wavlm_dual_embedding/dataset_samples/source2_{b}.wav", source[b,1,:].unsqueeze(0), 16000)


        breakpoint()
