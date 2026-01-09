import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from dataset.dataloader import LibriMixDataModule       
from models.dpcnn import DPCCN #TSE

from metrics import SE_metrics
import wandb
import sys
sys.path.append("/home/sidharth./codebase/")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper 
import random
random.seed(42)
import warnings
warnings.filterwarnings("ignore")

import auraloss
import yaml
from pathlib import Path
from omegaconf import OmegaConf

config_path = Path("/home/sidharth./codebase/wesep/confs/config_dpcnn.yaml")

with config_path.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]

hp = OmegaConf.merge(*docs)

import numpy as np



def strip_model_prefix(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v   # remove "model."
        else:
            new_state[k] = v
    return new_state


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

def cosine(a, b):
    """
    a: [B, D]
    b: [B, D]
    returns: [B]
    """
    dot = (a * b).sum(dim=-1)                 # [B]
    an = a.norm(dim=-1) + 1e-8                # [B]
    bn = b.norm(dim=-1) + 1e-8                # [B]
    return dot / (an * bn)

class E2EpSE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)

        device="cuda" if torch.cuda.is_available() else "cpu"   
   
        #Get the dual-emb model and teacher model
        
        dual_emb_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
        # dual_emb_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_3spft_wavlm_linear_dualemb_tr360/best-epoch=54-val_separation=0.000.ckpt"
        dual_emb_ckpt = torch.load(dual_emb_ckpt_path, map_location=device)
        state = strip_dual_model_weights(dual_emb_ckpt["state_dict"])
        self.dual_emb_model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
        self.dual_emb_model.load_state_dict(state, strict=True)
        self.dual_emb_model.to(device).eval()
        for param in self.dual_emb_model.parameters():
            param.requires_grad = False

        self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        teacher_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        state = ckpt["state_dict"]

        filtered = {}
        for k, v in state.items():
            # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
            if k.startswith("model.") and ("arcface" not in k):
                filtered[k.replace("model.", "", 1)] = v

        print("Loaded teacher keys:", len(filtered))

        self.single_sp_model.load_state_dict(filtered, strict=True)
        self.single_sp_model.eval()
        for param in self.single_sp_model.parameters():
            param.requires_grad = False



        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = SE_metrics(device="cpu")  # will overwrite device at runtime

        self.model = DPCCN(**hp.model_args.tse_model)
        self.loss = auraloss.time.SISDRLoss()


    def forward(self, wav, emb=None):
        """
        wav: [B, T] (or [B, 1, T])
        returns: [B, T] or ([B, 1, T])
        """
        if wav.ndim== 3:
            wav = wav.squeeze(1)  # [B, T]
        return self.model(wav, emb)

    # -----------------------------
    # TRAINING
    # -----------------------------
    def training_step(self, batch, batch_idx):
        """
        batch: (wav, speaker_label)
          wav: [B, T]
          labels: [B, 2]  (speaker IDs, already mapped to [0..num_classes-1])
        """
        mix, source, labels = batch
        # emb = self.forward(mix)                    # [B, 2, emb_dim]
        #change here
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B, emb_dim]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B, emb_dim]
            gt_embs = torch.stack([emb1, emb2], dim=1)  # [B, 2, emb_dim]
        
        randomly_chosen_source = random.randint(0,1) #0, or 1
        if randomly_chosen_source == 0:
            emb_tgt = emb1 #[B, emb_dim]
            target_speech = source[:, 0, :]
        else:
            emb_tgt = emb2
            target_speech = source[:, 1, :] #[B, T]
        
        embs = self.dual_emb_model(mix)# [B, 2, emb_dim]
        e1 = embs[:, 0, :]
        e2 = embs[:, 1, :]

        cosine1 = cosine(e1, emb_tgt)
        cosine2 = cosine(e2, emb_tgt)
        choose_mask = (cosine1 > cosine2).unsqueeze(-1)   # [B,1]
        pred_emb = torch.where(choose_mask, e1, e2)
        
        #condition dccrn on pred_emb
        #convert mix to spec
      
        out,_ = self.forward(mix, emb = pred_emb) 

        min_len = min(out.shape[-1], source.shape[-1])
        out = out[..., :min_len]

        source = target_speech[..., :min_len]
        loss = self.loss(out, source)
        # out_wav = self.audio_utils.spec2wav(out.detach().numpy(), mix_phase)

    


        self.log(
            "train/SI-SDR_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
        )
        return loss

    # -----------------------------
    # VALIDATION (per-batch)
    # -----------------------------


    def validation_step(self, batch, batch_idx):
        """
        For now we just compute arcface loss as a simple val loss.
        The clustering metrics are done in validation_epoch_end
        on the entire validation set.
        """
        mix, source, labels = batch
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B, emb_dim]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B, emb_dim]
            gt_embs = torch.stack([emb1, emb2], dim=1)  # [B, 2, emb_dim]
        
        randomly_chosen_source = random.randint(0,1) #0, or 1
        if randomly_chosen_source == 0:
            emb_tgt = emb1 #[B, emb_dim]
            #true
            target_speech = source[:, 0, :]
            #reversed
            # emb_tgt = emb2

        else:
            emb_tgt = emb2
            #true
            target_speech = source[:, 1, :] #[B, T]
            #reversed
            # emb_tgt = emb1

        
        embs = self.dual_emb_model(mix)# [B, 2, emb_dim]
        e1 = embs[:, 0, :]
        e2 = embs[:, 1, :]

        cosine1 = cosine(e1, emb_tgt)
        cosine2 = cosine(e2, emb_tgt)
        choose_mask = (cosine1 > cosine2).unsqueeze(-1)   # [B,1]
        pred_emb = torch.where(choose_mask, e1, e2)
        #condition dccrn on pred_emb
        out,_ = self.forward(mix, emb = pred_emb) #list of three wavs
        # out = out[0]

        min_len = min(out.shape[-1], source.shape[-1])
        out = out[..., :min_len]
        source = target_speech[..., :min_len]
        #compute validation metrics #PESQ, DNSMOS metrics

        '''
        metrics = {
            "PESQ": float(torch.tensor(self.pesq_scores).nanmean()),
            "STOI": float(torch.tensor(self.stoi_scores).nanmean()),
            "SI_SDR": float(torch.tensor(self.sisdr_scores).nanmean()),
            "SIG": float(torch.tensor(self.SIG).nanmean()),
            "BAK": float(torch.tensor(self.BAK).nanmean()),
            "OVRL": float(torch.tensor(self.OVRL).nanmean()),
        '''
        self.metrics.update(out, source)
        return {}

    # -----------------------------
    # VALIDATION (end of epoch)
    # -----------------------------
    def on_validation_epoch_end(self):
        # 1) Compute validation metrics
        m = self.metrics.compute()
        for k, v in m.items():
            self.log(f"val/{k}", v, prog_bar=True)
        self.metrics.reset()

        # 2) Log audio samples (5 fixed samples)
        if not hasattr(self, "fixed_val_batch"):
            # Save a fixed batch on first val step
            mix, src, _ = next(iter(self.trainer.datamodule.val_dataloader()))
            self.fixed_val_batch = (mix[:5], src[:5])

        mix, src = self.fixed_val_batch
        mix = mix.to(self.device)
        src = src.to(self.device)

        # Determine GT target for logging
        idx = random.randint(0,1)
        tgt = src[:, idx, :]   # always log speaker 0 for visualization

        # Run forward pass
        with torch.no_grad():
            # you already have selection logic in training_step
            # but for visualization pick one speaker deterministically
            emb1 = self.single_sp_model(src[:, 0, :])
            emb2 = self.single_sp_model(src[:, 1, :])
            if idx == 0:
                emb_tgt = emb1
            else:
                emb_tgt = emb2
            embs = self.dual_emb_model(mix)
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

            cosine1 = cosine(e1, emb_tgt)
            cosine2 = cosine(e2, emb_tgt)
            choose_mask = (cosine1 > cosine2).unsqueeze(-1)
            pred_emb = torch.where(choose_mask, e1, e2)

            pred,_ = self.forward(mix, emb = pred_emb)  # list of three wavs
            # pred = pred[0]


        # Match lengths
        min_len = min(pred.shape[-1], tgt.shape[-1])
        pred = pred[..., :min_len]
        tgt = tgt[..., :min_len]
        mix = mix[..., :min_len]

        # Log each sample
        for i in range(mix.shape[0]):
            m_np = mix[i].detach().cpu().numpy().astype("float32")
            t_np = tgt[i].detach().cpu().numpy().astype("float32")
            p_np = pred[i].detach().cpu().numpy().astype("float32")

     
            run = self.logger.experiment

            run.log({f"audio/mix_{i}":  wandb.Audio(m_np, sample_rate=16000)})
            run.log({f"audio/tgt_{i}":  wandb.Audio(t_np, sample_rate=16000)})
            run.log({f"audio/pred_{i}": wandb.Audio(p_np, sample_rate=16000)})


    # def get_pred_from_mix(self, mix, source):
    #     """
    #     mix:    [1, T]
    #     source: [1, 2, T]
    #     """
    #     with torch.no_grad():
    #         emb1 = self.single_sp_model(source[:, 0, :])
    #         emb2 = self.single_sp_model(source[:, 1, :])

    #         # randomly choose one target (for personalization)
    #         # but use deterministic behavior in validation:
    #         # emb_tgt = emb1  # always choose source[0] or use both
    #         idx = random.randint(0,1)
    #         if idx==0:
    #             emb_tgt = emb1
    #         else:
    #             emb_tgt = emb2


    #         embs = self.dual_emb_model(mix)
    #         e1 = embs[:, 0, :]
    #         e2 = embs[:, 1, :]

    #         cosine1 = cosine(e1, emb_tgt)
    #         cosine2 = cosine(e2, emb_tgt)
    #         #true
    #         pred_emb = torch.where((cosine1 > cosine2).unsqueeze(-1), e1, e2)
            

    #         pred = self.model(mix, emb=pred_emb)[1]  # [1, T']

    #         # trim
    #         min_len = min(pred.shape[-1], source.shape[-1])
    #         pred = pred[..., :min_len]
    #         tgt = source[:, idx, :min_len]  # or 1 depending on emb_tgt

    #     return pred, tgt
    def on_test_start(self):
        # Separate accumulator so test doesn't mix with val
        self.test_metrics = SE_metrics(device="cpu")

    def test_step(self, batch, batch_idx):
        mix, source, labels = batch  # mix: [B,T], source: [B,2,T]

        # --- teacher embeddings from clean sources ---
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B,D]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B,D]

            # --- dual embeddings from mixture (unordered) ---
            embs = self.dual_emb_model(mix)               # [B,2,D]
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

            # Evaluate BOTH targets for each mixture
            # idx = np.random.choice([0, 1])
            for idx in [0, 1]:
                emb_tgt = emb1 if idx == 0 else emb2
                tgt_wav = source[:, idx, :]               # [B,T]

                # pick the mixture-derived embedding closer to emb_tgt
                c1 = cosine(e1, emb_tgt)
                c2 = cosine(e2, emb_tgt)
                choose_mask = (c1 > c2).unsqueeze(-1)     # [B,1]
                pred_emb = torch.where(choose_mask, e1, e2)

                # TasNet forward (list of 3 wavs)
                out = self.forward(mix, emb=pred_emb)
                pred = out[0]  # [B, T']
        
                # trim to match
                min_len = min(pred.shape[-1], tgt_wav.shape[-1])
                pred = pred[..., :min_len]
                tgt_wav = tgt_wav[..., :min_len]

                # accumulate metrics
                self.test_metrics.update(pred, tgt_wav)

        return {}

    def on_test_epoch_end(self):
        m = self.test_metrics.compute()
        for k, v in m.items():
            self.log(f"test/{k}", v, prog_bar=True)
        self.test_metrics.reset()
    # -----------------------------
    # OPTIMIZER + SCHEDULER
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        # return optimizer

        # monitor one of the embedding metrics, e.g., separation (higher is better)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min", 
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/SI-SDR_loss",
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix" 
    SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    # SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"

    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=2, 
        num_workers=24, # Set this to your preference
        num_speakers=2
    )

    model = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )

    wandb_logger = WandbLogger(
        project="pDCCRN_2sp",
        name="pDCCRN_2sp_dpccn",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/SI-SDR_loss",
        mode="min",
        save_top_k=-1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn/"
    )

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="gpu",
        # precision="16-mixed",    # <-- mixed precision
        devices=[0, 1, 2, 3],
        # devices=[0],
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[ckpt],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
        
    )

    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=[0],
    #     max_epochs=100,
    #     logger=wandb_logger,
    #     overfit_batches=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    #     enable_checkpointing=False,
    # )

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    # )
    # trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt")

    # trainer.validate(model, datamodule=dm, ckpt_path = "/mnt/disks/data/model_ckpts/archive_ckpt/pFCCRN_2sp/best-epoch=60-val_separation=0.000.ckpt")
    wandb.finish()
