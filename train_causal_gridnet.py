import json
import random
import argparse
import warnings
from pathlib import Path

import auraloss
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.dataloader import LibriMixDataModule
from metrics import SE_metrics
from models.gridnet_causal_net import Net as CausalGridNet

import sys

sys.path.append("/home/sidcs/codebase/")

from wavlm_dual_embedding.loss import LossWraper
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper


random.seed(42)
warnings.filterwarnings("ignore")


CAUSAL_GRIDNET_ARGS = {
    "spk_emb_dim": 256,
    "stft_chunk_size": 128,
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

def norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total = total + (grad.detach() ** 2).sum()
    return torch.sqrt(total + 1e-12)


def cosine(a, b):
    dot = (a * b).sum(dim=-1)
    a_norm = a.norm(dim=-1) + 1e-8
    b_norm = b.norm(dim=-1) + 1e-8
    return dot / (a_norm * b_norm)


class E2EpSE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        slot_repulsion_weight: float = 0.1,
        slot_repulsion_margin: float = 0.0,
        speaker_map_path: str = "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json",
        model_args: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_args"])
        self.model_args = dict(CAUSAL_GRIDNET_ARGS if model_args is None else model_args)

        with open(speaker_map_path, "r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

        self.dual_emb_model = SpeakerEncoderDualWrapper(
            emb_dim=emb_dim,
            finetune_wavlm=True,
        )
        self.dual_emb_loss = LossWraper(
            slot_repulsion_weight=0,
            slot_repulsion_margin=0,
            emb_dim=emb_dim,
        )

        self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        teacher_ckpt_path = Path(
            "/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
        )
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        filtered_state = {}
        for key, value in ckpt["state_dict"].items():
            if key.startswith("model.") and "arcface" not in key:
                filtered_state[key.replace("model.", "", 1)] = value

        self.single_sp_model.load_state_dict(filtered_state, strict=True)
        self.single_sp_model.eval()
        for param in self.single_sp_model.parameters():
            param.requires_grad = False

        self.metrics = SE_metrics(device="cpu")
        self.model = CausalGridNet(**self.model_args)
        self.loss = auraloss.time.SISDRLoss()
        self.mr_stft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],
            hop_sizes=[64, 128, 256],
            win_lengths=[256, 512, 1024],
            scale=None,
            sample_rate=16000,
            perceptual_weighting=False,
        )

    def load_compatible_checkpoint(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[init] loaded compatible weights from {ckpt_path}")
        if missing:
            print(f"[init] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[init] unexpected keys ({len(unexpected)}): {unexpected}")

    def forward(self, wav, emb=None, input_state=None, pad=True):
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if emb is not None and emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb[:, 0]

        outputs = self.model(
            {"mixture": wav, "embedding": emb},
            input_state=input_state,
            pad=pad,
        )
        estimate = outputs["output"]
        if estimate.ndim == 3 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
        return estimate, outputs["next_state"]

    def _teacher_embeddings(self, source):
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])
            emb2 = self.single_sp_model(source[:, 1, :])
        return emb1, emb2

    def _select_target(self, source, emb1, emb2, deterministic_idx=None):
        active_mask = source.abs().sum(dim=-1) > 1e-8
        batch_size = source.shape[0]
        device = source.device

        if deterministic_idx is not None:
            if isinstance(deterministic_idx, int):
                idx = torch.full((batch_size,), deterministic_idx, device=device, dtype=torch.long)
            else:
                idx = deterministic_idx.to(device=device, dtype=torch.long)
        else:
            idx = torch.randint(0, 2, (batch_size,), device=device)
            only_spk0 = active_mask[:, 0] & (~active_mask[:, 1])
            only_spk1 = active_mask[:, 1] & (~active_mask[:, 0])
            idx = torch.where(only_spk0, torch.zeros_like(idx), idx)
            idx = torch.where(only_spk1, torch.ones_like(idx), idx)

        target_emb = torch.where(idx.unsqueeze(1) == 0, emb1, emb2)
        other_emb = torch.where(idx.unsqueeze(1) == 0, emb2, emb1)
        target_speech = torch.where(idx.unsqueeze(1) == 0, source[:, 0, :], source[:, 1, :])
        other_speech = torch.where(idx.unsqueeze(1) == 0, source[:, 1, :], source[:, 0, :])
        return target_emb, target_speech, other_emb, other_speech

    def _mixture_conditioning_embedding(self, mix, emb_tgt):
        embs = self.dual_emb_model(mix)
        e1 = embs[:, 0, :]
        e2 = embs[:, 1, :]

        cos1 = cosine(e1, emb_tgt)
        cos2 = cosine(e2, emb_tgt)
        scores = torch.stack([cos1, cos2], dim=1)

        tau = 0.5
        weights = torch.softmax(scores / tau, dim=1)
        pred_emb = weights[:, 0:1] * e1 + weights[:, 1:2] * e2

        return embs, pred_emb

    def _run_separator(self, mix, pred_emb):
        estimate, _ = self.forward(mix, emb=pred_emb)
        return estimate

    def _trim_pair(self, pred, target):
        min_len = min(pred.shape[-1], target.shape[-1])
        return pred[..., :min_len], target[..., :min_len]

    def training_step(self, batch, batch_idx):
        mix, source, _ = batch

        emb1, emb2 = self._teacher_embeddings(source)
        gt_embs = torch.stack([emb1, emb2], dim=1)
        # silence_mask = source.abs().sum(dim=-1) <= 1e-8
        silence_mask = None

        emb_tgt, target_speech, _, _ = self._select_target(source, emb1, emb2)
        embs, pred_emb = self._mixture_conditioning_embedding(mix, emb_tgt)

        loss_emb_out = self.dual_emb_loss(
            embs,
            gt_embs,
            silence_mask=silence_mask,
            return_components=True,
        )
        loss_emb = loss_emb_out["loss"]
        estimate = self._run_separator(mix, pred_emb)
        estimate, target_speech = self._trim_pair(estimate, target_speech)

        loss_tse = self.loss(estimate, target_speech)

        if estimate.ndim ==2 and target_speech.ndim == 2:



            loss_mrstft = self.mr_stft(estimate.unsqueeze(1), target_speech.unsqueeze(1))
        else:
            loss_mrstft = self.mr_stft(estimate, target_speech)

        loss_tse = loss_tse + 0.1 * loss_mrstft #change it later if needed the weights

        emb_params = [p for p in self.dual_emb_model.parameters() if p.requires_grad]
        grad_tse = torch.autograd.grad(
            loss_tse,
            emb_params,
            retain_graph=True,
            allow_unused=True,
        )
        grad_emb = torch.autograd.grad(
            loss_emb,
            emb_params,
            retain_graph=True,
            allow_unused=True,
        )
        alpha = (norm(grad_tse) / (norm(grad_emb) + 1e-8)).clamp(0.01, 100.0)
        loss = loss_tse + alpha * loss_emb

        self.log(
            "train/SI-SDR+COS_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
        )
        self.log(
            "train/MRSTFT_loss",
            loss_mrstft,
            on_step=True,
            on_epoch=True,
            prog_bar=True, 
            logger=True,
            batch_size=mix.shape[0],
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mix.shape[0],
        )
        self.log(
            "train/dual_match_loss",
            loss_emb_out["match_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=mix.shape[0],
        )
        self.log(
            "train/dual_slot_repulsion_loss",
            loss_emb_out["slot_repulsion_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=mix.shape[0],
        )
        self.log(
            "train/dual_silence_proto_norm",
            loss_emb_out["silence_proto_norm"],
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=mix.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx, drift=False):
        mix, source, _ = batch
        emb1, emb2 = self._teacher_embeddings(source)
        emb_tgt, target_speech, _, _ = self._select_target(source, emb1, emb2)

        with torch.no_grad():
            _, pred_emb = self._mixture_conditioning_embedding(mix, emb_tgt)
            if drift:
                raise NotImplementedError("Drift modeling not implemented for causal GridNet.")
            estimate = self._run_separator(mix, pred_emb)

        estimate, target_speech = self._trim_pair(estimate, target_speech)
        self.metrics.update(estimate, target_speech)
        return {}

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=True)
        self.metrics.reset()

        if not hasattr(self, "fixed_val_batch"):
            mix, src, _ = next(iter(self.trainer.datamodule.val_dataloader()))
            self.fixed_val_batch = (mix[:5], src[:5])

        mix, src = self.fixed_val_batch
        mix = mix.to(self.device)
        src = src.to(self.device)

        with torch.no_grad():
            emb1 = self.single_sp_model(src[:, 0, :])
            emb2 = self.single_sp_model(src[:, 1, :])
            emb_tgt, tgt, _, _ = self._select_target(src, emb1, emb2, deterministic_idx=None)
            _, pred_emb = self._mixture_conditioning_embedding(mix, emb_tgt)
            pred = self._run_separator(mix, pred_emb)

        pred, tgt = self._trim_pair(pred, tgt)
        mix = mix[..., : pred.shape[-1]]

        run = self.logger.experiment
        for i in range(mix.shape[0]):
            run.log({f"audio/mix_{i}": wandb.Audio(mix[i].detach().cpu().numpy().astype("float32"), sample_rate=16000)})
            run.log({f"audio/tgt_{i}": wandb.Audio(tgt[i].detach().cpu().numpy().astype("float32"), sample_rate=16000)})
            run.log({f"audio/pred_{i}": wandb.Audio(pred[i].detach().cpu().numpy().astype("float32"), sample_rate=16000)})

    def on_test_start(self):
        self.test_metrics = SE_metrics(device="cpu")

    def test_step(self, batch, batch_idx):
        mix, source, _ = batch

        with torch.no_grad():
            emb1, emb2 = self._teacher_embeddings(source)
            emb_tgt, target_speech, _, _ = self._select_target(source, emb1, emb2)

            _, pred_emb = self._mixture_conditioning_embedding(mix, emb_tgt)
            pred = self._run_separator(mix, pred_emb)
            pred, target_speech = self._trim_pair(pred, target_speech)

            self.test_metrics.update(pred, target_speech)

        return {}

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )
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
                "monitor": "train_loss",
                "interval": "epoch",
            },
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from-ckpt", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    DATA_ROOT = "/home/sidcs/datasets/LibriMix/LibriMix"
    HARD_PAIR_ROOT = Path("/home/sidcs/datasets/LibriMix/LibriMix/hard_easy_pairs_teacher_centroid")
    SPEAKER_MAP = str(HARD_PAIR_ROOT / "metadata" / "train_mapping.json")
    TRAIN_META = str(HARD_PAIR_ROOT / "metadata" / "mixture_train.csv")
    VAL_META = str(HARD_PAIR_ROOT / "metadata" / "mixture_val.csv")
    RUN_NAME = "causal_gridnet_joint_training_hardeasypairs_mrstft_0.1"
    SAVE_DIR = Path(f"/home/sidcs/model_ckpts/{RUN_NAME}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=16,
        num_workers=20,
        num_speakers=2,
        train_metadata_path=TRAIN_META,
        val_metadata_path=VAL_META,
        test_metadata_path=VAL_META,
        add_online_noise=False,
    )

    model = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        slot_repulsion_weight=0.1,
        slot_repulsion_margin=0.0,
        speaker_map_path=SPEAKER_MAP,
        model_args=CAUSAL_GRIDNET_ARGS,
    )
    if cli_args.init_from_ckpt:
        model.load_compatible_checkpoint(cli_args.init_from_ckpt)

    wandb_logger = WandbLogger(
        project="causal_gridnet_2sp",
        name=RUN_NAME,
        log_model=False,
        save_dir=str(SAVE_DIR / "wandb_logs"),
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=-1,
        filename="epoch{epoch:02d}-trainloss{train_loss:.3f}",
        dirpath=str(SAVE_DIR),
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0,1,2,3,4,5,6],

        max_epochs=300,
        logger=wandb_logger,
        callbacks=[ckpt],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
    )

    if cli_args.resume_ckpt and cli_args.init_from_ckpt:
        raise ValueError("Use either --resume-ckpt or --init-from-ckpt, not both.")

    trainer.fit(model, datamodule=dm, ckpt_path=cli_args.resume_ckpt)
    # trainer.test(model, datamodule=dm)
    # trainer.fit(model, datamodule=dm, ckpt_path = "/home/sidcs/model_ckpts/causal_gridnet_joint_training_hardpairs_silence_mrstft_0.1/epochepoch=64-trainlosstrain_loss=1.445.ckpt")
    wandb.finish()
