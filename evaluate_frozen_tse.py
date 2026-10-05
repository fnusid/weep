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
import sys
sys.path.append("/home/sidcs/")  # must precede the imports below: models/* import `wesep.*`
from dataset.dataloader import LibriMixDataModule
from models.dpcnn import DPCCN #TSE

from metrics import SE_metrics
import wandb

# from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper  # WavLM teacher
from wavlm_single_embedding.model import ECAPA_TDNN as SingleSpeakerEncoderWrapper
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper 
from wavlm_dual_embedding.loss import LossWraper
import random
random.seed(42)
import warnings
warnings.filterwarnings("ignore")

# torchaudio's ffmpeg StreamReader deprecation warning slips past the blanket
# "ignore" above (some downstream import resets the filter list via
# warnings.simplefilter), so drop it directly at the display step instead.
_default_showwarning = warnings.showwarning
def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    if "StreamingMediaDecoder has been deprecated" in str(message):
        return
    _default_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _filtered_showwarning

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
import auraloss
import yaml
from pathlib import Path
from omegaconf import OmegaConf

config_path = Path("/home/sidcs/wesep/confs/config_dpcnn.yaml")

with config_path.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]

hp = OmegaConf.merge(*docs)

import numpy as np


# import pickle
# with open("/home/sidcs.csegpu1/codebase/wavlm_dual_embedding/cached_embeddings.pkl", "rb") as f:
#     CACHED_EMBS = pickle.load(f)

def norm(grads):
    # grads can contain None if allow_unused=True
    s = 0.0
    for g in grads:
        if g is None: 
            continue
        s = s + (g.detach()**2).sum()
    return torch.sqrt(s + 1e-12)

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
        speaker_map_path: str = "/tmp/sidcs/turbo/sidcs_backup/datasets/LibriMix/LibriMix/Libriuni_03_08/Libri2Mix_ovl30to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)

        device="cuda" if torch.cuda.is_available() else "cpu"   
   
        #Get the dual-emb model and teacher model
        
        # dual_emb_ckpt_path = "/tmp/sidcs/turbo/sidcs_backup/model_ckpts/CAUSAL_ECAPA_UNMIX_2048_teacher_ECAPA/best-epoch=70-loss=0.1346.ckpt"
        # dual_emb_ckpt_path = "/tmp/sidcs/turbo/sidcs_backup/model_ckpts/CAUSAL_ECAPA_UNMIX_2048_teacher_ECAPA_redo/best-epoch=48-loss=0.1658.ckpt"
        # dual_emb_ckpt_path="/tmp/sidcs/turbo/sidcs_backup/model_ckpts/ECAPA_UNMIX_2048_teacher_ECAPA//best-epoch=74-val_separation=0.000.ckpt"
        #relational
        dual_emb_ckpt_path = "/tmp/sidcs/turbo/sidcs_backup/model_ckpts/ECAPA_UNMIX_2048_teacher_ECAPA_relation_0.1/best-epoch=49-val_separation=0.000.ckpt"
        #relational 1 (hinge)
        #dual_emb_ckpt_path = "/tmp/sidcs/turbo/sidcs_backup/model_ckpts/ECAPA_UNMIX_2048_teacher_ECAPA_relation_1_hinge/best-epoch=57-val_separation=0.000.ckpt"
        # dual_emb_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/best-epoch=12-val_separation=0.000.ckpt"
        # dual_emb_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_3spft_wavlm_linear_dualemb_tr360/best-epoch=54-val_separation=0.000.ckpt"
        dual_emb_ckpt = torch.load(dual_emb_ckpt_path, map_location=device)
        state = strip_dual_model_weights(dual_emb_ckpt["state_dict"])
        # self.dual_emb_model = SpeakerEncoderDualWrapper(emb_dim=emb_dim, finetune_wavlm=True) #joint training 
        self.dual_emb_model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
        self.dual_emb_model.load_state_dict(state, strict=True)

        # linear_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_linearmappingtotr/best-epoch=47-val_separation=0.000.ckpt"

        # self.linear_layer = nn.Linear(256, 256).to("cuda")
        # linear_ckpt = torch.load(linear_ckpt_path, map_location="cuda")
        # linear_state = linear_ckpt["state_dict"]
        # linear_sd = {k[len("linear_map."):]: v for k, v in linear_state.items() if k.startswith("linear_map.")}
        # self.linear_layer.load_state_dict(linear_sd)

        # mlp_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_mlpmappingtotr/val_monitored/best-epoch=9-val_separation=0.000.ckpt"
        # self.mlp_layer = nn.Sequential(nn.Linear(256, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 256)).to("cuda")

        # mlp_ckpt = torch.load(mlp_ckpt_path, map_location="cuda")
        # mlp_state = mlp_ckpt["state_dict"]
        # mlp_sd = {k[len("mlp."):]: v for k, v in mlp_state.items() if k.startswith("mlp.")}
        # self.mlp_layer.load_state_dict(mlp_sd)


        # mlp_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_mlpmappingtotr/linear_init/best-epoch=1-val_separation=0.000.ckpt"

        # self.mlp_layer = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),  # No final ReLU
        # )

        # ckpt = torch.load(
        #     mlp_ckpt_path,
        #     map_location="cpu",
        #     weights_only=True,
        # )

        # # Training used self.mlp, so checkpoint keys start with "mlp."
        # mlp_state = {
        #     key[len("mlp."):]: value
        #     for key, value in ckpt["state_dict"].items()
        #     if key.startswith("mlp.")
        # }

        # self.mlp_layer.load_state_dict(mlp_state, strict=True)
        # self.mlp_layer.eval()
        # self.mlp_layer.requires_grad_(False)

        # print("Loaded trained MLP:", mlp_ckpt_path)

        self.dual_emb_loss = LossWraper()
        self.num_mixtures = 0
        self.num_both_assigned = 0
        # Sums of per-mixture PIT SI-SDR (mean of the two outputs), split by
        # whether the two outputs were assigned to distinct or the same speaker.
        self.sisdr_sum_distinct = 0.0
        self.sisdr_sum_duplicate = 0.0
        # Per-mixture embedding cosine similarities, split by assignment
        # outcome. Each entry holds [sum, sum_of_squares, count].
        self.emb_sim_keys = [
            "student_e1_vs_e2",
            "head1_vs_own_teacher",
            "head2_vs_own_teacher",
            "teacher_t1_vs_t2",
        ]
        self.emb_sim_groups = ["distinct", "duplicate", "all"]
        self._reset_emb_sim_acc()


        # self.dual_emb_model.to(device).eval()
        # for param in self.dual_emb_model.parameters():
        #     param.requires_grad = False

        # self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)  # WavLM teacher
        self.single_sp_model = SingleSpeakerEncoderWrapper(C=1024)  # ECAPA teacher, 256-d output
        teacher_ckpt_path = "/tmp/sidcs/turbo/sidcs_backup/model_ckpts/ecapa_tdnn_arcface_tr360/best-epoch=30-val_separation=0.000.ckpt"
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        state = ckpt["state_dict"]

        filtered = {}
        for k, v in state.items():
            # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
            if k.startswith("model.") and ("arcface" not in k):
                filtered[k.replace("model.", "", 1)] = v

        print("Loaded teacher keys:", len(filtered))
        self.wav_pit = "True"
        self.use_cached_embs = "False"
        self.sisdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
        self.single_sp_model.load_state_dict(filtered, strict=True)
        self.single_sp_model.eval()
        for param in self.single_sp_model.parameters():
            param.requires_grad = False



        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = SE_metrics(device="cpu")  # will overwrite device at runtime
        # self.metrics = SE_metrics(device='cuda', use_only_sisdr=True)

        self.model = DPCCN(**hp.model_args.tse_model)
        self.loss = auraloss.time.SISDRLoss()

        self.emb_scaling = {f"{i}":[] for i in [0.5, 1, 1.5, 2, 2.5, 3, 4]}



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

        # cosine1 = cosine(e1, emb_tgt)
        # cosine2 = cosine(e2, emb_tgt)
        # choose_mask = (cosine1 > cosine2).unsqueeze(-1)   # [B,1]
        # pred_emb = torch.where(choose_mask, e1, e2)

        
        #condition dccrn on pred_emb
        #convert mix to spec
      
        out,_ = self.forward(mix, emb = pred_emb) 

        min_len = min(out.shape[-1], source.shape[-1])
        out = out[..., :min_len]

        source = target_speech[..., :min_len]
        loss_tse = self.loss(out, source)
        # out_wav = self.audio_utils.spec2wav(out.detach().numpy(), mix_phase)
        #adjust weights accordingly
        # breakpoint()
        # loss = loss_tse + loss_emb

    


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


    def validation_step(self, batch, batch_idx, drift=False):
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
            other_emb_tgt = emb2
            #true
            target_speech = source[:, 0, :]
            other_speech = source[:, 1, :]
            #reversed
            # emb_tgt = emb2

        else:
            emb_tgt = emb2
            other_emb_tgt = emb1
            #true
            target_speech = source[:, 1, :] #[B, T]
            other_speech = source[:, 0, :]
            #reversed
            # emb_tgt = emb1

        with torch.no_grad():
            embs = self.dual_emb_model(mix)# [B, 2, emb_dim]
            e1 = embs[:, 0, :]
            e2 = embs[:, 1, :]

        cosine1 = cosine(e1, emb_tgt)
        cosine2 = cosine(e2, emb_tgt)
        choose_mask = (cosine1 > cosine2).unsqueeze(-1)   # [B,1]
        pred_emb = torch.where(choose_mask, e1, e2)

        cosine11 =  cosine(e1, other_emb_tgt)
        cosine22 = cosine(e2, other_emb_tgt)
        choose_mask_other = (cosine11 > cosine22).unsqueeze(-1)   # [B,1]
        pred_emb2 = torch.where(choose_mask_other, e1, e2)
        #condition dccrn on pred_emb
        if drift != True:
            out,_ = self.forward(mix, emb = pred_emb) #list of three wavs
        # out = out[0]
        else:
            combinations = [[1,0],[0.75, 0.25], [0.5,0.5], [0.25,0.75],[0,1]]
            metrics = {}
            # metrics[i for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']] = []
            # for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
            for i in ['SI_SDR']:
                metrics[i] = []
            # breakpoint()
            for comb in combinations:
                weight1 = comb[0]
                weight2 = comb[1]
                drifted_emb = weight1 * pred_emb + weight2 * pred_emb2
                out,_ = self.forward(mix, emb = drifted_emb) #list of three wavs
                min_len = min(out.shape[-1], source.shape[-1])
                out = out[..., :min_len]
                target_speech = target_speech[..., :min_len]
                other_speech = other_speech[..., :min_len]
                self.metrics.update(out, target_speech)
                met1 = self.metrics.compute()
                self.metrics.reset()
                self.metrics.update(out, other_speech)
                met2 = self.metrics.compute()
                self.metrics.reset()
                for k in metrics.keys():
                    metrics[k].append((met1[k], met2[k]))

                '''
                return {
                "PESQ": float(torch.tensor(self.pesq_scores).nanmean()),
                "STOI": float(torch.tensor(self.stoi_scores).nanmean()),
                "SI_SDR": float(torch.tensor(self.sisdr_scores).nanmean()),
                "SIG": float(torch.tensor(self.SIG).nanmean()),
                "BAK": float(torch.tensor(self.BAK).nanmean()),
                "OVRL": float(torch.tensor(self.OVRL).nanmean()),
                }
                '''
            return metrics


        

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
        # self.test_metrics = SE_metrics(device="cpu")
        self.test_metrics = SE_metrics(device="cpu")
        # self.num_mixtures = 0
        # self.num_both_assigned = 0

        # direct_id_path = (
        #     "/home/sidcs.csegpu1/model_ckpts/"
        #     "ft_wavlm_linear_dualemb_noteacher_tr360/"
        #     "best-epoch=12-val_separation=0.000.ckpt"
        # )
        # ckpt = torch.load(direct_id_path, map_location="cpu")
        # expected = strip_dual_model_weights(ckpt["state_dict"])
        # actual = self.dual_emb_model.state_dict()

        # assert expected.keys() == actual.keys(), "Encoder keys differ"

        # changed = [
        #     k for k in expected
        #     if not torch.equal(
        #         expected[k].cpu(),
        #         actual[k].detach().cpu().to(expected[k].dtype),
        #     )
        # ]
        # assert not changed, (
        #     f"Direct-ID encoder was changed! Example keys: {changed[:5]}"
        # )

        # assert not isinstance(self.mlp_layer[-1], nn.ReLU)
        # print("Verified intended direct-ID encoder and no final ReLU")

    def _build_cached_emb_index(self):
        """Flatten CACHED_EMBS ({stem_id: {"embs": [...], "labels": [...]}})
        into parallel arrays once, for fast distractor sampling."""
        all_embs, all_labels = [], []
        for stem_id, d in CACHED_EMBS.items():
            for emb, lab in zip(d["embs"], d["labels"]):
                all_embs.append(emb)
                all_labels.append(lab)
        self._cached_embs_arr = np.stack(all_embs, axis=0)     # [N, D]
        self._cached_labels_arr = np.array(all_labels)         # [N]

    def sample_distractor_embs(self, labels):
        """
        labels: [B, 2]
        Returns: [B, 2, D], one embedding pair per donor mixture.
        Both donor speakers must be absent from the current mixture.
        """
        labels_np = (
            labels.detach().cpu().numpy()
            if torch.is_tensor(labels)
            else np.asarray(labels)
        )

        pairs = []

        for item_labels in labels_np:
            present = set(item_labels.tolist())

            eligible = [
                d for d in CACHED_EMBS.values()
                if len(d["embs"]) == 2
                and len(d["labels"]) == 2
                and len(set(d["labels"])) == 2
                and present.isdisjoint(set(d["labels"]))
            ]

            if not eligible:
                raise ValueError(
                    f"No eligible donor mixture for speakers {present}"
                )

            donor = random.choice(eligible)

            pair = torch.stack([
                torch.as_tensor(
                    emb, dtype=torch.float32, device=self.device
                ).detach()
                for emb in donor["embs"]
            ])  # [2, D]

            pairs.append(pair)

        return torch.stack(pairs)  # [B, 2, D]


    def test_step(self, batch, batch_idx):

        mix, source, labels = batch  # mix: [B,T], source: [B,2,T]

        # Sample a distractor embedding per batch item, drawn from CACHED_EMBS,
        # from a speaker NOT present in that item's mixture (labels[i]).
        # distractor_embs = self.sample_distractor_embs(labels)  # [B, D]

        # --- teacher embeddings from clean sources ---
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B,D]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B,D]

            # --- dual embeddings from mixture (unordered) ---
            embs = self.dual_emb_model(mix)               # [B,2,D]
            e1 = embs[:, 0, :]
            # e1 = self.mlp_layer(e1)
            # e1 = self.linear_layer(e1)
            e2 = embs[:, 1, :]
            # e2 = self.mlp_layer(e2)
            # e2 = self.linear_layer(e2)

            # Evaluate BOTH targets for each mixture
            # idx = np.random.choice([0, 1])
            tgt1 = source[:, 0, :]  # [B,T]
            tgt2 = source[:, 1, :]  # [B,T]

            if self.use_cached_embs =="True":
                # Generate BOTH outputs using student embeddings.
                dist_e1 = distractor_embs[:, 0, :]
                dist_e2 = distractor_embs[:, 1, :]
                out1 = self.forward(mix, emb=dist_e1)[0]
                out2 = self.forward(mix, emb=dist_e2)[0]


                min_len = min(
                    out1.shape[-1], out2.shape[-1],
                    tgt1.shape[-1], tgt2.shape[-1],
                )
                out1 = out1[..., :min_len]
                out2 = out2[..., :min_len]
                tgt1 = tgt1[..., :min_len]
                tgt2 = tgt2[..., :min_len]

                # Per-example scores [B], not batch-averaged scalars.
                # Match zero_mean to your existing SI-SDR metric's setting.
                s11 = si_sdr(out1.float(), tgt1.float(), zero_mean=False)
                s12 = si_sdr(out1.float(), tgt2.float(), zero_mean=False)
                s21 = si_sdr(out2.float(), tgt1.float(), zero_mean=False)
                s22 = si_sdr(out2.float(), tgt2.float(), zero_mean=False)

                # One-to-one waveform PIT, separately for each mixture.
                swap = ((s12 + s21) > (s11 + s22)).unsqueeze(-1)  # [B,1]

                matched_tgt1 = torch.where(swap, tgt2, tgt1)
                matched_tgt2 = torch.where(swap, tgt1, tgt2)

                self.test_metrics.update(out1, matched_tgt1)
                self.test_metrics.update(out2, matched_tgt2)

                # Independent matching ONLY for coverage diagnostics.
                # Ties consistently select reference 1.
                out1_prefers_tgt2 = s12 > s11
                out2_prefers_tgt2 = s22 > s21
                both_assigned = out1_prefers_tgt2 != out2_prefers_tgt2

                self.num_both_assigned += both_assigned.sum().item()
                self.num_mixtures += mix.shape[0]

                matched_s1 = torch.where(swap.squeeze(-1), s12, s11)
                matched_s2 = torch.where(swap.squeeze(-1), s21, s22)
                self._accumulate_assignment_sisdr(
                    matched_s1, matched_s2, both_assigned
                )
            if self.wav_pit == 'True':
                # Generate BOTH outputs using student embeddings, scaled by
                # each factor in self.emb_scaling, and score each scale.
                for scale_str in self.emb_scaling.keys():
                    scale = float(scale_str)
                    e1_scaled = e1 * scale
                    e2_scaled = e2 * scale

                    out1 = self.forward(mix, emb=e1_scaled)[0]
                    out2 = self.forward(mix, emb=e2_scaled)[0]

                    min_len = min(
                        out1.shape[-1], out2.shape[-1],
                        tgt1.shape[-1], tgt2.shape[-1],
                    )
                    out1_t = out1[..., :min_len]
                    out2_t = out2[..., :min_len]
                    tgt1_t = tgt1[..., :min_len]
                    tgt2_t = tgt2[..., :min_len]

                    # Per-example scores [B], not batch-averaged scalars.
                    # Match zero_mean to your existing SI-SDR metric's setting.
                    s11 = si_sdr(out1_t.float(), tgt1_t.float(), zero_mean=False)
                    s12 = si_sdr(out1_t.float(), tgt2_t.float(), zero_mean=False)
                    s21 = si_sdr(out2_t.float(), tgt1_t.float(), zero_mean=False)
                    s22 = si_sdr(out2_t.float(), tgt2_t.float(), zero_mean=False)

                    # One-to-one waveform PIT, separately for each mixture.
                    swap = (s12 + s21) > (s11 + s22)  # [B]

                    matched_s1 = torch.where(swap, s12, s11)
                    matched_s2 = torch.where(swap, s21, s22)
                    self.emb_scaling[scale_str].extend(matched_s1.tolist())
                    self.emb_scaling[scale_str].extend(matched_s2.tolist())

                    if scale == 1.0:
                        swap_b = swap.unsqueeze(-1)  # [B,1]
                        matched_tgt1 = torch.where(swap_b, tgt2_t, tgt1_t)
                        matched_tgt2 = torch.where(swap_b, tgt1_t, tgt2_t)

                        self.test_metrics.update(out1_t, matched_tgt1)
                        self.test_metrics.update(out2_t, matched_tgt2)

                        # Independent matching ONLY for coverage diagnostics.
                        # Ties consistently select reference 1.
                        out1_prefers_tgt2 = s12 > s11
                        out2_prefers_tgt2 = s22 > s21
                        both_assigned = out1_prefers_tgt2 != out2_prefers_tgt2

                        self.num_both_assigned += both_assigned.sum().item()
                        self.num_mixtures += mix.shape[0]

                        self._accumulate_assignment_sisdr(
                            matched_s1, matched_s2, both_assigned
                        )

                        # "Own" teacher of each head = the source its output
                        # was matched to by the one-to-one waveform PIT.
                        own_t1 = torch.where(swap_b, emb2, emb1)
                        own_t2 = torch.where(swap_b, emb1, emb2)
                        self._accumulate_emb_sims({
                            "student_e1_vs_e2": cosine(e1, e2),
                            "head1_vs_own_teacher": cosine(e1, own_t1),
                            "head2_vs_own_teacher": cosine(e2, own_t2),
                            "teacher_t1_vs_t2": cosine(emb1, emb2),
                        }, both_assigned)

            else:

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

    def _accumulate_assignment_sisdr(self, matched_s1, matched_s2, both_assigned):
        """Add per-mixture PIT SI-SDR (mean of both outputs) to the
        distinct- or duplicate-assignment bucket."""
        mix_sisdr = 0.5 * (matched_s1 + matched_s2)  # [B]
        self.sisdr_sum_distinct += mix_sisdr[both_assigned].sum().item()
        self.sisdr_sum_duplicate += mix_sisdr[~both_assigned].sum().item()

    def _reset_emb_sim_acc(self):
        self.emb_sim_acc = {
            k: {g: [0.0, 0.0, 0] for g in self.emb_sim_groups}
            for k in self.emb_sim_keys
        }

    def _accumulate_emb_sims(self, sims, both_assigned):
        """sims: {key: [B] cosine tensor}; both_assigned: [B] bool."""
        masks = {
            "distinct": both_assigned,
            "duplicate": ~both_assigned,
            "all": torch.ones_like(both_assigned),
        }
        for k, v in sims.items():
            v = v.detach().float()
            for g, m in masks.items():
                acc = self.emb_sim_acc[k][g]
                acc[0] += v[m].sum().item()
                acc[1] += (v[m] ** 2).sum().item()
                acc[2] += int(m.sum().item())

    def _ddp_sum(self, value):
        """Sum a scalar across all DDP processes (no-op if not distributed)."""
        t = torch.tensor(float(value), device=self.device)
        return self.all_gather(t).sum().item()

    def on_test_epoch_end(self):
        # Aggregate coverage-diagnostic counters across all GPUs.
        total_mixtures = self._ddp_sum(self.num_mixtures)
        total_both_assigned = self._ddp_sum(self.num_both_assigned)
        total_sisdr_distinct = self._ddp_sum(self.sisdr_sum_distinct)
        total_sisdr_duplicate = self._ddp_sum(self.sisdr_sum_duplicate)
        self.num_mixtures = 0
        self.num_both_assigned = 0
        self.sisdr_sum_distinct = 0.0
        self.sisdr_sum_duplicate = 0.0

        # Aggregate embedding-similarity stats across all GPUs (fixed key
        # order so every rank issues the same all_gather sequence).
        emb_sim_results = {}
        for k in self.emb_sim_keys:
            for g in self.emb_sim_groups:
                s, sq, n = self.emb_sim_acc[k][g]
                s, sq, n = self._ddp_sum(s), self._ddp_sum(sq), self._ddp_sum(n)
                if n > 0:
                    mean = s / n
                    std = max(sq / n - mean ** 2, 0.0) ** 0.5
                    emb_sim_results[(k, g)] = (mean, std, int(n))
        self._reset_emb_sim_acc()

        m = self.test_metrics.compute()
        for k, v in m.items():
            self.log(f"test/{k}", v, prog_bar=True)
        self.test_metrics.reset()

        # Aggregate per-scale SI-SDR sums/counts across all GPUs.
        scale_results = {}
        for scale_str, scores in self.emb_scaling.items():
            total_sum = self._ddp_sum(sum(scores))
            total_count = self._ddp_sum(len(scores))
            scores.clear()
            if total_count > 0:
                mean_score = total_sum / total_count
                scale_results[scale_str] = (mean_score, int(total_count))
                self.log(f"test/si-sdr_scale_{scale_str}", mean_score, prog_bar=False)

        if self.trainer.is_global_zero:
            if total_mixtures > 0:
                both_rate = total_both_assigned / total_mixtures
                print("Both-speaker assignment rate:", both_rate)
                print("Duplicate-assignment rate:", 1.0 - both_rate)
                print("Mean assignment coverage:", 0.5 + 0.5 * both_rate)

                total_duplicate = total_mixtures - total_both_assigned
                if total_both_assigned > 0:
                    print(
                        f"SI-SDR (distinct assignment): "
                        f"{total_sisdr_distinct / total_both_assigned:.4f} dB "
                        f"(n={int(total_both_assigned)} mixtures)"
                    )
                if total_duplicate > 0:
                    print(
                        f"SI-SDR (duplicate assignment): "
                        f"{total_sisdr_duplicate / total_duplicate:.4f} dB "
                        f"(n={int(total_duplicate)} mixtures)"
                    )
                print(
                    f"SI-SDR (overall): "
                    f"{(total_sisdr_distinct + total_sisdr_duplicate) / total_mixtures:.4f} dB "
                    f"(n={int(total_mixtures)} mixtures)"
                )

            print("Embedding cosine similarity (mean ± std):")
            print(f"  {'':<24}" + "".join(f"{g:>26}" for g in self.emb_sim_groups))
            for k in self.emb_sim_keys:
                row = f"  {k:<24}"
                for g in self.emb_sim_groups:
                    if (k, g) in emb_sim_results:
                        mean, std, n = emb_sim_results[(k, g)]
                        row += f"{f'{mean:.4f} ± {std:.4f} (n={n})':>26}"
                    else:
                        row += f"{'n/a':>26}"
                print(row)

            print("Embedding-scaling SI-SDR results:")
            for scale_str, (mean_score, n) in scale_results.items():
                print(f"  scale={scale_str}: SI-SDR={mean_score:.4f} dB (n={n})")
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
    # One DDP process per SLURM task (one GPU each); fall back to 1 when run outside SLURM
    NUM_GPUS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    NUM_NODES = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    DATA_ROOT = "/tmp/sidcs/turbo/sidcs_backup/datasets/LibriMix/LibriMix"
    SPEAKER_MAP = "/tmp/sidcs/turbo/sidcs_backup/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    # SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/3sp/Libri3Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"

    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=8, 
        num_workers=CPUS_PER_TASK - 1,  # per GPU process; leave one core for the main process
        num_speakers=2
    )

    model = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )

    # wandb_logger = WandbLogger(
    #     project="pDCCRN_2sp",
    #     name="pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss",
    #     # name='test_run',
    #     log_model=False,
    #     save_dir="/home/sidcs.csegpu1/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/wandb_logs",
    # )
    wandb_logger=None
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/SI-SDR_loss",
        mode="min",
        save_top_k=-1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/tmp/sidcs/turbo/sidcs_backup/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/"
    )

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator="gpu",
        # precision="16-mixed",    # <-- mixed precision
        # devices=[0, 1, 2, 3,4,5,6,7],
        devices=NUM_GPUS,
        num_nodes=NUM_NODES,
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
    model.strict_loading=False
    trainer.test(model, datamodule=dm, ckpt_path="/tmp/sidcs/turbo/sidcs_backup/model_ckpts/pDCCRN_2sp_teacher_emb_ECAPA_normalised/best-epoch=75-val_separation=0.000.ckpt")
    # trainer.test(model, datamodule=dm, ckpt_path="/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt")
    # trainer.test(model, datamodule=dm, ckpt_path="/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/best-epoch=12-val_separation=0.000.ckpt")
    # trainer.validate(model, datamodule=dm, ckpt_path = "/mnt/disks/data/model_ckpts/archive_ckpt/pFCCRN_2sp/best-epoch=60-val_separation=0.000.ckpt")
    wandb.finish()
