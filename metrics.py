import torch
import torch.nn as nn
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore


class SE_metrics(nn.Module):
    """
    Computes: PESQ, STOI, SI-SDR, DNSMOS(personalized)
    Accumulates results and returns per-epoch averages.
    """
    def __init__(self, fs=16000, device="cpu"):
        super().__init__()

        self.fs = fs
        self.device = device

        # --- Initialize metrics ---
        self.pesq_metric = PerceptualEvaluationSpeechQuality(
            fs=fs, mode="wb"  # wideband 16k
        )

        self.stoi_metric = ShortTimeObjectiveIntelligibility(
            fs=fs, extended=False
        )

        self.sisdr_metric = ScaleInvariantSignalDistortionRatio()

        self.dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(
            fs=16000,
            personalized=False, #turn this thing False for the final metric
            device=self.device,
            num_threads=4,
        )
        # --- storage for epoch ---
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.pesq_scores = []
        self.stoi_scores = []
        self.sisdr_scores = []
        self.SIG = []
        self.BAK = []
        self.OVRL = []

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, pred_audio, target_audio):
        """
        pred_audio:  [B, T]
        target_audio:[B, T]
        """

        # Move to CPU for PESQ / DNSMOS
        pred = pred_audio.detach().cpu()
        tgt  = target_audio.detach().cpu()

        B = pred.shape[0]

        for b in range(B):
            p = pred[b].unsqueeze(0)
            t = tgt[b].unsqueeze(0)
       
            # PESQ
            try:
                pesq_val = self.pesq_metric(p, t).item()
            except Exception:
                pesq_val = float("nan")

            # STOI
            try:
                stoi_val = self.stoi_metric(p, t).item()
            except Exception:
                stoi_val = float("nan")

            # SI-SDR
            try:
                sisdr_val = self.sisdr_metric(p, t).item()
            except Exception:
                sisdr_val = float("nan")

            # DNSMOS (ONNX-based)

            # dnsmos_val = self.dnsmos_metric(p, t)["ovrl_mos"]

            dns_mos_scores = self.dnsmos_metric(p) #[p808_mos, mos_sig, mos_bak, mos_ovr]
            SIG = dns_mos_scores[1]
            BAK = dns_mos_scores[2]
            OVRL = dns_mos_scores[-1]
                

            # push to buffers
            self.pesq_scores.append(pesq_val)
            self.stoi_scores.append(stoi_val)
            self.sisdr_scores.append(sisdr_val)
            self.SIG.append(SIG)
            self.BAK.append(BAK)
            self.OVRL.append(OVRL)
            # self.dnsmos_scores.append(dnsmos_val)

    # ------------------------------------------------------------------
    def compute(self):
        """Return mean of all metrics."""
        return {
            "PESQ": float(torch.tensor(self.pesq_scores).nanmean()),
            "STOI": float(torch.tensor(self.stoi_scores).nanmean()),
            "SI_SDR": float(torch.tensor(self.sisdr_scores).nanmean()),
            "SIG": float(torch.tensor(self.SIG).nanmean()),
            "BAK": float(torch.tensor(self.BAK).nanmean()),
            "OVRL": float(torch.tensor(self.OVRL).nanmean()),
        }

    # ------------------------------------------------------------------
    def compute(self):
        """Return mean of all metrics."""
        return {
            "PESQ": float(torch.tensor(self.pesq_scores).nanmean()),
            "STOI": float(torch.tensor(self.stoi_scores).nanmean()),
            "SI_SDR": float(torch.tensor(self.sisdr_scores).nanmean()),
            "SIG": float(torch.tensor(self.SIG).nanmean()),
            "BAK": float(torch.tensor(self.BAK).nanmean()),
            "OVRL": float(torch.tensor(self.OVRL).nanmean()),
        }