#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

from dataset.dataloader import LibriMixDataModule    
from train_dpcnn import E2EpSE        # your Lightning module
from metrics import SE_metrics  # same class you use in E2EpSE
import sys
sys.path.append('/home/sidharth./codebase')
from wavlm_dual_embedding.eval_metrics import compute_clustering_metrics, load_dual_model
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def save_consolidated_npz(consolidated_metr_spk1, consolidated_metr_spk2, out_dir="drift_plots"):
    os.makedirs(out_dir, exist_ok=True)

    pack = {}
    # for name in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
    for name in ['SI_SDR']:
        m1, s1 = consolidated_metr_spk1[name]
        m2, s2 = consolidated_metr_spk2[name]
        pack[f"{name}_spk1_mean"] = m1.detach().cpu().numpy()
        pack[f"{name}_spk1_std"]  = s1.detach().cpu().numpy()
        pack[f"{name}_spk2_mean"] = m2.detach().cpu().numpy()
        pack[f"{name}_spk2_std"]  = s2.detach().cpu().numpy()

    save_path = os.path.join(out_dir, "drift_metrics_consolidated.npz")
    np.savez(save_path, **pack)
    print("Saved:", save_path)

def plot_consolidated_metrics_both(
    consolidated_metr_spk1,
    consolidated_metr_spk2,
    out_dir="drift_plots",
    title_prefix="PSE drift",
    align_direction=False,   # reverse spk2 so both plot as speaker1 -> speaker2
):
    os.makedirs(out_dir, exist_ok=True)
    # metric_names = ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']
    metric_names = ['SI_SDR']
    # infer n from first metric
    mean0, _ = consolidated_metr_spk1[metric_names[0]]
    n = int(mean0.numel())
    x = np.arange(n)

    fracs = np.linspace(0.0, 1.0, n) if n >= 2 else np.array([0.0])
    xlabels = []
    for i, f in enumerate(fracs):
        if i == 0:
            xlabels.append("speaker1")
        elif i == n - 1:
            xlabels.append("speaker2")
        else:
            xlabels.append(f"{f:.2f}")

    for name in metric_names:
        # --- speaker1-target curve ---
        mean1_t, std1_t = consolidated_metr_spk1[name]
        mean1 = mean1_t.detach().cpu().numpy()
        std1  = std1_t.detach().cpu().numpy()

        # --- speaker2-target curve ---
        mean2_t, std2_t = consolidated_metr_spk2[name]
        mean2 = mean2_t.detach().cpu().numpy()
        std2  = std2_t.detach().cpu().numpy()

        # Optionally align both curves to same drift direction (speaker1 -> speaker2)
        # If spk2 dict is computed as (speaker2 -> speaker1), reverse it.
        if align_direction:
            mean2 = mean2[::-1]
            std2  = std2[::-1]

        plt.figure()
        plt.errorbar(x, mean1, yerr=std1, marker="o", capsize=4, label="Target = speaker1")
        plt.errorbar(x, mean2, yerr=std2, marker="o", capsize=4, label="Target = speaker2")

        plt.xticks(x, xlabels)
        plt.xlabel("Embedding drift (speaker1 → speaker2)")
        plt.ylabel(name)
        plt.title(f"{title_prefix}: {name} (mean ± std)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"{name}_drift_both.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to: {out_dir}")

# call

def main():
    # ------------------------
    # 1) Config
    # ------------------------
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = (
        "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/"
        "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    )

    ckpt_path = (
        "/mnt/disks/data/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt"
    )

    embedding_ckpt = "/mnt/disks/data/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"


    batch_size = 16
    num_workers = 20
    num_speakers = 2
    sample_rate = 16000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    emb_model = load_dual_model(embedding_ckpt, emb_dim=256, device=device)
    emb_model.eval()

    # ------------------------
    # 2) DataModule & loader
    # ------------------------
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

    # ------------------------
    # 3) Build model & load checkpoint
    # ------------------------
    print(f"Loading E2EpSE checkpoint from: {ckpt_path}")
    system = E2EpSE(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=speaker_map_path,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    # breakpoint()
    system.load_state_dict(ckpt["state_dict"], strict=True)
    system.to(device)
    system.eval()

    # *** IMPORTANT: use the SAME metrics object & validation_step logic ***
    system.metrics = SE_metrics(fs=sample_rate, device='cuda', use_only_sisdr=True)  # rebuild metrics exactly once
    # system.metrics.to(device)
    system.metrics.reset()

    total_batches = len(val_loader)
    print(f"Evaluating PSE using model.validation_step() on {total_batches} batches ...")
    metr_spk1 = {}
    metr_spk2 = {}
    # for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
    for i in ['SI_SDR']:
        metr_spk1[i] = []
        metr_spk2[i] = []

    consolidated_metr_spk1 = {}
    consolidated_metr_spk2 = {}
    # for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
    for i in ['SI_SDR']:
        consolidated_metr_spk1[i] = []
        consolidated_metr_spk2[i] = []
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=total_batches, desc="Evaluating (drift)", leave=True)
        for batch_idx, batch in pbar:
            # move batch to device the same way Lightning does
            mix, src, labels = batch
            mix = mix.to(device)
            src = src.to(device)
            labels = labels.to(device)
            # breakpoint()
            ##MODIFY THE VALIDATION STEP TO MODEL THE DRIFT, ADD A PARAMETER TO THE FUNCTION.
            # reuse the exact validation code from E2EpSE
            metrics = system.validation_step((mix, src, labels), batch_idx, drift=True)
            # breakpoint()
            # for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
            for i in ['SI_SDR']:

                metr_spk1[i].append([item[0] for item in metrics[i]])
                metr_spk2[i].append([item[1] for item in metrics[i]])

            
            # if (batch_idx + 1) % 10 == 0:
            #     print(f"  Processed {batch_idx+1}/{total_batches} batches")



    # for i in ['PESQ','STOI','SI_SDR','SIG','BAK','OVRL']:
    for i in ['SI_SDR']:
        '''
        metr[i] is a list of list : [[x1,y1,z1,t1], [x2,y2,z2,t2], ...  ]
        stack and take mean over 0th axis : [(mean_x, std_x), (mean_y, std_y), (mean_z, std_z), (mean_t, std_t)]
        '''
        metr_spk1[i] = torch.tensor(metr_spk1[i])
        metr_spk2[i] = torch.tensor(metr_spk2[i])
        mean_spk1 = torch.mean(metr_spk1[i], dim=0)
        std_spk1 = torch.std(metr_spk1[i], dim=0)
        mean_spk2 = torch.mean(metr_spk2[i], dim=0)
        std_spk2 = torch.std(metr_spk2[i], dim=0)

        consolidated_metr_spk1[i] = (mean_spk1, std_spk1)
        consolidated_metr_spk2[i] = (mean_spk2, std_spk2)
        
        
 
    #save the metrics 
    save_consolidated_npz(consolidated_metr_spk1, consolidated_metr_spk2, out_dir="drift_plots")
    plot_consolidated_metrics_both(consolidated_metr_spk1, consolidated_metr_spk2, out_dir="drift_plots", align_direction=False)

        
        
    

    #plot the individual metrics with x axis as speaker 1 and speaker 2




if __name__ == "__main__":
    main()