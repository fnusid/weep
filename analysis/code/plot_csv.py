import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/sidcs/codebase/wesep/analysis/chunked_tradeoff_sisdr.csv"
df = pd.read_csv(csv_path)

print(df.head())

# Plot SI-SDRi vs chunk size for each embedding context
for update_sec in sorted(df["emb_update_interval_sec"].unique()):
    sub = df[df["emb_update_interval_sec"] == update_sec]

    plt.figure(figsize=(8, 5))

    for emb_sec in sorted(sub["emb_context_sec"].unique()):
        s = sub[sub["emb_context_sec"] == emb_sec].sort_values("chunk_ms")
        plt.plot(
            s["chunk_ms"],
            s["SI_SDRi"],
            marker="o",
            label=f"emb ctx = {emb_sec}s",
        )

    plt.xlabel("TSE chunk size (ms)")
    plt.ylabel("SI-SDRi (dB)")
    plt.title(f"SI-SDRi vs Chunk Size | Emb update every {update_sec}s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = f"/home/sidcs/codebase/wesep/analysis/sisdri_tradeoff_update_{update_sec}s.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("saved:", out_path)