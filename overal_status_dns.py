import json
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, required=False, default = "/home/sidharth./codebase/wesep/dns_p835_out_tasnet/per_file.json")
    args = ap.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    def get_vals(key1, key2):
        vals = []
        for r in data:
            v = r.get(key1, {}).get(key2, None)
            if v is None:
                continue
            vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    noisy_sig  = get_vals("noisy", "SIG")
    noisy_bak  = get_vals("noisy", "BAK")
    noisy_ovrl = get_vals("noisy", "OVRL")

    enh_sig  = get_vals("enh", "SIG")
    enh_bak  = get_vals("enh", "BAK")
    enh_ovrl = get_vals("enh", "OVRL")

    def summarize(name, arr):
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
        }

    out = {
        "noisy_SIG":  summarize("noisy_SIG", noisy_sig),
        "noisy_BAK":  summarize("noisy_BAK", noisy_bak),
        "noisy_OVRL": summarize("noisy_OVRL", noisy_ovrl),
        "enh_SIG":    summarize("enh_SIG", enh_sig),
        "enh_BAK":    summarize("enh_BAK", enh_bak),
        "enh_OVRL":   summarize("enh_OVRL", enh_ovrl),
    }

    # deltas (computed from arrays if lengths match; otherwise from per-row delta)
    if noisy_sig.size == enh_sig.size and noisy_sig.size > 0:
        out["delta_SIG_mean"]  = float(np.mean(enh_sig - noisy_sig))
        out["delta_BAK_mean"]  = float(np.mean(enh_bak - noisy_bak))
        out["delta_OVRL_mean"] = float(np.mean(enh_ovrl - noisy_ovrl))
    else:
        # fallback: use stored delta if present
        def get_delta(metric):
            vals = []
            for r in data:
                v = r.get("delta", {}).get(metric, None)
                if v is None:
                    continue
                vals.append(float(v))
            return np.array(vals, dtype=np.float64)
        dSIG  = get_delta("SIG")
        dBAK  = get_delta("BAK")
        dOVRL = get_delta("OVRL")
        out["delta_SIG_mean"]  = float(np.mean(dSIG))  if dSIG.size  else float("nan")
        out["delta_BAK_mean"]  = float(np.mean(dBAK))  if dBAK.size  else float("nan")
        out["delta_OVRL_mean"] = float(np.mean(dOVRL)) if dOVRL.size else float("nan")

    # Pretty print
    print("Counts:", len(data))
    print("\nNoisy means:")
    print("  SIG :", out["noisy_SIG"]["mean"])
    print("  BAK :", out["noisy_BAK"]["mean"])
    print("  OVRL:", out["noisy_OVRL"]["mean"])

    print("\nEnhanced means:")
    print("  SIG :", out["enh_SIG"]["mean"])
    print("  BAK :", out["enh_BAK"]["mean"])
    print("  OVRL:", out["enh_OVRL"]["mean"])

    print("\nMean deltas (enh - noisy):")
    print("  ΔSIG :", out["delta_SIG_mean"])
    print("  ΔBAK :", out["delta_BAK_mean"])
    print("  ΔOVRL:", out["delta_OVRL_mean"])

if __name__ == "__main__":
    main()