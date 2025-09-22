#!/usr/bin/env python3
"""
plot_timings_variants.py

Plots latency distributions from a timings CSV (e.g., produced by yolo_infer_variants.py).
Understands columns: image, latency_s, dataset_label, mode, cpu_percent, rss_mb, timestamp.

Outputs (PNG + CSV) go to --out.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_float_series(s: pd.Series) -> pd.Series:
    # coerce empty strings to NaN, cast to float
    return pd.to_numeric(s, errors="coerce")

def pct(series: pd.Series, q: float) -> float:
    a = series.dropna().to_numpy()
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    x = df["latency_s"].dropna()
    if x.empty:
        return pd.DataFrame([{
            "count": 0, "mean": np.nan, "median": np.nan,
            "p90": np.nan, "p95": np.nan, "p99": np.nan,
            "min": np.nan, "max": np.nan, "std": np.nan
        }])
    return pd.DataFrame([{
        "count": x.count(),
        "mean": x.mean(),
        "median": x.median(),
        "p90": pct(x, 90),
        "p95": pct(x, 95),
        "p99": pct(x, 99),
        "min": x.min(),
        "max": x.max(),
        "std": x.std(ddof=1) if x.count() > 1 else 0.0
    }])

def summarize_by(df: pd.DataFrame, cols):
    g = df.groupby(cols, dropna=False, sort=True)
    rows = []
    for keys, sub in g:
        sm = summarize(sub).iloc[0].to_dict()
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**{c: k for c, k in zip(cols, keys)}, **sm})
    return pd.DataFrame(rows)

def plot_hist(df: pd.DataFrame, out: Path, title="Latency Histogram"):
    plt.figure()
    df["latency_s"].dropna().plot(kind="hist", bins=40, edgecolor="black")
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title(title)
    plt.savefig(out / "latency_hist.png", bbox_inches="tight")
    plt.close()

def plot_cdf(df: pd.DataFrame, out: Path, fname="latency_cdf.png", title="Latency CDF"):
    x = np.sort(df["latency_s"].dropna().to_numpy())
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size
    plt.figure()
    plt.plot(x, y, drawstyle="steps-post")
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out / fname, bbox_inches="tight")
    plt.close()

def plot_box_by(df: pd.DataFrame, col: str, out: Path, fname: str):
    if col not in df.columns:
        return
    # keep ordering by median latency
    med = df.groupby(col)["latency_s"].median().sort_values()
    ordered = med.index.tolist()
    data = [df.loc[df[col] == k, "latency_s"].dropna().to_numpy() for k in ordered]
    if not any(len(d) for d in data):
        return
    plt.figure()
    plt.boxplot(data, labels=ordered, showfliers=False)
    plt.xlabel(col)
    plt.ylabel("Latency (s)")
    plt.title(f"Latency by {col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / f"box_by_{col}.png", bbox_inches="tight")
    plt.close()

def plot_group_cdfs(df: pd.DataFrame, col: str, out: Path, fname: str, max_groups=8):
    if col not in df.columns:
        return
    # pick top groups by count
    top = df[col].value_counts().head(max_groups).index.tolist()
    plt.figure()
    for g in top:
        sub = df[df[col] == g]["latency_s"].dropna().to_numpy()
        if sub.size == 0: 
            continue
        x = np.sort(sub)
        y = np.arange(1, x.size + 1) / x.size
        plt.plot(x, y, label=str(g))
    if plt.gca().lines:
        plt.xlabel("Latency (s)")
        plt.ylabel("CDF")
        plt.title(f"Latency CDF by {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / fname, bbox_inches="tight")
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to timings CSV (from yolo_infer_variants.py)")
    ap.add_argument("--out", default="plots", help="Output folder for PNGs and CSV summaries")
    ap.add_argument("--group-cols", default="dataset_label,mode",
                    help="Comma-separated columns to group by (if present). Default: dataset_label,mode")
    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    df = pd.read_csv(args.csv)

    # Choose latency column: prefer 'latency_s', fall back to 't_total_s'
    if "latency_s" not in df.columns and "t_total_s" in df.columns:
        df = df.rename(columns={"t_total_s": "latency_s"})
    if "latency_s" not in df.columns:
        raise SystemExit("CSV must have 'latency_s' (or 't_total_s').")

    # Clean/cast
    df["latency_s"] = to_float_series(df["latency_s"])
    if "cpu_percent" in df.columns: df["cpu_percent"] = to_float_series(df["cpu_percent"])
    if "rss_mb" in df.columns: df["rss_mb"] = to_float_series(df["rss_mb"])

    # Drop rows without latency
    df = df[df["latency_s"].notna()].copy()
    if df.empty:
        raise SystemExit("No valid latency values found.")

    # ---- summaries ----
    overall = summarize(df)
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    existing_group_cols = [c for c in group_cols if c in df.columns]
    # per-column summaries
    for c in existing_group_cols:
        s = summarize_by(df, [c]).sort_values(["median", "mean"])
        s.to_csv(out_dir / f"summary_by_{c}.csv", index=False)
    # multi-column summary if both present
    if len(existing_group_cols) >= 2:
        s2 = summarize_by(df, existing_group_cols).sort_values(existing_group_cols + ["median"])
        s2.to_csv(out_dir / f"summary_by_{'_'.join(existing_group_cols)}.csv", index=False)

    # ---- plots ----
    plot_hist(df, out_dir)
    plot_cdf(df, out_dir)

    # boxplots
    for c in existing_group_cols:
        plot_box_by(df, c, out_dir, f"box_by_{c}.png")

    # per-group CDFs
    for c in existing_group_cols:
        plot_group_cdfs(df, c, out_dir, f"cdf_by_{c}.png")

    print(f"Saved plots and summaries in: {out_dir}")

if __name__ == "__main__":
    main()

