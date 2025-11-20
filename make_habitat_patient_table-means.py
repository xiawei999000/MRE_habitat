
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a cohort-level table with per-patient habitat features:
- For each habitat k in {1..K}:
    - Weighted mean of c_mean and p_mean
    - Weighted mean of c_entropy and p_entropy
    - Volume proportion (by voxel count or mm3)
Outputs a single CSV with one row per patient.

Requirements:
    pip install pandas numpy

Expected per-patient files (default names; change via CLI):
    - supervoxels_features.csv      (has label_id, size_vox/size_mm3, c_mean, p_mean, c_entropy, p_entropy ...)
    - supervoxels_habitat.csv       (has label_id, habitat)

Typical usage:
    python make_habitat_patient_table.py \
        --root_dir "F:\HCC-RJ\MRE\0826-max mask" \
        --k 3 \
        --out_excel "patient_habitat_features_k3.xlsx"
"""

import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def log(msg: str):
    print(msg, flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_patients(root_dir: str) -> List[str]:
    return sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])


def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def pick_weight_column(df_feat: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """
    Choose weight column in order:  size_mm3 -> ones -> size_vox
    Returns (col_name, weights_array)
    """
    if "size_mm3" in df_feat.columns:
        w = df_feat["size_mm3"].fillna(0).to_numpy()
        return "size_mm3", w
    elif "size_vox" in df_feat.columns:
        w = df_feat["size_vox"].fillna(0).to_numpy()
        return "size_vox", w
    else:
        w = np.ones(len(df_feat), dtype=float)
        return "ones", w


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    m = float(np.sum(values * weights))
    w = float(np.sum(weights))
    return m / w if w > 0 else 0.0

def build_patient_row(
    patient_id: str,
    df_feat: pd.DataFrame,
    df_hab: pd.DataFrame,
    K: int = 3,
) -> dict:
    """
    Merge features with habitat labels and compute per-habitat aggregates.
    """
    # Merge on label_id
    if "label_id" not in df_feat.columns or "label_id" not in df_hab.columns:
        raise ValueError("Both CSVs must contain 'label_id' column.")
    if "habitat" not in df_hab.columns:
        raise ValueError("'supervoxels_habitat.csv' must contain 'habitat' column.")

    df = pd.merge(df_feat, df_hab[["label_id", "habitat"]], on="label_id", how="inner")
    # Keep only labeled habitats > 0 and <= K
    df = df[(df["habitat"] >= 1) & (df["habitat"] <= K)].copy()

    # Determine weights
    w_name, weights_all = pick_weight_column(df)
    df["_w"] = weights_all

    # Columns that may exist
    has_c_mean = "c_mean" in df.columns
    has_p_mean = "p_mean" in df.columns

    has_c_std = "c_std" in df.columns
    has_p_std = "p_std" in df.columns

    has_c_p25 = "c_p25" in df.columns
    has_p_p25 = "p_p25" in df.columns

    has_c_p75 = "c_p75" in df.columns
    has_p_p75 = "p_p75" in df.columns

    has_c_ent  = "c_entropy" in df.columns
    has_p_ent  = "p_entropy" in df.columns

    # Totals for proportions
    total_w = float(df["_w"].sum())
    row = {"patient_id": patient_id, "total_weight": total_w, "weight_col": w_name}

    # Iterate habitats
    for k in range(1, K + 1):
        mask = (df["habitat"] == k)
        w = df.loc[mask, "_w"].to_numpy(dtype=float)
        # Proportion by chosen weight
        prop = float(w.sum()) / total_w if total_w > 0 else 0.0
        row[f"prop_h{k}"] = prop
        row[f"weight_h{k}"] = float(w.sum())

        # Weighted means (if columns present)
        if has_c_mean:
            row[f"c_mean_h{k}"] = df.loc[mask, "c_mean"].astype(float).mean() if mask.any() else 0.0
        if has_p_mean:
            row[f"p_mean_h{k}"] = df.loc[mask, "p_mean"].astype(float).mean() if mask.any() else 0.0

        if has_c_std:
            row[f"c_std_h{k}"] = df.loc[mask, "c_std"].astype(float).mean() if mask.any() else 0.0
        if has_p_std:
            row[f"p_std_h{k}"] = df.loc[mask, "p_std"].astype(float).mean() if mask.any() else 0.0

        if has_c_p25:
            row[f"c_p25_h{k}"] = df.loc[mask, "c_p25"].astype(float).mean() if mask.any() else 0.0
        if has_p_p25:
            row[f"p_p25_h{k}"] = df.loc[mask, "p_p25"].astype(float).mean() if mask.any() else 0.0

        if has_c_p75:
            row[f"c_p75_h{k}"] = df.loc[mask, "c_p75"].astype(float).mean() if mask.any() else 0.0
        if has_p_p75:
            row[f"p_p75_h{k}"] = df.loc[mask, "p_p75"].astype(float).mean() if mask.any() else 0.0

        if has_c_ent:
            row[f"c_entropy_h{k}"] = df.loc[mask, "c_entropy"].astype(float).mean() if mask.any() else 0.0
        if has_p_ent:
            row[f"p_entropy_h{k}"] = df.loc[mask, "p_entropy"].astype(float).mean() if mask.any() else 0.0

    # Optional: overall composition entropy across habitats (diversity)
    props = np.array([row.get(f"prop_h{k}", 0.0) for k in range(1, K + 1)], dtype=float)
    props = props[(props > 0)]
    comp_entropy = float(-np.sum(props * np.log(props))) if props.size > 0 else 0.0
    row["habitat_entropy"] = comp_entropy

    return row


def main():
    ap = argparse.ArgumentParser(description="Summarize per-patient habitat features into one cohort CSV")
    ap.add_argument("--root_dir", type=str, required=True, help="Root directory containing patient subfolders")
    ap.add_argument("--features_csv_name", type=str, default="supervoxels_features.csv",
                    help="Per-patient features CSV filename")
    ap.add_argument("--habitat_csv_name", type=str, default="supervoxels_habitat.csv",
                    help="Per-patient habitat assignment CSV filename")
    ap.add_argument("--k", type=int, default=3, help="Number of habitats")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--out_excel", type=str, default="patient_habitat_features.xlsx",
                    help="Output cohort excel")
    args = ap.parse_args()

    root = args.root_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    patients = list_patients(root)
    if not patients:
        log(f"[Error] No patient subfolders found under: {root}")
        sys.exit(1)

    rows = []
    for pdir in patients:
        pid = os.path.basename(pdir)
        f_feat = os.path.join(pdir, args.features_csv_name)
        f_hab  = os.path.join(pdir, args.habitat_csv_name)

        df_feat = load_csv(f_feat)
        df_hab  = load_csv(f_hab)
        if df_feat is None or df_hab is None:
            log(f"[Skip] Missing CSV for {pid}: "
                f"{'features missing' if df_feat is None else ''} "
                f"{'habitat missing' if df_hab is None else ''}")
            continue
        try:
            row = build_patient_row(pid, df_feat, df_hab, K=args.k)
            rows.append(row)
            log(f"[OK] {pid} summarized.")
        except Exception as e:
            log(f"[Fail] {pid}: {e}")

    if not rows:
        log("[Error] No patient summarized. Check inputs.")
        sys.exit(2)

    df_out = pd.DataFrame(rows)
    # Order columns: id, totals, proportions, means, entropies
    cols = ["patient_id", "weight_col", "total_weight"]
    K = args.k
    cols += [f"prop_h{k}" for k in range(1, K + 1)]
    # Add means/entropies only if present in df
    for prefix in ["c_mean", "p_mean", "c_std", "p_std", "c_p25", "p_p25", "c_p75", "p_p75", "c_entropy", "p_entropy"]:
        cands = [f"{prefix}_h{k}" for k in range(1, K + 1)]
        existing = [c for c in cands if c in df_out.columns]
        cols += existing
    cols += ["habitat_entropy"]
    # Include raw weights per habitat at the end (optional)
    cols += [f"weight_h{k}" for k in range(1, K + 1) if f"weight_h{k}" in df_out.columns]

    # Reindex safely
    cols = [c for c in cols if c in df_out.columns]
    df_out = df_out.reindex(columns=cols)

    out_path = os.path.join(out_dir, args.out_excel) if not os.path.isabs(args.out_excel) else args.out_excel
    df_out.to_excel(out_path, index=False)
    log(f"[DONE] Cohort table saved to: {out_path}")


if __name__ == "__main__":
    main()
