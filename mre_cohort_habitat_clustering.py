
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cohort-level clustering + habitat backfilling (complete)

Per-patient expected inputs (filenames configurable by CLI):
  - supervoxels_features.csv   (one row per supervoxel with features)
  - supervoxels_slic.nii.gz    (label volume; voxel values are supervoxel IDs)

Outputs:
  - model_dir (default: <root_dir>/_habitat_model):
      scaler.pkl, kmeans.pkl, centers.npy, feature_list.json, metadata.json
      patient_habitat_summary.csv
  - per patient:
      habitat_map.nii.gz
      supervoxels_habitat.csv

Typical usage:
  python mre_cohort_habitat_clustering.py \
    --root_dir "F:\HCC-RJ\MRE\0826-max mask" \
    --mode both --k_range 3:6 --scaler robust --min_size_vox_train 3
"""

import os
import sys
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:
    print("Please install SimpleITK: pip install SimpleITK", flush=True)
    sys.exit(1)

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
import joblib


# ---------------------------
# Utilities
# ---------------------------

def log(msg: str):
    print(msg, flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_patients(root_dir: str) -> List[str]:
    return sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])


def parse_k_range(s: str) -> Tuple[int, int]:
    """Accept '3:8' or '3-8'."""
    if ":" in s:
        a, b = s.split(":")
    elif "-" in s:
        a, b = s.split("-")
    else:
        raise ValueError(f"Invalid k range: {s} (use '3:8' or '3-8')")
    a, b = int(a), int(b)
    if a < 2 or b <= a:
        raise ValueError("k range must satisfy 2 <= k_min < k_max")
    return a, b


def parse_feature_cols(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def load_features_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def default_feature_list(include_size: bool = False) -> List[str]:
    feats = [
        "c_mean", "c_std", "c_p25", "c_p75",
        "p_mean", "p_std", "p_p25", "p_p75",
        "c_entropy", "p_entropy",
    ]
    if include_size:
        feats += ["size_vox", "size_mm3"]
    return feats


def sanitize_X(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    cols_present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing feature columns (will be skipped): {missing}")
    if len(cols_present) == 0:
        return np.zeros((len(df), 0), dtype=np.float32), cols_present
    X = df[cols_present].to_numpy(dtype=np.float32)
    # Replace NaN/Inf with column medians; fallback to zeros
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            X[:, j] = 0.0
        else:
            med = float(np.median(col[finite]))
            col = np.where(np.isfinite(col), col, med)
            X[:, j] = col
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols_present


def sample_indices_per_patient(n: int, max_n: int, rng: np.random.RandomState) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=max_n, replace=False)


def distances_to_centers(Xn: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Efficient Euclidean distances: [N, D] vs [K, D] -> [N, K]
    x2 = np.sum(Xn * Xn, axis=1, keepdims=True)        # [N, 1]
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # [1, K]
    d2 = x2 + c2 - 2.0 * (Xn @ centers.T)              # [N, K]
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float32)


# ---------------------------
# Data aggregation (TRAIN)
# ---------------------------

def aggregate_cohort(
    root_dir: str,
    features_csv_name: str,
    feature_cols: List[str],
    min_size_vox_train: int = 3,
    per_patient_sample: int = 0,
    random_state: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Returns:
      X_all: [N, D] feature matrix
      meta:  DataFrame with columns [patient_id, label_id, size_vox, size_mm3]
      cols_used: list of feature columns actually used (order)
    """
    rng = check_random_state(random_state)
    rows = []
    mats = []
    used_cols = None

    patients = list_patients(root_dir)
    if not patients:
        raise RuntimeError("No patient subfolders found under root_dir.")

    log(f"[Scan] Found {len(patients)} patient folders under: {root_dir}")

    for pdir in patients:
        csv_path = os.path.join(pdir, features_csv_name)
        if not os.path.isfile(csv_path):
            log(f"[Skip] Features CSV not found: {csv_path}")
            continue
        df = load_features_csv(csv_path)

        if "label_id" not in df.columns:
            log(f"[Skip] 'label_id' column not found in {csv_path}")
            continue

        # Filter by size_vox if available
        if "size_vox" in df.columns:
            before = len(df)
            df = df[df["size_vox"].fillna(0) >= min_size_vox_train].copy()
            after = len(df)
            if after == 0:
                log(f"[Skip] After filtering size_vox>={min_size_vox_train}, empty: {csv_path}")
                continue
            elif after < before:
                log(f"[Info] {os.path.basename(pdir)} filtered {before-after}/{before} tiny supervoxels")

        X, cols_present = sanitize_X(df, feature_cols)
        if X.shape[1] == 0:
            log(f"[Skip] No usable features in {csv_path}")
            continue
        if used_cols is None:
            used_cols = cols_present
        else:
            # Align to first patient's cols
            X, _ = sanitize_X(df, used_cols)

        # per-patient sampling
        idx = sample_indices_per_patient(len(df), per_patient_sample, rng)
        Xs = X[idx]
        mats.append(Xs)

        # meta
        cols_meta = ["label_id"] + [c for c in ["size_vox", "size_mm3"] if c in df.columns]
        dfm = df.iloc[idx][cols_meta].copy()
        dfm.insert(0, "patient_id", os.path.basename(pdir))
        rows.append(dfm)

    if not mats:
        raise RuntimeError("No data aggregated; check inputs and filters.")

    X_all = np.concatenate(mats, axis=0)
    meta = pd.concat(rows, axis=0, ignore_index=True)
    log(f"[Agg] Aggregated samples: {X_all.shape[0]} | feature_dim: {X_all.shape[1]}")
    return X_all, meta, used_cols


# ---------------------------
# Training
# ---------------------------

def train_model(
    X: np.ndarray,
    scaler_type: str = "robust",
    k: Optional[int] = None,
    k_range: Optional[Tuple[int, int]] = None,
    random_state: int = 0,
    batch_size: int = 2048,
    silhouette_sample: int = 10000,
) -> Tuple[object, object, np.ndarray, Dict]:
    """
    Returns:
      scaler, kmeans, centers, info
    """
    # Scaler
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    Xn = scaler.fit_transform(X)
    info = {"scaler": scaler_type}

    # Choose K
    if k is None and k_range is None:
        raise ValueError("Provide either fixed k or k_range for auto-selection.")
    if k is None:
        k_min, k_max = k_range
        # Subsample for silhouette
        if silhouette_sample > 0 and Xn.shape[0] > silhouette_sample:
            rng = check_random_state(random_state)
            sel = rng.choice(Xn.shape[0], size=silhouette_sample, replace=False)
            X_eval = Xn[sel]
        else:
            X_eval = Xn

        best_k, best_score = None, -1.0
        for kk in range(k_min, k_max + 1):
            try:
                mbk = MiniBatchKMeans(n_clusters=kk, random_state=random_state, batch_size=batch_size)
                labels = mbk.fit_predict(X_eval)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    score = silhouette_score(X_eval, labels, metric="euclidean")
                log(f"[AutoK] k={kk}, silhouette={score:.4f}")
                if score > best_score:
                    best_k, best_score = kk, score
            except Exception as e:
                log(f"[AutoK] k={kk} failed silhouette: {e}")
        if best_k is None:
            raise RuntimeError("AutoK failed for all candidates.")
        k = best_k
        info["silhouette"] = float(best_score)
        log(f"[AutoK] Selected k={k} (silhouette={best_score:.4f})")

    # Final model on full X
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size)
    labels = kmeans.fit_predict(Xn)
    centers = kmeans.cluster_centers_
    print("centers=", centers)
    info["k"] = int(k)
    info["n_train"] = int(X.shape[0])
    return (scaler, kmeans, centers, info)


def save_model(model_dir: str, scaler, kmeans, centers: np.ndarray, feature_cols: List[str], info: Dict):
    ensure_dir(model_dir)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans.pkl"))
    np.save(os.path.join(model_dir, "centers.npy"), centers.astype(np.float32))
    with open(os.path.join(model_dir, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log(f"[Model] Saved to: {model_dir}")


def load_model(model_dir: str):
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
    centers = np.load(os.path.join(model_dir, "centers.npy"))
    with open(os.path.join(model_dir, "feature_list.json"), "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(os.path.join(model_dir, "metadata.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    return scaler, kmeans, centers, feature_cols, info


# ---------------------------
# Backfilling
# ---------------------------

def backfill_patient(
    patient_dir: str,
    scaler,
    centers: np.ndarray,
    feature_cols: List[str],
    features_csv_name: str,
    labels_nii_name: str,
    out_habitat_name: str,
    out_assign_name: str,
) -> Optional[Dict]:
    """
    Create habitat map and assignment CSV for one patient.
    Returns summary dict or None if missing files.
    """
    csv_path = os.path.join(patient_dir, features_csv_name)
    lab_path = os.path.join(patient_dir, labels_nii_name)
    if not (os.path.isfile(csv_path) and os.path.isfile(lab_path)):
        log(f"[Skip] Missing inputs for {os.path.basename(patient_dir)}")
        return None

    df = load_features_csv(csv_path)
    if "label_id" not in df.columns:
        log(f"[Skip] label_id not found in {csv_path}")
        return None

    # Prepare features (align to feature_cols)
    X_present, cols_used = sanitize_X(df, feature_cols)
    if cols_used != feature_cols:
        # rebuild with exact order; missing columns -> zeros
        X_full = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
        for j, c in enumerate(feature_cols):
            if c in df.columns:
                X_full[:, j] = df[c].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X = X_present

    Xn = scaler.transform(X)

    # Distances and assignments
    dists = distances_to_centers(Xn, centers)  # [N, K]
    nearest = np.argmin(dists, axis=1)  # 0..K-1
    # Confidence via margin ratio
    sorted_idx = np.argsort(dists, axis=1)
    dmin = dists[np.arange(len(df)), sorted_idx[:, 0]]
    d2nd = dists[np.arange(len(df)), sorted_idx[:, 1]] if dists.shape[1] > 1 else np.copy(dmin)
    confidence = (d2nd - dmin) / (d2nd + 1e-6)
    habitats = nearest + 1  # 1..K

    # Save assignment CSV
    assign = df.copy()
    assign["habitat"] = habitats
    assign["dist_min"] = dmin
    assign["dist_2nd"] = d2nd
    assign["confidence"] = confidence
    assign.to_csv(os.path.join(patient_dir, out_assign_name), index=False)

    # Backfill to volume
    lab_img = sitk.ReadImage(lab_path)
    lab_arr = sitk.GetArrayFromImage(lab_img).astype(np.int32)  # (z,y,x)
    habitat_arr = np.zeros_like(lab_arr, dtype=np.uint16)

    # Map label_id -> habitat
    id2hab = dict(zip(assign["label_id"].astype(int).tolist(), habitats.tolist()))
    mask_nonzero = lab_arr > 0
    unique_ids = np.unique(lab_arr[mask_nonzero])
    missing_ids = []
    for lid in unique_ids:
        h = id2hab.get(int(lid), 0)
        if h > 0:
            habitat_arr[lab_arr == int(lid)] = int(h)
        else:
            missing_ids.append(int(lid))
    if missing_ids:
        log(f"[Warn] {os.path.basename(patient_dir)}: {len(missing_ids)} labels in NIfTI not found in CSV (left as 0)")

    # Save NIfTI
    out_img = sitk.GetImageFromArray(habitat_arr)
    out_img.CopyInformation(lab_img)
    out_path = os.path.join(patient_dir, out_habitat_name)
    sitk.WriteImage(out_img, out_path)
    log(f"[OK] Habitat map saved: {out_path}")

    # Summaries
    K = centers.shape[0]
    vox_counts = np.array([np.sum(habitat_arr == (k + 1)) for k in range(K)], dtype=np.int64)
    total_vox = int(np.sum(vox_counts))
    proportions = (vox_counts / total_vox).tolist() if total_vox > 0 else [0.0] * K

    # mm3 using lab_img spacing
    sx, sy, sz = lab_img.GetSpacing()  # (x,y,z)
    vvx = float(sx * sy * sz)
    vols_mm3 = (vox_counts * vvx).tolist()

    return dict(
        patient_id=os.path.basename(patient_dir),
        total_vox=total_vox,
        spacing_xyz=[sx, sy, sz],
        voxel_volume_mm3=vvx,
        vox_counts=vox_counts.tolist(),
        vols_mm3=vols_mm3,
        proportions=proportions,
    )


def summarize_cohort(summaries: List[Dict], model_dir: str):
    if not summaries:
        return
    K = len(summaries[0]["proportions"])
    rows = []
    for s in summaries:
        row = dict(patient_id=s["patient_id"], total_vox=s["total_vox"])
        for k in range(K):
            row[f"vox_h{k+1}"] = s["vox_counts"][k]
            row[f"vol_mm3_h{k+1}"] = s["vols_mm3"][k]
            row[f"prop_h{k+1}"] = s["proportions"][k]
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(model_dir, "patient_habitat_summary.csv")
    df.to_csv(out_csv, index=False)
    log(f"[OK] Cohort summary -> {out_csv}")


# ---------------------------
# CLI and Main
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Cohort-level clustering + habitat backfilling from supervoxel CSVs"
    )
    ap.add_argument("--root_dir", type=str, required=True, help="Root directory with patient subfolders")
    ap.add_argument("--mode", type=str, default="both", choices=["train", "apply", "both"],
                    help="Workflow mode")
    ap.add_argument("--features_csv_name", type=str, default="supervoxels_features.csv",
                    help="Per-patient features CSV filename")
    ap.add_argument("--labels_nii_name", type=str, default="supervoxels_slic.nii.gz",
                    help="Per-patient supervoxel labels NIfTI filename")
    ap.add_argument("--out_habitat_name", type=str, default="habitat_map.nii.gz",
                    help="Output NIfTI filename for habitat map")
    ap.add_argument("--out_assign_name", type=str, default="supervoxels_habitat.csv",
                    help="Output CSV filename with habitat assignment per supervoxel")
    ap.add_argument("--model_dir", type=str, default=None,
                    help="Directory to save/load model; default: <root_dir>/_habitat_model")

    # Training options
    ap.add_argument("--scaler", type=str, default="robust", choices=["robust", "standard"],
                    help="Feature scaler type for training and application")
    ap.add_argument("--k", type=int, default=None, help="Fixed number of clusters")
    ap.add_argument("--k_range", type=str, default=None,
                    help="Range for auto-K, e.g., '3:6' or '3-6'")
    ap.add_argument("--min_size_vox_train", type=int, default=3,
                    help="Filter out supervoxels smaller than this during training aggregation")
    ap.add_argument("--per_patient_sample", type=int, default=0,
                    help="Max samples per patient for training (0 for all)")
    ap.add_argument("--random_state", type=int, default=0, help="Random seed")
    ap.add_argument("--batch_size", type=int, default=2048, help="MiniBatchKMeans batch size")
    ap.add_argument("--silhouette_sample", type=int, default=10000,
                    help="Max samples for silhouette evaluation; 0 to disable")
    ap.add_argument("--feature_cols", type=str, default=None,
                    help="Comma-separated feature columns. If omitted, use default set.")
    return ap.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    if not os.path.isdir(root_dir):
        log(f"[Error] root_dir not found: {root_dir}")
        sys.exit(1)

    model_dir = args.model_dir or os.path.join(root_dir, "_habitat_model")
    ensure_dir(model_dir)

    # Features to use
    user_cols = parse_feature_cols(args.feature_cols)
    feature_cols = user_cols if user_cols else default_feature_list(include_size=False)
    log(f"[Config] mode={args.mode} | scaler={args.scaler} | feature_cols={feature_cols}")

    scaler = None
    kmeans = None
    centers = None

    if args.mode in ("train", "both"):
        # Aggregate cohort
        X_all, meta, used_cols = aggregate_cohort(
            root_dir=root_dir,
            features_csv_name=args.features_csv_name,
            feature_cols=feature_cols,
            min_size_vox_train=args.min_size_vox_train,
            per_patient_sample=args.per_patient_sample,
            random_state=args.random_state,
        )
        feature_cols = used_cols  # lock the actual used order

        # Train model
        k_range = parse_k_range(args.k_range) if (args.k is None and args.k_range) else None
        scaler, kmeans, centers, info = train_model(
            X=X_all,
            scaler_type=args.scaler,
            k=args.k,
            k_range=k_range,
            random_state=args.random_state,
            batch_size=args.batch_size,
            silhouette_sample=args.silhouette_sample,
        )
        info.update({
            "root_dir": root_dir,
            "features_csv_name": args.features_csv_name,
            "labels_nii_name": args.labels_nii_name,
            "feature_cols": feature_cols,
            "min_size_vox_train": args.min_size_vox_train,
            "per_patient_sample": args.per_patient_sample,
            "random_state": args.random_state,
            "batch_size": args.batch_size,
        })
        save_model(model_dir, scaler, kmeans, centers, feature_cols, info)

    if args.mode in ("apply", "both"):
        if scaler is None or centers is None:
            # Load model if not from training step in this run
            scaler, kmeans, centers, feature_cols, info = load_model(model_dir)
            log(f"[Model] Loaded from: {model_dir} | k={info.get('k')} | scaler={info.get('scaler') }")
            print("centers=", centers)

        # Apply to all patients
        patients = list_patients(root_dir)
        if not patients:
            log("[Error] No patient subfolders to apply.")
            sys.exit(1)

        summaries = []
        for pdir in patients:
            try:
                s = backfill_patient(
                    patient_dir=pdir,
                    scaler=scaler,
                    centers=centers,
                    feature_cols=feature_cols,
                    features_csv_name=args.features_csv_name,
                    labels_nii_name=args.labels_nii_name,
                    out_habitat_name=args.out_habitat_name,
                    out_assign_name=args.out_assign_name,
                )
                if s is not None:
                    summaries.append(s)
            except Exception as e:
                log(f"[Fail] Apply {os.path.basename(pdir)} -> {e}")

        summarize_cohort(summaries, model_dir)

    log("Done.")


if __name__ == "__main__":
    main()
