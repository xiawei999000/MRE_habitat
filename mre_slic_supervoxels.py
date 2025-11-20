
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized MRE SLIC supervoxel generation (3D/2D adaptive) with tiny-ROI fallback.
Now supports:
- Keyword-based file matching for cMap/phiMap (e.g., '000781162_cMap.nii' matched by keyword 'cMap')
- Feature export includes Shannon entropy for cMap and phiMap

Input structure:
  root_dir/
    PatientA/
      000781162_cMap.nii
      000781162_phiMap.nii
      label.nii.gz
    PatientB/
      ...

Outputs (per patient):
  supervoxels_slic.nii.gz   (labels: 0=outside ROI, 1..N=supervoxels)
  [optional] supervoxels_features.csv  (per supervoxel stats: mean/std/p25/p75/entropy, sizes)

Key features:
- 3D SLIC on multichannel (cMap, phiMap) within tumor ROI
- Auto 2D-per-slice SLIC when ROI spans few slices, with total n_segments proportionally distributed
- Tiny-ROI fallback (single supervoxel or 2-cluster intensity split)
- Spacing-aware (handles anisotropic voxels), robust normalization in ROI
- Small-fragment cleanup and nearest-neighbor fill
- Optional feature export per supervoxel (mean/std/p25/p75/entropy, size in vox/mm3)
"""

import os
import sys
import argparse
import numpy as np

# Core dependencies
try:
    import SimpleITK as sitk
except ImportError:
    print("Please install SimpleITK: pip install SimpleITK")
    sys.exit(1)

try:
    from skimage.segmentation import slic, relabel_sequential
    from skimage.morphology import remove_small_objects
except ImportError:
    print("Please install scikit-image: pip install scikit-image")
    sys.exit(1)

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    print("Please install SciPy: pip install scipy")
    sys.exit(1)


def log(msg):
    print(msg, flush=True)


def read_image(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # numpy array in z,y,x order
    return img, arr


def resample_to_reference(moving_img, reference_img, is_label=False):
    """
    Resample moving_img to reference_img space using SimpleITK Resample.
    - Linear for images (cMap, phiMap)
    - NearestNeighbor for labels
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    res = sitk.Resample(
        moving_img,
        reference_img,
        sitk.Transform(),
        interp,
        reference_img.GetOrigin(),
        reference_img.GetSpacing(),
        reference_img.GetDirection(),
        0.0,
        moving_img.GetPixelID()
    )
    return res


def robust_normalize_in_mask(arr, mask, eps=1e-6):
    """
    Robust normalization per channel: (x - median) / IQR within mask.
    Fallback to std if IQR ~ 0. NaN/Inf -> 0.
    """
    arr_norm = arr.astype(np.float32).copy()
    arr_norm[~np.isfinite(arr_norm)] = 0.0

    vals = arr_norm[mask > 0]
    if vals.size == 0:
        return arr_norm

    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    if iqr < eps:
        std = float(np.std(vals)) + eps
        arr_norm = (arr_norm - med) / std
    else:
        arr_norm = (arr_norm - med) / iqr

    # clamp extremes for stability
    arr_norm = np.clip(arr_norm, -10.0, 10.0)
    return arr_norm


def compute_n_segments_3d(mask, spacing_xyz, target_size_mm3, min_segs, max_segs):
    """
    Estimate total number of supervoxels in 3D from desired physical size.
    """
    voxel_vol = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    n_vox = int(np.sum(mask > 0))
    if n_vox == 0:
        return 0
    tumor_vol = n_vox * voxel_vol
    est = int(round(tumor_vol / max(target_size_mm3, 1.0)))
    est = max(min_segs, min(max_segs, est))
    return est


def clean_small_segments(labels, mask, min_size_vox):
    labels = labels.copy()
    inside = (mask > 0)

    labels_clean = remove_small_objects(labels, min_size=int(min_size_vox), connectivity=1)
    labels_clean = labels_clean.astype(labels.dtype)

    # 如果全部被清空，回退为单类
    if not np.any(labels_clean > 0) and np.any(inside):
        labels_clean[inside] = 1
        return labels_clean

    holes = (labels_clean == 0) & inside
    keep = (labels_clean > 0)
    if np.any(holes) and np.any(keep):
        _, inds = distance_transform_edt(~keep, return_indices=True)
        filled = labels_clean[tuple(inds)]
        labels_clean[holes] = filled[holes]

    labels_clean[~inside] = 0
    labels_final, _, _ = relabel_sequential(labels_clean)
    return labels_final


def run_slic_with_fallback(feats, mask=None, spacing=None, n_segments=100, compactness=0.1,
                           sigma=0.0, max_num_iter=10, slic_zero=True, start_label=1, channel_axis=-1):
    """
    Robustly call skimage.segmentation.slic with version differences handled.
    - feats: (..., C), C>=1
    """
    labels = None
    base_kwargs = dict(
        n_segments=int(max(1, n_segments)),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=int(start_label),
        channel_axis=channel_axis,
        enforce_connectivity=True,
        spacing=spacing,
    )

    # Attempt 1: full args incl. mask, max_num_iter, slic_zero, convert2lab=False
    try:
        labels = slic(
            feats, mask=mask, max_num_iter=int(max_num_iter), slic_zero=bool(slic_zero),
            convert2lab=False, **base_kwargs
        )
        return labels
    except TypeError:
        pass

    # Attempt 2: without 'mask'
    try:
        labels = slic(
            feats, max_num_iter=int(max_num_iter), slic_zero=bool(slic_zero),
            convert2lab=False, **base_kwargs
        )
    except TypeError:
        labels = None

    if labels is not None:
        if mask is not None:
            labels = labels.astype(np.int32)
            labels[~mask] = 0
        return labels

    # Attempt 3: drop 'max_num_iter' (older versions use max_iter)
    try:
        labels = slic(
            feats, slic_zero=bool(slic_zero), convert2lab=False, **base_kwargs
        )
    except TypeError:
        # Attempt 4: drop 'slic_zero' too
        try:
            labels = slic(
                feats, convert2lab=False, **base_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"SLIC failed across fallbacks: {e}")

    if labels is not None and mask is not None:
        labels = labels.astype(np.int32)
        labels[~mask] = 0
    return labels


def run_slic_supervoxels_3d(cmap_arr, phimap_arr, mask_arr, spacing_xyz, n_segments,
                            compactness=0.1, sigma=0.0, max_num_iter=10, min_size_vox=20,
                            slic_zero=True, z_boost=1.0):
    """
    3D SLIC within mask. 支持单/双通道：
    - cmap_arr 可为 None
    - phimap_arr 可为 None
    至少保证有一个通道非空
    """
    if n_segments <= 0:
        return np.zeros_like(mask_arr, dtype=np.int32)

    m = (mask_arr > 0)
    chans = []
    if cmap_arr is not None:
        chans.append(robust_normalize_in_mask(cmap_arr, m))
    if phimap_arr is not None:
        chans.append(robust_normalize_in_mask(phimap_arr, m))
    if len(chans) == 0:
        raise ValueError("No feature channels provided to 3D SLIC (both cmap_arr and phimap_arr are None).")

    feats = np.stack(chans, axis=-1).astype(np.float32)  # (z,y,x,C>=1)

    spacing_zyx = (spacing_xyz[2] * float(z_boost), spacing_xyz[1], spacing_xyz[0])

    labels = run_slic_with_fallback(
        feats, mask=m, spacing=spacing_zyx, n_segments=n_segments, compactness=compactness,
        sigma=sigma, max_num_iter=max_num_iter, slic_zero=slic_zero, start_label=1, channel_axis=-1
    )

    labels = labels.astype(np.int32)
    labels[~m] = 0
    labels = clean_small_segments(labels, m, min_size_vox=int(min_size_vox))
    return labels


def run_slic_superpixels_2d_per_slice(cmap_arr, phimap_arr, mask_arr, spacing_xyz,
                                      n_segments_total, compactness=0.15, sigma=0.0,
                                      max_num_iter=10, min_size_vox=3, slic_zero=True):
    """
    2D SLIC per-slice when ROI spans very few z-layers. 支持单/双通道。
    """
    zdim, ydim, xdim = mask_arr.shape
    s_x, s_y, s_z = spacing_xyz
    m3d = (mask_arr > 0)

    if n_segments_total <= 0:
        return np.zeros_like(mask_arr, dtype=np.int32)

    # Compute per-slice voxel counts and volumes
    vox_per_slice = [int(np.sum(m3d[z])) for z in range(zdim)]
    vol_per_slice = [v * s_x * s_y * s_z for v in vox_per_slice]
    total_vol = float(sum(vol_per_slice))
    if total_vol <= 0:
        return np.zeros_like(mask_arr, dtype=np.int32)

    # Prepare normalized features
    chans_norm = []
    if cmap_arr is not None:
        chans_norm.append(robust_normalize_in_mask(cmap_arr, m3d))
    if phimap_arr is not None:
        chans_norm.append(robust_normalize_in_mask(phimap_arr, m3d))
    if len(chans_norm) == 0:
        raise ValueError("No feature channels provided to 2D SLIC (both cmap_arr and phimap_arr are None).")

    labels_3d = np.zeros_like(mask_arr, dtype=np.int32)
    next_label = 1

    for z in range(zdim):
        m2d = m3d[z]
        if not np.any(m2d):
            continue

        # Allocate number of segments for this slice (at least 1 if any mask)
        frac = vol_per_slice[z] / total_vol
        n_seg_slice = max(1, int(round(n_segments_total * frac)))

        # Stack features on this slice
        feats2d_chans = [ch[z] for ch in chans_norm]
        feats2d = np.stack(feats2d_chans, axis=-1).astype(np.float32)  # (y,x,C>=1)
        spacing_yx = (s_y, s_x)

        lab2d = run_slic_with_fallback(
            feats2d, mask=m2d, spacing=spacing_yx, n_segments=n_seg_slice, compactness=compactness,
            sigma=sigma, max_num_iter=max_num_iter, slic_zero=slic_zero, start_label=1, channel_axis=-1
        )
        lab2d = lab2d.astype(np.int32)
        lab2d[~m2d] = 0

        # Clean small segments in 2D slice and relabel
        lab2d = clean_small_segments(lab2d, m2d, min_size_vox=int(min_size_vox))

        # Offset labels to ensure uniqueness across slices
        lab2d_nonzero = lab2d > 0
        unique_labels = np.unique(lab2d[lab2d_nonzero])
        if unique_labels.size > 0:
            remap = {old: (i + next_label) for i, old in enumerate(unique_labels)}
            mapped = np.zeros_like(lab2d, dtype=np.int32)
            for old, newv in remap.items():
                mapped[lab2d == old] = newv
            next_label += len(unique_labels)
            labels_3d[z] = mapped
        else:
            labels_3d[z] = lab2d  # all zeros

    labels_final, _, _ = relabel_sequential(labels_3d)
    return labels_final


def handle_tiny_roi(cmap_arr, phimap_arr, mask_arr, policy="single"):
    """
    Fallback for extremely small ROIs.
    - single: one supervoxel (label=1) inside ROI
    - kmeans2: 2-cluster on features; 1D 或 2D 均可
    """
    m = (mask_arr > 0)
    labels = np.zeros_like(mask_arr, dtype=np.int32)
    n_vox = int(np.sum(m))
    if n_vox == 0:
        return labels

    if policy == "single" or n_vox < 12:
        labels[m] = 1
        return labels

    # Build feature matrix in ROI (1D or 2D)
    chans = []
    if cmap_arr is not None:
        chans.append(robust_normalize_in_mask(cmap_arr, m)[m])
    if phimap_arr is not None:
        chans.append(robust_normalize_in_mask(phimap_arr, m)[m])
    if len(chans) == 0:
        labels[m] = 1
        return labels

    X = np.stack(chans, axis=1).astype(np.float32)  # shape (N, C>=1)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        lab_roi = km.labels_ + 1
    except Exception:
        # Fallback: 1D median split or PCA->median split for multi-dim
        if X.shape[1] == 1:
            v = X[:, 0]
        else:
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            v = Xc @ Vt[0]
        thr = np.median(v)
        lab_roi = (v > thr).astype(np.int32) + 1

    labels[m] = lab_roi
    labels = relabel_sequential(labels)[0]
    return labels


# --- Utilities: keyword-based file matching and Shannon entropy ---

def find_file_by_keyword(folder, keyword, exts=(".nii", ".nii.gz")):
    """
    Find first file whose name contains the keyword (case-insensitive) and ends with one of exts.
    Returns full path or None. Sorts candidates to ensure determinism.
    """
    kw = str(keyword).lower()
    candidates = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        lower = fname.lower()
        if kw in lower and lower.endswith(exts):
            candidates.append(fname)
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(folder, candidates[0])


def shannon_entropy(values, bins=32):
    """
    Shannon entropy (base-2) of continuous values by histogram binning.
    """
    if values.size == 0:
        return 0.0
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / float(total)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_supervoxel_features(labels, cmap_arr, phimap_arr, spacing_xyz, out_csv_path,
                                features_used_only=False):
    """
    Export per-supervoxel statistics to CSV.

    当 features_used_only=True 时：
      - 仅输出传入的非 None 通道的列（即被用于分割的通道）
      - 列顺序固定为: 基本列 -> c 通道列(若包含) -> p 通道列(若包含)

    当 features_used_only=False 时（默认，兼容旧逻辑）：
      - 始终输出 c 与 p 两组列；传入为 None 的通道填 NaN
    """
    import csv
    s_x, s_y, s_z = spacing_xyz
    voxel_vol = float(s_x * s_y * s_z)
    lbls = labels.astype(np.int32)
    ulabels = np.unique(lbls)
    ulabels = ulabels[ulabels > 0]

    # 确定哪些通道需要导出
    if features_used_only:
        include_c = (cmap_arr is not None)
        include_p = (phimap_arr is not None)
    else:
        include_c = True
        include_p = True

    # 构造列头（基本列 + 动态通道列）
    header = ["label_id", "size_vox", "size_mm3"]
    if include_c:
        header += ["c_mean", "c_std", "c_p25", "c_p75", "c_entropy"]
    if include_p:
        header += ["p_mean", "p_std", "p_p25", "p_p75", "p_entropy"]

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        if ulabels.size == 0:
            return

        for lab in ulabels:
            idx = (lbls == lab)
            n = int(np.sum(idx))
            if n <= 0:
                continue

            row = [int(lab), n, n * voxel_vol]

            # c 通道（根据 include_c 决定是否写入列）
            if include_c:
                if cmap_arr is not None:
                    cvals = cmap_arr[idx].astype(np.float32)
                    c_mean = float(np.mean(cvals))
                    c_std  = float(np.std(cvals))
                    c_p25  = float(np.percentile(cvals, 25))
                    c_p75  = float(np.percentile(cvals, 75))
                    c_ent  = float(shannon_entropy(cvals, bins=32))
                else:
                    # features_used_only=False 情况下仍写 NaN
                    c_mean = c_std = c_p25 = c_p75 = c_ent = float("nan")
                row += [c_mean, c_std, c_p25, c_p75, c_ent]

            # p 通道（根据 include_p 决定是否写入列）
            if include_p:
                if phimap_arr is not None:
                    pvals = phimap_arr[idx].astype(np.float32)
                    p_mean = float(np.mean(pvals))
                    p_std  = float(np.std(pvals))
                    p_p25  = float(np.percentile(pvals, 25))
                    p_p75  = float(np.percentile(pvals, 75))
                    p_ent  = float(shannon_entropy(pvals, bins=32))
                else:
                    p_mean = p_std = p_p25 = p_p75 = p_ent = float("nan")
                row += [p_mean, p_std, p_p25, p_p75, p_ent]

            writer.writerow(row)


def process_patient(patient_dir, args):
    """
    Process one patient folder.
    """
    # Locate files: cMap/phiMap by keyword; label by exact name
    label_path = os.path.join(patient_dir, args.label_name)
    cmap_path = find_file_by_keyword(patient_dir, args.cmap_name, exts=(".nii", ".nii.gz"))
    phimap_path = find_file_by_keyword(patient_dir, args.phimap_name, exts=(".nii", ".nii.gz"))

    # 必须存在 label；通道存在性取决于 --channels
    if not os.path.isfile(label_path):
        log(f"[Skip] Missing label in: {patient_dir} | label: {args.label_name}")
        return False

    if args.channels == "both":
        if (cmap_path is None) or (phimap_path is None):
            log(f"[Skip] Need both channels in 'both' mode: "
                f"cMap like '*{args.cmap_name}*.nii*' -> {cmap_path} | "
                f"phiMap like '*{args.phimap_name}*.nii*' -> {phimap_path}")
            return False
    elif args.channels == "c":
        if cmap_path is None:
            log(f"[Skip] Need cMap in 'c' mode: '*{args.cmap_name}*.nii*' not found")
            return False
        # 允许 phi 缺失
    elif args.channels == "phi":
        if phimap_path is None:
            log(f"[Skip] Need phiMap in 'phi' mode: '*{args.phimap_name}*.nii*' not found")
            return False
    else:
        log(f"[Error] Invalid channels option: {args.channels}")
        return False

    # Reference: label image space
    label_img, label_arr = read_image(label_path)
    spacing_xyz = label_img.GetSpacing()  # (x,y,z)
    origin = label_img.GetOrigin()
    direction = label_img.GetDirection()

    # Read cMap, phiMap and resample to label space if needed（按需读取）
    cmap_img = None
    phimap_img = None
    if cmap_path is not None:
        cmap_img = sitk.ReadImage(cmap_path)
    if phimap_path is not None:
        phimap_img = sitk.ReadImage(phimap_path)

    def same_space(a, b):
        return (a.GetSize() == b.GetSize()
                and np.allclose(a.GetSpacing(), b.GetSpacing())
                and np.allclose(a.GetOrigin(), b.GetOrigin())
                and np.allclose(a.GetDirection(), b.GetDirection()))

    if cmap_img is not None and not same_space(cmap_img, label_img):
        cmap_img = resample_to_reference(cmap_img, label_img, is_label=False)
    if phimap_img is not None and not same_space(phimap_img, label_img):
        phimap_img = resample_to_reference(phimap_img, label_img, is_label=False)

    cmap_arr = sitk.GetArrayFromImage(cmap_img).astype(np.float32) if cmap_img is not None else None
    phimap_arr = sitk.GetArrayFromImage(phimap_img).astype(np.float32) if phimap_img is not None else None
    mask_arr = (label_arr > 0).astype(np.uint8)

    n_vox = int(np.sum(mask_arr))
    if n_vox == 0:
        log(f"[Warn] Empty mask: {os.path.basename(patient_dir)}")
        return False

    # Count number of z-slices with ROI
    z_has = np.array([np.any(mask_arr[z] > 0) for z in range(mask_arr.shape[0])])
    nz_slices = int(np.sum(z_has))

    # Decide n_segments (total) using 3D volume
    n_segments_total = compute_n_segments_3d(
        mask_arr, spacing_xyz=spacing_xyz,
        target_size_mm3=args.target_size_mm3,
        min_segs=args.min_segments, max_segs=args.max_segments
    )

    # Tiny ROI fallback (early return)
    if n_vox < args.tiny_roi_vox:
        log(f"[TinyROI] {os.path.basename(patient_dir)} | vox={n_vox} | policy={args.tiny_policy} | channels={args.channels}")
        labels = handle_tiny_roi(
            cmap_arr if args.channels in ("both","c") else None,
            phimap_arr if args.channels in ("both","phi") else None,
            mask_arr, policy=args.tiny_policy
        )
    else:
        # Choose 2D or 3D path
        if args.enable_2d_auto and nz_slices <= args.max_slices_for_2d:
            log(f"[2D] {os.path.basename(patient_dir)} | slices_with_ROI={nz_slices} | n_segments_total={n_segments_total} | channels={args.channels}")
            labels = run_slic_superpixels_2d_per_slice(
                cmap_arr if args.channels in ("both","c") else None,
                phimap_arr if args.channels in ("both","phi") else None,
                mask_arr, spacing_xyz,
                n_segments_total=n_segments_total,
                compactness=args.compactness_2d,
                sigma=args.slic_sigma,
                max_num_iter=args.max_num_iter,
                min_size_vox=args.min_size_vox_2d,
                slic_zero=not args.no_slic_zero
            )
        else:
            log(f"[3D] {os.path.basename(patient_dir)} | vox_in_ROI={n_vox} | n_segments_total={n_segments_total} | spacing(x,y,z)={spacing_xyz} | z_boost={args.z_boost} | channels={args.channels}")
            labels = run_slic_supervoxels_3d(
                cmap_arr if args.channels in ("both","c") else None,
                phimap_arr if args.channels in ("both","phi") else None,
                mask_arr, spacing_xyz,
                n_segments=n_segments_total,
                compactness=args.compactness,
                sigma=args.slic_sigma,
                max_num_iter=args.max_num_iter,
                min_size_vox=args.min_size_vox,
                slic_zero=not args.no_slic_zero,
                z_boost=args.z_boost
            )

    # Save labels to NIfTI aligned with label image
    out_img = sitk.GetImageFromArray(labels.astype(np.uint16))
    out_img.SetSpacing(spacing_xyz)
    out_img.SetOrigin(origin)
    out_img.SetDirection(direction)

    out_path = os.path.join(patient_dir, args.out_name)
    sitk.WriteImage(out_img, out_path)
    log(f"[OK] Saved: {out_path}")

    # Optional feature export
    if args.export_features:
        feat_path = os.path.join(patient_dir, args.features_out_name)
        try:
            if args.features_used_only:
                # 仅传入用于分割的通道（未用于分割的通道传 None，从而不生成其列）
                c_for_feat = cmap_arr if args.channels in ("both", "c") else None
                p_for_feat = phimap_arr if args.channels in ("both", "phi") else None
            else:
                # 兼容旧逻辑：都传入，未找到的通道由函数写 NaN 列
                c_for_feat = cmap_arr
                p_for_feat = phimap_arr

            extract_supervoxel_features(
                labels,
                c_for_feat,
                p_for_feat,
                spacing_xyz,
                feat_path,
                features_used_only=args.features_used_only
            )
            log(f"[OK] Features: {feat_path}")
        except Exception as e:
            log(f"[Warn] Feature export failed: {e}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SLIC supervoxel masks for MRE patients (3D/2D adaptive), supporting single- or dual-channel.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root folder containing patient subfolders (e.g., F:\\HCC-RJ\\MRE\\0826-max mask)")

    # Channel selection
    parser.add_argument("--channels", type=str, choices=["both", "c", "phi"], default="both",
                        help="Which image channels to use for SLIC: both=use cMap+phiMap; c=cMap only; phi=phiMap only")

    # Filenames (cMap, phiMap via keyword matching; label remains exact name)
    parser.add_argument("--cmap_name", type=str, default="cMap",
                        help="Keyword to locate cMap file in patient folder (e.g., matches '*cMap*.nii*')")
    parser.add_argument("--phimap_name", type=str, default="phiMap",
                        help="Keyword to locate phiMap file in patient folder (e.g., matches '*phiMap*.nii*')")
    parser.add_argument("--label_name", type=str, default="label.nii.gz",
                        help="Exact filename of tumor mask (NIfTI)")

    # Outputs
    parser.add_argument("--out_name", type=str, default="supervoxels_slic.nii.gz",
                        help="Output filename of supervoxel labels")
    parser.add_argument("--export_features", action="store_true",
                        help="If set, export per-supervoxel features CSV")
    parser.add_argument("--features_out_name", type=str, default="supervoxels_features.csv",
                        help="Per-supervoxel features CSV filename")

    # Sizing and SLIC params (3D default)
    parser.add_argument("--target_size_mm3", type=float, default=125.0,
                        help="Target physical volume per supervoxel (mm^3). Smaller -> more segments.")
    parser.add_argument("--min_segments", type=int, default=80, help="Lower bound for total n_segments")
    parser.add_argument("--max_segments", type=int, default=1500, help="Upper bound for total n_segments")
    parser.add_argument("--compactness", type=float, default=0.1,
                        help="3D SLIC compactness: higher -> more spatially regular")
    parser.add_argument("--slic_sigma", type=float, default=0.0,
                        help="Gaussian smoothing prior to SLIC (in voxel units).")
    parser.add_argument("--max_num_iter", type=int, default=10, help="SLIC max iterations")
    parser.add_argument("--min_size_vox", type=int, default=20,
                        help="Remove 3D segments with fewer voxels than this")
    parser.add_argument("--no_slic_zero", action="store_true",
                        help="Disable slic_zero. By default, enabled to reduce intensity-scale dependence.")
    parser.add_argument("--z_boost", type=float, default=1.0,
                        help="Multiply z spacing by this factor to enhance 2D bias in 3D SLIC.")

    # 2D auto mode for thin/single-layer ROIs
    parser.add_argument("--enable_2d_auto", action="store_true",
                        help="Enable 2D-per-slice SLIC when ROI spans few slices")
    parser.add_argument("--max_slices_for_2d", type=int, default=2,
                        help="If ROI appears in <= this many slices, use 2D-per-slice SLIC")
    parser.add_argument("--compactness_2d", type=float, default=0.15,
                        help="2D SLIC compactness")
    parser.add_argument("--min_size_vox_2d", type=int, default=3,
                        help="Remove 2D segments with fewer pixels than this")

    # Tiny ROI fallback
    parser.add_argument("--tiny_roi_vox", type=int, default=50,
                        help="If ROI voxel count < this, use tiny-ROI fallback")
    parser.add_argument("--tiny_policy", type=str, default="single",
                        choices=["single", "kmeans2"],
                        help="Fallback for tiny ROI: single=one supervoxel; kmeans2=two intensity clusters")
    parser.add_argument("--features_used_only", action="store_true",
                        help="Only export feature columns for the channels actually used for SLIC (c or phi).")

    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root_dir

    if not os.path.isdir(root):
        log(f"[Error] root_dir not found: {root}")
        sys.exit(1)

    # Iterate immediate subdirectories as patients
    patients = [os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
    if not patients:
        log("[Error] No patient subfolders found.")
        sys.exit(1)

    ok, fail = 0, 0
    failed_patients = []
    for p in sorted(patients):
        try:
            res = process_patient(p, args)
            if res:
                ok += 1
            else:
                fail += 1
                failed_patients.append(p)
        except Exception as e:
            log(f"[Fail] {p} -> {e}")
            fail += 1

    log(f"Done. Success: {ok}, Fail: {fail}, Failed patients: {failed_patients}")


if __name__ == "__main__":
    main()
