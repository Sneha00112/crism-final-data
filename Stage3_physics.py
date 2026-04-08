#!/usr/bin/env python3
"""
stage3_physics.py  —  CRISM Physics-Based Correction
=====================================================
Handles the noises correctable by physical models:
  N3  Spectral spikes   → median filter replacement
  N4  Thermal emission  → vectorised Planck subtraction
  N5  Saturation        → local interpolation
  +   Illumination variation → row-median normalisation
  +   Atmospheric dust slope → multiplicative anchor correction

ML-targets (N1 stripe, N2 Gaussian) are NOT touched here;
they are passed to stage4_denoising.py intact.

Key fixes vs previous version
------------------------------
  FIX-PH-VECT: Planck loop is now fully vectorised (no Python pixel loop).
  FIX-PH-ANCHOR: Atmospheric anchor adapts to scene brightness.
  FIX-PH-SNR: SNR computed on good bands only.
"""

import os, glob, json, re, warnings
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

from crism_utils import (
    parse_pds3_label, load_crism_cube, find_label,
    get_sensor_waves, build_bad_mask,
    IF_MAX, IF_MIN,
)

DATA_DIR    = os.environ.get("CRISM_DATA_DIR",
              r"D:\NEW CROSS MISSION\Data\Scene 8\Crism")
EDA_DIR     = os.path.join(DATA_DIR, "eda_output")
NOISE_DIR   = os.path.join(DATA_DIR, "noise_output")
OUTPUT_DIR  = os.path.join(DATA_DIR, "physics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SNR helper (good bands only)
# ─────────────────────────────────────────────
def snr_good_bands(cube, good_mask):
    snrs = []
    for bi in np.where(good_mask)[0]:
        v = cube[:, :, bi]
        v = v[np.isfinite(v) & (v < IF_MAX) & (v > IF_MIN)]
        if v.size:
            snrs.append(abs(v.mean()) / (v.std() + 1e-8))
    return float(np.mean(snrs)) if snrs else 0.0


# ─────────────────────────────────────────────
# ILLUMINATION (row-median normalisation)
# ─────────────────────────────────────────────
def illumination_correction(cube, good_mask):
    """
    Divide each row by its median I/F (computed on good, finite pixels)
    to equalise across-track illumination gradients.
    Only applied to good bands.
    """
    H, W, B = cube.shape
    out = cube.copy()
    for bi in np.where(good_mask)[0]:
        band = out[:, :, bi]
        vm   = np.isfinite(band) & (band < IF_MAX) & (band > IF_MIN)
        row_meds = np.array([
            float(np.median(band[r, vm[r]])) if vm[r].any() else np.nan
            for r in range(H)])
        global_med = np.nanmedian(row_meds[np.isfinite(row_meds)])
        if np.isnan(global_med) or global_med < 1e-6:
            continue
        for r in range(H):
            if np.isfinite(row_meds[r]) and row_meds[r] > 1e-6:
                scale = global_med / row_meds[r]
                # Clamp scale to avoid amplifying noise-dominated rows
                scale = np.clip(scale, 0.5, 2.0)
                out[r, vm[r], bi] *= scale
    return out


# ─────────────────────────────────────────────
# N3 — SPIKE CORRECTION  (median replacement)
# ─────────────────────────────────────────────
def correct_spikes(cube, spike_bands, good_mask):
    if not spike_bands:
        return cube
    out = cube.copy()
    for bi in spike_bands:
        if not good_mask[bi]:
            continue
        band = out[:, :, bi]
        vm   = np.isfinite(band) & (band < IF_MAX) & (band > IF_MIN)
        if not vm.any():
            continue
        mf   = median_filter(band, size=3)
        std  = float(band[vm].std())
        spk  = (np.abs(band - mf) > 4.0 * std) & vm
        out[:, :, bi][spk] = mf[spk]
    return out


# ─────────────────────────────────────────────
# N5 — SATURATION CORRECTION  (spectral interp)
# ─────────────────────────────────────────────
def correct_saturation(cube, sat_bands, good_mask):
    if not sat_bands:
        return cube
    out = cube.copy()
    B   = cube.shape[2]
    for bi in sat_bands:
        if not good_mask[bi]:
            continue
        band = out[:, :, bi]
        vm   = np.isfinite(band) & (band < IF_MAX) & (band > IF_MIN)
        if not vm.any():
            continue
        v_max = band[vm].max()
        sat   = band >= 0.995 * v_max
        # Replace saturated pixels with linear interpolation from neighbours
        prev_bi = bi - 1 if bi > 0 else None
        next_bi = bi + 1 if bi < B - 1 else None
        if prev_bi is not None and next_bi is not None:
            interp = 0.5 * (out[:, :, prev_bi] + out[:, :, next_bi])
            out[:, :, bi][sat] = interp[sat]
        elif prev_bi is not None:
            out[:, :, bi][sat] = out[:, :, prev_bi][sat]
        elif next_bi is not None:
            out[:, :, bi][sat] = out[:, :, next_bi][sat]
    return out


# ─────────────────────────────────────────────
# N4 — THERMAL CORRECTION  (vectorised Planck)
# ─────────────────────────────────────────────
def _planck_bb(lam_um, T_K, epsilon):
    """Spectral radiance (arbitrary units) for a Planck blackbody."""
    h = 6.626e-34; c = 2.998e8; k = 1.381e-23
    lam = lam_um * 1e-6
    exp = np.clip(h * c / (lam * k * T_K), 0, 700)
    return epsilon / (lam ** 5 * (np.exp(exp) - 1.0 + 1e-30))


def correct_thermal_vectorised(cube, thermal_bands, waves):
    """
    FIX-PH-VECT: Fully vectorised Planck subtraction.

    Previous version looped over every (row, col) pixel — O(H×W) Python
    iterations, ~15 s for a 480×640 scene.

    New approach:
      1. Fit Planck to the scene-median spectrum in the thermal window.
      2. Compute a per-pixel scale = pixel_mean_thermal / model_mean_thermal
         using NumPy broadcasting — no Python loops.
      3. Subtract scale × Planck model from each pixel in one operation.

    This reduces 307,200 iterations to a handful of array operations.
    """
    if len(thermal_bands) < 4:
        return cube
    tb = np.array(sorted(thermal_bands))
    tw = waves[tb]
    print(f"  [N4 Thermal] Vectorised Planck subtraction "
          f"({len(tb)} bands, {tw[0]:.2f}–{tw[-1]:.2f} µm)")

    # Scene-median spectrum in thermal window
    thermal_cube = cube[:, :, tb]   # (H, W, len(tb))
    median_spec  = np.nanmedian(
        thermal_cube.reshape(-1, len(tb)), axis=0)   # (len(tb),)
    median_spec  = np.clip(median_spec, 1e-9, None)

    # Fit Planck to scene median
    fitted_ok = False
    try:
        popt, _ = curve_fit(
            _planck_bb, tw, median_spec,
            p0=[240.0, 1e10],
            bounds=([180.0, 0.0], [320.0, 1e14]),
            maxfev=3000)
        T_fit, eps_fit = popt
        if T_fit <= 185 or T_fit >= 315:
            raise RuntimeError(f"T={T_fit:.1f}K at bound")
        fitted_ok = True
        print(f"     Fitted T={T_fit:.1f} K, ε={eps_fit:.3e}")
    except Exception as e:
        print(f"     [WARN] Planck fit failed ({e}) — percentile fallback")

    out = cube.copy()
    if fitted_ok:
        model = _planck_bb(tw, T_fit, eps_fit)   # (len(tb),)
        model_mean = model.mean()
        # Per-pixel mean in thermal window
        pix_mean = np.nanmean(thermal_cube, axis=2)   # (H, W)
        # Scale: (H, W, 1) broadcast × (len(tb),)
        scale = (pix_mean / (model_mean + 1e-30))[:, :, np.newaxis]
        correction = scale * model[np.newaxis, np.newaxis, :]  # (H,W,len(tb))
        out[:, :, tb] = thermal_cube - correction
    else:
        # Percentile fallback: subtract 80% of 95th percentile per band
        for bi in thermal_bands:
            band = cube[:, :, bi]
            vm   = np.isfinite(band)
            if vm.any():
                p95 = np.percentile(band[vm], 95)
                out[:, :, bi] = np.where(vm, band - p95 * 0.8, band)

    return out


# ─────────────────────────────────────────────
# ATMOSPHERIC CORRECTION  (multiplicative anchor)
# ─────────────────────────────────────────────
_ATM_ANCHOR_WINDOWS = [(1.00, 1.10), (2.45, 2.58)]

def atmospheric_correction(cube, waves, good_mask):
    """
    Multiplicative anchor correction (FIX-PH-2b retained and improved).

    The anchor reflectance adapts to the scene: instead of a fixed
    dark-basalt value of 0.05, we use the 10th percentile of the
    scene median in each anchor window as the target.  This handles
    bright surfaces (ice, carbonate) without over-darkening.

    Scale is clamped to [0.5, 2.0] to prevent explosion.
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(float)
    phys = (flat > IF_MIN) & (flat < IF_MAX) & np.isfinite(flat)
    flat_c = np.where(phys, flat, np.nan)
    scene_med = np.nanmedian(flat_c, axis=0)   # (B,)

    anchor_idx, anchor_scale = [], []
    for lo, hi in _ATM_ANCHOR_WINDOWS:
        win = [(i, waves[i]) for i in range(B)
               if lo <= waves[i] <= hi and good_mask[i]
               and np.isfinite(scene_med[i]) and scene_med[i] > 1e-4]
        if not win:
            continue
        w_idx = [x[0] for x in win]
        w_med = float(np.median(scene_med[w_idx]))
        # Adaptive target: 10th percentile of valid pixels in window
        win_vals = flat_c[:, w_idx]
        valid_w  = win_vals[np.all(np.isfinite(win_vals), axis=1)]
        if valid_w.size:
            target = float(np.percentile(valid_w, 10))
            target = max(target, 0.02)   # floor at 2% to avoid blackening
        else:
            target = 0.05
        sf = float(np.clip(target / (w_med + 1e-6), 0.5, 2.0))
        for idx, _ in win:
            anchor_idx.append(idx)
            anchor_scale.append(sf)

    if len(anchor_idx) < 2:
        print("  [ATM] Fewer than 2 anchor bands — skipping.")
        return cube, 'none'

    all_bands = np.arange(B, dtype=float)
    scale_full = np.interp(all_bands,
                           np.array(anchor_idx, float),
                           np.array(anchor_scale, float),
                           left=anchor_scale[0],
                           right=anchor_scale[-1])

    out = cube.copy()
    for bi in range(B):
        if not good_mask[bi]:
            continue
        band = cube[:, :, bi]
        vm   = np.isfinite(band) & (band > IF_MIN) & (band < IF_MAX)
        out[:, :, bi] = np.where(vm, band * scale_full[bi], band)

    out = np.clip(out, IF_MIN, IF_MAX)
    corr_med = float(np.nanmedian(out[np.isfinite(out) & (out > 0)]))
    print(f"  [ATM] Scale range: [{scale_full[good_mask].min():.3f}, "
          f"{scale_full[good_mask].max():.3f}]  "
          f"Post-correction median I/F: {corr_med:.4f}")
    return out, 'multiplicative_anchor'


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_physics_correction(data_dir=DATA_DIR,
                           eda_dir=EDA_DIR,
                           noise_dir=NOISE_DIR,
                           output_dir=OUTPUT_DIR):
    print("=" * 65)
    print("  STAGE 3: PHYSICS CORRECTION")
    print("  Corrects: N3 Spikes | N4 Thermal | N5 Saturation |"
          " Illumination | Atmosphere")
    print("  Passes through: N1 Stripes & N2 Gaussian → Stage 4 ML")
    print("=" * 65)

    eda_json   = os.path.join(eda_dir,   'eda_summary.json')
    noise_json = os.path.join(noise_dir, 'noise_map.json')
    for p in (eda_json, noise_json):
        if not os.path.exists(p):
            print(f"  ❌ Missing: {p}"); return None

    with open(eda_json)   as f: eda = json.load(f)
    with open(noise_json) as f: nm  = json.load(f)

    img_files = sorted(
        glob.glob(os.path.join(data_dir, '**', '*.img'), recursive=True) +
        glob.glob(os.path.join(data_dir, '*.img'))
    )
    seen = set()
    img_files = [f for f in img_files
                 if not (os.path.basename(f) in seen
                         or seen.add(os.path.basename(f)))]

    logs = {}

    for img_path in img_files:
        sname = os.path.splitext(os.path.basename(img_path))[0]
        if sname not in nm:
            continue

        print(f"\n{'━'*55}")
        print(f"  {sname}")

        lbl_path = find_label(img_path)
        meta = parse_pds3_label(lbl_path) if lbl_path else {}
        try:
            cube = load_crism_cube(img_path, meta)
        except Exception as e:
            print(f"  ❌ {e}"); continue

        H, W, B = cube.shape
        waves    = get_sensor_waves(B, meta.get('wavelengths'))
        bad_mask = build_bad_mask(waves)
        good_mask = ~bad_mask

        dead = set(eda.get(sname, {}).get('dead_bands', []))
        noise = nm[sname]

        # Mask sentinels + unphysical
        cube = cube.astype(float)
        cube[(cube >= 65534) | (cube > IF_MAX) | (cube < IF_MIN)] = np.nan
        for d in dead:
            if d < B: cube[:, :, d] = np.nan
        for bi in np.where(bad_mask)[0]:
            cube[:, :, bi] = np.nan

        pre_snr = snr_good_bands(cube, good_mask)

        print("  → Illumination correction")
        cube = illumination_correction(cube, good_mask)

        print(f"  → N3 Spike correction ({len(noise['spike_bands'])} bands)")
        cube = correct_spikes(cube, noise['spike_bands'], good_mask)

        print(f"  → N5 Saturation correction ({len(noise['saturated_bands'])} bands)")
        cube = correct_saturation(cube, noise['saturated_bands'], good_mask)

        print(f"  → N4 Thermal correction ({len(noise['thermal_bands'])} bands)")
        cube = correct_thermal_vectorised(cube, noise['thermal_bands'], waves)

        print("  → Atmospheric correction")
        cube, atm_method = atmospheric_correction(cube, waves, good_mask)

        post_snr = snr_good_bands(cube, good_mask)
        snr_pct  = ((post_snr - pre_snr) / pre_snr * 100) if pre_snr > 0 else 0.0
        print(f"  SNR (good bands): {pre_snr:.3f} → {post_snr:.3f}  "
              f"Δ = {snr_pct:+.1f}%")

        # Determine output subdirectory
        rel = os.path.relpath(img_path, data_dir)
        parts = rel.replace('\\', '/').split('/')
        sub_dir = parts[0] if len(parts) > 1 else '.'
        out_sub = os.path.join(output_dir, sub_dir)
        os.makedirs(out_sub, exist_ok=True)

        out_path = os.path.join(out_sub, f'{sname}_S3.npy')
        np.save(out_path, cube.astype(np.float32))
        print(f"  → Saved: {out_path}")

        logs[sname] = {
            'snr_before': round(pre_snr, 4),
            'snr_after':  round(post_snr, 4),
            'snr_delta_pct': round(snr_pct, 2),
            'atm_method': atm_method,
            'thermal_corrected': len(noise['thermal_bands']) > 0,
            'spikes_corrected':  len(noise['spike_bands']) > 0,
            'sat_corrected':     len(noise['saturated_bands']) > 0,
            'output_path': out_path,
        }

    with open(os.path.join(output_dir, 'stage3_log.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"\n  ✅ Stage 3 complete  →  {output_dir}")
    return logs


if __name__ == '__main__':
    run_physics_correction()