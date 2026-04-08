#!/usr/bin/env python3
"""
stage1_eda.py  —  CRISM Exploratory Data Analysis
==================================================
Outputs
-------
  eda_output/eda_summary.json   → consumed by all downstream stages
  eda_output/eda_summary.csv
  eda_output/<name>_band_stats.png
  eda_output/cross_sample_*.png

Usage
-----
  python stage1_eda.py
  CRISM_IMG=path/to/file.img python stage1_eda.py
"""

import os, glob, json, warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from crism_utils import (
    parse_pds3_label, load_crism_cube, find_label,
    get_sensor_waves, build_bad_mask,
    IF_MAX, IF_MIN,
)

# ─────────────────────────────────────────────
# CONFIG  — override via environment variables
# ─────────────────────────────────────────────
DATA_DIR   = os.environ.get("CRISM_DATA_DIR",
             r"D:\NEW CROSS MISSION\Data\Scene 8\Crism")
OUTPUT_DIR = os.path.join(DATA_DIR, "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISY_STD_PCT           = 95
LOW_SIGNAL_THRESHOLD    = 1e-4
DEAD_BAND_VAR_THRESHOLD = 1e-12

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor':  '#161b22',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d', 'grid.alpha': 0.5,
    'font.size': 10,
})


# ─────────────────────────────────────────────
# BAND STATISTICS
# ─────────────────────────────────────────────
def compute_band_stats(cube: np.ndarray) -> dict:
    H, W, B = cube.shape
    total   = H * W
    stats   = {k: np.zeros(B) for k in
               ('mean', 'std', 'min', 'max', 'variance', 'nan_frac')}
    for b in range(B):
        band = cube[:, :, b]
        nan_mask = ~np.isfinite(band)
        stats['nan_frac'][b] = nan_mask.sum() / total
        v = band[np.isfinite(band) & (band < IF_MAX) & (band > IF_MIN)]
        if v.size:
            stats['mean'][b]     = v.mean()
            stats['std'][b]      = v.std()
            stats['min'][b]      = v.min()
            stats['max'][b]      = v.max()
            stats['variance'][b] = v.var()
    return stats


def classify_bands(stats: dict, n_bands: int) -> dict:
    std_v = stats['std']
    var_v = stats['variance']
    mean_v = stats['mean']

    positive_stds = std_v[std_v > 0]
    noisy_thr = (np.percentile(positive_stds, NOISY_STD_PCT)
                 if positive_stds.size else np.inf)

    dead, noisy, low_sig = [], [], []
    for b in range(n_bands):
        if var_v[b] < DEAD_BAND_VAR_THRESHOLD:
            dead.append(b)
        elif std_v[b] > noisy_thr:
            noisy.append(b)
        if abs(mean_v[b]) < LOW_SIGNAL_THRESHOLD:
            low_sig.append(b)

    valid = [b for b in range(n_bands) if b not in dead]
    return {'dead': dead, 'noisy': noisy, 'low_signal': low_sig, 'valid': valid}


# ─────────────────────────────────────────────
# PER-SAMPLE PLOT
# ─────────────────────────────────────────────
def plot_sample(name, waves, stats, clf, out_dir):
    B   = len(waves)
    idx = np.arange(B)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Band statistics — {name}', color='#58a6ff',
                 fontsize=13, fontweight='bold', y=0.99)

    ax = axes[0]
    ax.plot(idx, stats['mean'], color='#58a6ff', lw=1.2, label='Mean I/F')
    ax.fill_between(idx, stats['mean'] - stats['std'],
                    stats['mean'] + stats['std'], alpha=0.12, color='#58a6ff')
    ax.axhline(IF_MAX, color='#ff7b72', lw=0.8, ls='--',
               label=f'Physical max ({IF_MAX})')
    if clf['dead']:
        ax.scatter(clf['dead'], stats['mean'][clf['dead']],
                   c='#ff7b72', s=25, zorder=5, marker='x',
                   label=f"Dead ({len(clf['dead'])})")
    if clf['noisy']:
        ax.scatter(clf['noisy'], stats['mean'][clf['noisy']],
                   c='#ffa657', s=25, zorder=5, marker='^',
                   label=f"Noisy ({len(clf['noisy'])})")
    ax.set_ylabel('Mean I/F'); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_title('Mean reflectance', fontsize=10, color='#8b949e')

    axes[1].plot(idx, stats['std'], color='#3fb950', lw=1.0)
    axes[1].set_ylabel('Std dev'); axes[1].grid(True, alpha=0.25)
    axes[1].set_title('Standard deviation per band', fontsize=10, color='#8b949e')

    axes[2].bar(idx, stats['nan_frac'] * 100, color='#d2a8ff', width=1.0)
    axes[2].set_ylabel('NaN %'); axes[2].grid(True, alpha=0.25)
    axes[2].set_title('NaN fraction per band', fontsize=10, color='#8b949e')

    # Wavelength axis on bottom
    ax4 = axes[3]
    ax4.plot(idx, waves, color='#79c0ff', lw=1.0)
    ax4.set_ylabel('Wavelength (µm)'); ax4.set_xlabel('Band index')
    ax4.grid(True, alpha=0.25)
    ax4.set_title('Wavelength axis (from label or empirical table)',
                  fontsize=10, color='#8b949e')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, f'{name}_band_stats.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out


# ─────────────────────────────────────────────
# MAIN EDA
# ─────────────────────────────────────────────
def run_eda(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    print("=" * 65)
    print("  STAGE 1: EDA PIPELINE")
    print("=" * 65)
    print(f"  Data dir  : {data_dir}")
    print(f"  Output dir: {output_dir}\n")

    # Collect .img files
    img_files = sorted(
        glob.glob(os.path.join(data_dir, '**', '*.img'), recursive=True) +
        glob.glob(os.path.join(data_dir, '*.img'))
    )
    # Deduplicate by basename
    seen = set()
    img_files = [f for f in img_files
                 if not (os.path.basename(f) in seen
                         or seen.add(os.path.basename(f)))]

    if not img_files:
        print("  ❌ No .img files found.")
        return None, None

    print(f"  Found {len(img_files)} .img file(s)\n")

    all_results  = OrderedDict()
    summary_rows = []

    for idx, img_path in enumerate(img_files):
        fname = os.path.basename(img_path)
        sname = os.path.splitext(fname)[0]
        print(f"\n{'━'*55}")
        print(f"  [{idx+1}/{len(img_files)}] {fname}")

        lbl_path = find_label(img_path)
        meta = parse_pds3_label(lbl_path) if lbl_path else {}
        if lbl_path:
            print(f"  Label: {os.path.basename(lbl_path)}")
        else:
            print("  [WARN] No label found — using heuristics")

        try:
            cube = load_crism_cube(img_path, meta)
        except Exception as e:
            print(f"  ❌ Load error: {e}")
            all_results[sname] = {'error': str(e),
                                  'file_bytes': os.path.getsize(img_path)}
            continue

        H, W, B = cube.shape
        waves   = get_sensor_waves(B, meta.get('wavelengths'))
        bad_mask = build_bad_mask(waves)

        print(f"  Cube: {H} × {W} × {B}  |  "
              f"Sensor: {'L' if B > 200 else 'S'}  |  "
              f"dtype: {cube.dtype}")
        print(f"  Wavelength range: {waves[0]:.3f} – {waves[-1]:.3f} µm  "
              f"({'label' if meta.get('wavelengths') else 'empirical table'})")

        # I/F sanity
        finite = cube[np.isfinite(cube)]
        if finite.size:
            raw_min, raw_max = float(finite.min()), float(finite.max())
            frac_hi = float(np.mean(finite > IF_MAX))
            frac_lo = float(np.mean(finite < 0))
            print(f"  I/F range: [{raw_min:.4f}, {raw_max:.4f}]  "
                  f"| >{IF_MAX}: {frac_hi*100:.2f}%  "
                  f"| <0: {frac_lo*100:.2f}%")
            if frac_hi > 0.001:
                print("  [WARN] Unphysical I/F values — check calibration.")

        stats = compute_band_stats(cube)
        clf   = classify_bands(stats, B)
        print(f"  Bands: {B} total | {len(clf['valid'])} valid | "
              f"{len(clf['dead'])} dead | {len(clf['noisy'])} noisy | "
              f"{len(clf['low_signal'])} low-signal")

        plot_sample(sname, waves, stats, clf, output_dir)

        all_results[sname] = {
            'total_bands':      B,
            'valid_bands':      clf['valid'],
            'dead_bands':       clf['dead'],
            'noisy_bands':      clf['noisy'],
            'low_signal_bands': clf['low_signal'],
            'spatial_dims':     [H, W],
            'file_bytes':       os.path.getsize(img_path),
            'sensor':           meta.get('sensor_id', 'L' if B > 200 else 'S'),
            'wavelength_source': 'label' if meta.get('wavelengths') else 'empirical',
            'wave_min':         float(waves[0]),
            'wave_max':         float(waves[-1]),
            'if_min':           float(finite.min()) if finite.size else 0.0,
            'if_max':           float(finite.max()) if finite.size else 0.0,
            'bad_band_count':   int(bad_mask.sum()),
        }
        summary_rows.append({
            'Sample': sname, 'H': H, 'W': W, 'Bands': B,
            'Sensor': 'L' if B > 200 else 'S',
            'Valid': len(clf['valid']), 'Dead': len(clf['dead']),
            'Noisy': len(clf['noisy']), 'LowSig': len(clf['low_signal']),
            'Wave_min': f"{waves[0]:.3f}", 'Wave_max': f"{waves[-1]:.3f}",
            'IF_min': f"{all_results[sname]['if_min']:.4f}",
            'IF_max': f"{all_results[sname]['if_max']:.4f}",
            'Bytes': os.path.getsize(img_path),
        })
        del cube

    # Save outputs
    df = pd.DataFrame(summary_rows)
    print(f"\n{'═'*55}\n  SUMMARY\n{'═'*55}")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(output_dir, 'eda_summary.csv'), index=False)

    # JSON — strip numpy types
    json_safe = {}
    for s, v in all_results.items():
        json_safe[s] = {
            k: (val.tolist() if isinstance(val, np.ndarray)
                else (int(val) if isinstance(val, (np.integer,))
                      else (float(val) if isinstance(val, (np.floating,))
                            else val)))
            for k, val in v.items()
        }
    with open(os.path.join(output_dir, 'eda_summary.json'), 'w') as f:
        json.dump(json_safe, f, indent=2)

    print(f"\n  ✅ EDA complete  →  {output_dir}")
    return all_results, df


if __name__ == '__main__':
    run_eda()