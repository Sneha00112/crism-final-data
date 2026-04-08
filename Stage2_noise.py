#!/usr/bin/env python3
"""
stage2_noise.py  —  CRISM Noise Characterisation
=================================================
Identifies the noises that affect mineral identification accuracy:

  N1  Column-correlated stripe noise
       Origin : detector readout non-uniformity; persists across ALL bands
       Impact : false absorption troughs / peaks aligned with columns;
                corrupts BD2300, BD2100, BD1900 when stripe amplitude
                exceeds the band-depth threshold (~0.02)

  N2  Gaussian random noise (low SNR bands)
       Origin : shot noise, read noise; highest at long-λ and near
                atmospheric absorptions
       Impact : noisy spectra give unreliable continuum fits;
                false positives in single-band indices

  N3  Spectral spike artefacts
       Origin : cosmic-ray hits and detector hot pixels; isolated to
                one or a few bands
       Impact : single-band index (e.g. BD1435 for CO2-ice) can fire
                on a cosmic-ray spike

  N4  Thermal emission excess (L-sensor, λ > 3.0 µm)
       Origin : detector self-emission increases with T; planetary
                thermal emission; not mineralogical
       Impact : raises continuum at long λ; corrupts BD3400, BD3500

  N5  Detector saturation (very bright surfaces or specular reflection)
       Impact : saturated bands show clipped signal; index = 0 or NaN

All five affect mineral identification; the pipeline later decides which
are correctable by physics (N3, N4, N5) vs. ML (N1 Stripe, N2 Gaussian).

Outputs
-------
  noise_output/noise_map.json    → consumed by stages 3 & 4
  noise_output/<n>_snr.png
"""

import os, glob, json, re, warnings
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

DATA_DIR   = os.environ.get("CRISM_DATA_DIR",
             r"D:\NEW CROSS MISSION\Data\Scene 8\Crism")
EDA_DIR    = os.path.join(DATA_DIR, "eda_output")
OUTPUT_DIR = os.path.join(DATA_DIR, "noise_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAN_SKIP_FRAC = 0.80   # skip band if > 80% pixels are NaN

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9', 'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
})


# ═══════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════
def preprocess(cube: np.ndarray, dead_bands: list,
               bad_mask: np.ndarray) -> np.ndarray:
    """
    Mask sentinels, unphysical I/F, dead bands, and detector-artefact
    bands BEFORE any noise statistic is computed.
    """
    c = cube.copy().astype(float)
    c[(c >= 65534) | (c > IF_MAX) | (c < IF_MIN)] = np.nan
    for d in dead_bands:
        c[:, :, d] = np.nan
    for bi in np.where(bad_mask)[0]:
        c[:, :, bi] = np.nan
    return c


# ═══════════════════════════════════════════════════
# N1 — COLUMN-CORRELATED STRIPE NOISE
# ═══════════════════════════════════════════════════
def detect_stripes(cube: np.ndarray, bad_mask: np.ndarray) -> dict:
    """
    Stripe noise = per-column DC offset that is correlated across
    bands.  Method:
      1. For each band, compute column means.
      2. Subtract the spatial (row-mean) to get the residual column
         pattern (strip out real surface structure).
      3. Score = std of column-mean residuals.  High score → stripes.
      4. A band is flagged as striped if its score exceeds mean + 2σ
         of the distribution across good bands.

    Also computes cross-band stripe correlation: if the column offset
    pattern is correlated across bands, we have genuine detector
    striping (not just a bright feature).
    """
    H, W, B = cube.shape
    col_residual_std = np.zeros(B)   # per-band stripe score
    col_patterns     = np.full((B, W), np.nan)

    for bi in range(B):
        if bad_mask[bi]:
            continue
        band = cube[:, :, bi]
        vm   = np.isfinite(band)
        if vm.mean() < (1 - NAN_SKIP_FRAC):
            continue
        col_means = np.array([
            np.nanmean(band[:, c]) if vm[:, c].any() else np.nan
            for c in range(W)])
        if not np.any(np.isfinite(col_means)):
            continue
        col_means -= np.nanmean(col_means)   # remove global offset
        col_patterns[bi] = col_means
        col_residual_std[bi] = float(np.nanstd(col_means))

    good_scores = col_residual_std[~bad_mask & (col_residual_std > 0)]
    if good_scores.size < 2:
        return {'striped_bands': [], 'stripe_scores': col_residual_std.tolist(),
                'cross_band_corr': 0.0, 'stripe_amplitude': 0.0}

    mu, sig = good_scores.mean(), good_scores.std()
    threshold = mu + 2.0 * sig
    striped_bands = [int(bi) for bi in range(B)
                     if not bad_mask[bi] and col_residual_std[bi] > threshold]

    # Cross-band stripe correlation (key diagnostic)
    # If column patterns in different bands are correlated → real detector stripe
    good_bi = [bi for bi in range(B)
               if not bad_mask[bi] and np.any(np.isfinite(col_patterns[bi]))]
    cross_corrs = []
    for i in range(min(20, len(good_bi))):
        for j in range(i + 1, min(20, len(good_bi))):
            p1 = col_patterns[good_bi[i]]
            p2 = col_patterns[good_bi[j]]
            ok = np.isfinite(p1) & np.isfinite(p2)
            if ok.sum() > 5:
                c12 = float(np.corrcoef(p1[ok], p2[ok])[0, 1])
                if np.isfinite(c12):
                    cross_corrs.append(abs(c12))

    mean_cross_corr = float(np.mean(cross_corrs)) if cross_corrs else 0.0
    stripe_amp = float(np.nanmean(col_residual_std[striped_bands])) \
        if striped_bands else 0.0

    print(f"  [N1 Stripes] {len(striped_bands)} striped bands  "
          f"| cross-band corr: {mean_cross_corr:.3f}  "
          f"| amplitude: {stripe_amp:.5f} I/F")
    if mean_cross_corr > 0.3:
        print("    → Cross-band correlation confirms detector stripe origin")

    return {
        'striped_bands':   striped_bands,
        'stripe_scores':   col_residual_std.tolist(),
        'cross_band_corr': mean_cross_corr,
        'stripe_amplitude': stripe_amp,
    }


# ═══════════════════════════════════════════════════
# N2 — GAUSSIAN RANDOM NOISE  (SNR per band)
# ═══════════════════════════════════════════════════
def detect_gaussian_noise(cube: np.ndarray,
                          bad_mask: np.ndarray) -> dict:
    """
    Compute per-band SNR = |mean| / std on spatially-valid pixels.
    Bands with SNR < 5 are high-noise; 5–15 medium-noise.

    For mineral identification, the critical threshold is whether
    the noise floor (std) exceeds typical band-depth values (~0.02–0.05).
    We report this directly.
    """
    H, W, B = cube.shape
    snr  = np.zeros(B)
    noise_floor = np.zeros(B)   # std in I/F units

    for bi in range(B):
        if bad_mask[bi]:
            continue
        v = cube[:, :, bi]
        v = v[np.isfinite(v)]
        if v.size / (H * W) < (1 - NAN_SKIP_FRAC):
            continue
        mu_v  = np.abs(np.mean(v))
        std_v = np.std(v)
        noise_floor[bi] = std_v
        snr[bi] = (mu_v / std_v) if std_v > 1e-8 else 0.0

    high_noise = [int(b) for b in range(B)
                  if not bad_mask[b] and 0 < snr[b] < 5]
    med_noise  = [int(b) for b in range(B)
                  if not bad_mask[b] and 5 <= snr[b] < 15]

    # Bands where noise floor exceeds BD threshold (0.02 I/F)
    # — these will give unreliable mineral indices
    BD_THRESHOLD = 0.02
    noisy_for_mineralogy = [int(b) for b in range(B)
                            if not bad_mask[b] and noise_floor[b] > BD_THRESHOLD]

    print(f"  [N2 Gaussian] high-noise: {len(high_noise)}  "
          f"| med-noise: {len(med_noise)}  "
          f"| noise>BD_threshold: {len(noisy_for_mineralogy)}")

    return {
        'snr_per_band':    snr.tolist(),
        'noise_floor':     noise_floor.tolist(),
        'high_noise_bands': high_noise,
        'medium_noise_bands': med_noise,
        'bands_noisy_for_mineralogy': noisy_for_mineralogy,
    }


# ═══════════════════════════════════════════════════
# N3 — SPECTRAL SPIKE ARTEFACTS
# ═══════════════════════════════════════════════════
def detect_spikes(cube: np.ndarray, bad_mask: np.ndarray) -> dict:
    """
    A spectral spike = isolated bright/dark pixel in one band
    that is statistically inconsistent with its spectral neighbourhood.
    Uses MAD-based outlier detection in the spectral domain.
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    med_sp = np.nanmedian(flat, axis=0)
    mad_sp = np.nanmedian(np.abs(flat - med_sp), axis=0)
    mad_sp[mad_sp < 1e-5] = 1e-5

    spike_counts = np.zeros(B)
    for bi in range(B):
        if bad_mask[bi]:
            continue
        ok = np.isfinite(flat[:, bi])
        if ok.sum() == 0:
            continue
        z = np.abs(flat[:, bi] - med_sp[bi]) / mad_sp[bi]
        spike_counts[bi] = np.sum(z[ok] > 6.0)   # 6-MAD threshold

    spike_bands = [int(b) for b in range(B)
                   if not bad_mask[b]
                   and spike_counts[b] / (H * W) > 0.01]   # >1% pixels spiked
    print(f"  [N3 Spikes]   {len(spike_bands)} spike-affected bands")
    return {'spike_bands': spike_bands,
            'spike_counts': spike_counts.tolist()}


# ═══════════════════════════════════════════════════
# N4 — THERMAL EMISSION EXCESS
# ═══════════════════════════════════════════════════
def detect_thermal(cube: np.ndarray, waves: np.ndarray,
                   bad_mask: np.ndarray) -> dict:
    """
    Thermal emission from the Martian surface starts contributing
    significantly at λ > 3.0 µm (band ~330 for 438-band L-sensor).

    Detection: fit a monotonically increasing trend in the
    3.0–3.92 µm window.  If the mean I/F in that window exceeds
    the extrapolated continuum from 2.5–3.0 µm, thermal is present.
    """
    B = len(waves)
    if waves[-1] < 2.5:
        return {'thermal_bands': [], 'thermal_present': False}

    # Continuum window: 2.45–2.59 µm (between CO2 and deep CO2)
    cont_mask = (waves >= 2.45) & (waves <= 2.59) & (~bad_mask)
    # Thermal window: 3.23–3.55 µm (between minor CO2 and thermal tail)
    therm_mask = (waves >= 3.23) & (waves <= 3.55) & (~bad_mask)

    if cont_mask.sum() < 2 or therm_mask.sum() < 2:
        return {'thermal_bands': [], 'thermal_present': False}

    means = np.nanmean(cube, axis=(0, 1))
    cont_mean  = float(np.nanmean(means[cont_mask]))
    therm_mean = float(np.nanmean(means[therm_mask]))

    thermal_present = therm_mean > cont_mean * 1.05   # 5% excess

    # Identify specific bands
    thermal_bands = []
    if thermal_present:
        # All bands where mean > continuum + 2σ of continuum region
        cont_std = float(np.nanstd(means[cont_mask]))
        thr = cont_mean + 2.0 * cont_std
        thermal_bands = [int(b) for b in range(B)
                         if waves[b] > 3.0 and not bad_mask[b]
                         and means[b] > thr]
        # Physical cap: 3.0–3.92 µm is ~80 bands on L-sensor
        if len(thermal_bands) > 80:
            thermal_bands = thermal_bands[:80]

    print(f"  [N4 Thermal]  present={thermal_present}  "
          f"({len(thermal_bands)} bands)  "
          f"cont={cont_mean:.4f}  therm={therm_mean:.4f} I/F")
    return {
        'thermal_bands': thermal_bands,
        'thermal_present': thermal_present,
        'continuum_mean': cont_mean,
        'thermal_mean': therm_mean,
    }


# ═══════════════════════════════════════════════════
# N5 — SATURATION
# ═══════════════════════════════════════════════════
def detect_saturation(cube: np.ndarray, bad_mask: np.ndarray) -> dict:
    """
    Saturated pixels = clipped at or near detector full-well.
    Physically, after unphysical-I/F masking, genuine saturation
    shows as a band where a large fraction of pixels sit exactly
    at the maximum observed value (within 0.5%).
    """
    H, W, B = cube.shape
    sat_bands = []
    sat_ratios = {}

    for bi in range(B):
        if bad_mask[bi]:
            continue
        band = cube[:, :, bi]
        v = band[np.isfinite(band)]
        if v.size == 0:
            continue
        v_max = v.max()
        if v_max <= 0:
            continue
        near_max = np.sum(v >= 0.995 * v_max) / v.size
        if near_max > 0.05:
            sat_bands.append(bi)
            sat_ratios[bi] = float(near_max)

    print(f"  [N5 Saturat.] {len(sat_bands)} saturated bands")
    return {'saturated_bands': sat_bands, 'saturation_ratios': sat_ratios}


# ═══════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════
def plot_snr(sname, waves, snr_arr, bad_mask,
             striped, high_n, output_dir):
    B   = len(waves)
    idx = np.arange(B)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Noise characterisation — {sname}',
                 color='#58a6ff', fontsize=13, fontweight='bold')

    # SNR plot
    colors = ['#444444' if bad_mask[b] else
              ('#ff7b72' if snr_arr[b] < 5 and snr_arr[b] > 0 else
               ('#ffa657' if snr_arr[b] < 15 else '#3fb950'))
              for b in range(B)]
    ax1.bar(idx, snr_arr, color=colors, width=1.0)
    ax1.axhline(5,  color='#ff7b72', ls='--', lw=0.8, label='SNR=5')
    ax1.axhline(15, color='#ffa657', ls='--', lw=0.8, label='SNR=15')
    ax1.set_ylabel('SNR (|mean|/std)'); ax1.legend(fontsize=8)
    ax1.set_title('Per-band SNR (colored by noise class)', fontsize=10,
                  color='#8b949e')
    ax1.grid(axis='y', alpha=0.2)

    # Mark stripe bands
    for b in striped:
        ax1.axvline(b, color='#d2a8ff', alpha=0.4, lw=0.6)
    for b in high_n:
        ax1.axvline(b, color='#ff7b72', alpha=0.25, lw=0.6)

    # Wavelength vs band
    ax2.plot(idx, waves, color='#79c0ff', lw=1.0)
    ax2.set_xlabel('Band index'); ax2.set_ylabel('Wavelength (µm)')
    ax2.grid(alpha=0.2)
    ax2.set_title('Wavelength axis', fontsize=10, color='#8b949e')

    from matplotlib.patches import Patch
    legend_els = [
        Patch(color='#3fb950', label='Low noise (SNR≥15)'),
        Patch(color='#ffa657', label='Med noise (5–15)'),
        Patch(color='#ff7b72', label='High noise (<5)'),
        Patch(color='#444444', label='Bad band (masked)'),
        Patch(color='#d2a8ff', label='Stripe band'),
    ]
    ax1.legend(handles=legend_els, fontsize=7, loc='upper right')
    plt.tight_layout()
    out = os.path.join(output_dir, f'{sname}_snr.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def run_noise_characterisation(data_dir=DATA_DIR,
                               eda_dir=EDA_DIR,
                               output_dir=OUTPUT_DIR):
    print("=" * 65)
    print("  STAGE 2: NOISE CHARACTERISATION")
    print("  Targets: N1 Stripes | N2 Gaussian | N3 Spikes |"
          " N4 Thermal | N5 Saturation")
    print("=" * 65)

    eda_json = os.path.join(eda_dir, 'eda_summary.json')
    if not os.path.exists(eda_json):
        print(f"  ❌ EDA summary not found at {eda_json}")
        print("     Run stage1_eda.py first.")
        return None

    with open(eda_json) as f:
        eda = json.load(f)

    img_files = sorted(
        glob.glob(os.path.join(data_dir, '**', '*.img'), recursive=True) +
        glob.glob(os.path.join(data_dir, '*.img'))
    )
    seen = set()
    img_files = [f for f in img_files
                 if not (os.path.basename(f) in seen
                         or seen.add(os.path.basename(f)))]

    noise_map = {}

    for img_path in img_files:
        sname = os.path.splitext(os.path.basename(img_path))[0]
        if sname not in eda:
            print(f"  Skipping {sname} — not in EDA summary")
            continue

        print(f"\n{'━'*55}")
        print(f"  {sname}")

        lbl_path = find_label(img_path)
        meta = parse_pds3_label(lbl_path) if lbl_path else {}
        try:
            cube_raw = load_crism_cube(img_path, meta)
        except Exception as e:
            print(f"  ❌ Load error: {e}"); continue

        H, W, B = cube_raw.shape
        waves    = get_sensor_waves(B, meta.get('wavelengths'))
        bad_mask = build_bad_mask(waves)
        dead     = eda[sname].get('dead_bands', [])

        cube = preprocess(cube_raw, dead, bad_mask)

        n1 = detect_stripes(cube, bad_mask)
        n2 = detect_gaussian_noise(cube, bad_mask)
        n3 = detect_spikes(cube, bad_mask)
        n4 = detect_thermal(cube, waves, bad_mask)
        n5 = detect_saturation(cube, bad_mask)

        snr_arr = np.array(n2['snr_per_band'])
        plot_snr(sname, waves, snr_arr, bad_mask,
                 n1['striped_bands'], n2['high_noise_bands'], output_dir)

        # Noise summary table
        rows = [
            ("N1 Stripe",    len(n1['striped_bands']),
             "High" if n1['cross_band_corr'] > 0.3 else "Low",
             "1D CNN per column"),
            ("N2 Gaussian",  len(n2['high_noise_bands']),
             "Medium", "Spectral DAE"),
            ("N3 Spikes",    len(n3['spike_bands']),
             "Low", "Median filter (physics)"),
            ("N4 Thermal",   len(n4['thermal_bands']),
             "High" if n4['thermal_present'] else "None",
             "Planck model subtraction (physics)"),
            ("N5 Saturation",len(n5['saturated_bands']),
             "High" if n5['saturated_bands'] else "None",
             "Interpolation (physics)"),
        ]
        df = pd.DataFrame(rows,
                          columns=['Noise', 'Bands', 'Severity', 'Remedy'])
        print(f"\n  Noise summary for {sname}:")
        print(df.to_string(index=False))

        noise_map[sname] = {
            'bad_mask_applied': True,
            'n1_stripes':   n1,
            'n2_gaussian':  n2,
            'n3_spikes':    n3,
            'n4_thermal':   n4,
            'n5_saturation': n5,
            # Convenience flat lists for downstream stages
            'striped_bands':      n1['striped_bands'],
            'high_noise_bands':   n2['high_noise_bands'],
            'medium_noise_bands': n2['medium_noise_bands'],
            'spike_bands':        n3['spike_bands'],
            'thermal_bands':      n4['thermal_bands'],
            'saturated_bands':    n5['saturated_bands'],
        }
        csv_out = os.path.join(output_dir, f'{sname}_noise.csv')
        df.to_csv(csv_out, index=False)
        del cube_raw, cube

    out_json = os.path.join(output_dir, 'noise_map.json')
    with open(out_json, 'w') as f:
        json.dump(noise_map, f, indent=2)
    print(f"\n  ✅ Noise map saved → {out_json}")
    return noise_map


if __name__ == '__main__':
    run_noise_characterisation()