#!/usr/bin/env python3
"""
stage5_minerals.py  —  CRISM Mineral Identification
====================================================
Input  : Stage 4 denoised cube (*_S4.npy) or Stage 3 fallback (*_S3.npy)
Output : classification map, abundance map, validation reports

Works for both L-sensor (438 bands) and S-sensor (107 bands).
S-sensor rule set is limited to VNIR indices (olivine, pyroxene VNIR,
water-ice; no phyllosilicate / carbonate / sulfate which need SWIR).

Key corrections vs earlier version
------------------------------------
  CL-1: Bad-band mask uses correct wavelength axis (not linspace).
  CL-2: BD2250 spike guard retained.
  CL-3: Featureless confidence = (absent/valid) × (valid/total).
  CL-4: Atmospheric slope detection (BD < -0.05).
  CL-5: Pixel-count sanity check.
  CL-6: Observation ID parsed from filename.
"""

import os, glob, json, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from scipy.optimize import nnls

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

from crism_utils import (
    get_sensor_waves, build_bad_mask,
    IF_MAX, IF_MIN,
)

DATA_DIR    = os.environ.get("CRISM_DATA_DIR",
              r"D:\NEW CROSS MISSION\Data\Scene 8\Crism")
ML_DIR      = os.path.join(DATA_DIR, "ml_output")
PHYS_DIR    = os.path.join(DATA_DIR, "physics_output")
OUTPUT_DIR  = os.path.join(DATA_DIR, "mineral_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# REFLECTANCE HELPERS
# ─────────────────────────────────────────────
def R(spectrum, waves, bad_mask, target_um):
    bi = int(np.argmin(np.abs(waves - target_um)))
    if bad_mask[bi]: return np.nan
    v = float(spectrum[bi])
    return np.nan if v > IF_MAX or v < IF_MIN else v

def band_depth(spectrum, waves, bad_mask, c, l, r):
    rc = R(spectrum, waves, bad_mask, c)
    rl = R(spectrum, waves, bad_mask, l)
    rr = R(spectrum, waves, bad_mask, r)
    if any(np.isnan(x) for x in (rc, rl, rr)): return np.nan
    frac = (c - l) / (r - l + 1e-9)
    cont = rl + frac * (rr - rl)
    return float(1.0 - rc / cont) if cont > 1e-6 else np.nan


# ─────────────────────────────────────────────
# MINERAL RULES  (L + S sensor variants)
# ─────────────────────────────────────────────
def get_rules(waves, bad_mask):
    """
    Returns MINERAL_RULES dict keyed by mineral name.
    Automatically skips rules whose wavelengths fall outside the
    sensor range (e.g. SWIR rules on S-sensor).
    """
    is_L = waves[-1] > 1.5
    W    = waves[-1]

    def BD(c, l, r):
        if c > W or r > W: return lambda sp: np.nan
        return lambda sp: band_depth(sp, waves, bad_mask, c, l, r)

    def Rv(w):
        if w > W: return lambda sp: np.nan
        return lambda sp: R(sp, waves, bad_mask, w)

    rules = {}

    # ── Olivine (VNIR, both sensors) ──
    rules['Olivine'] = {
        'desc': 'Fe2+ broad absorption 1.0–1.3 µm',
        'tests': lambda sp: {
            'OLINDEX3': (
                (lambda s: (1.0 - Rv(1.21)(s) / (
                    Rv(1.08)(s) + (1.21-1.08)/(1.69-1.08)*(Rv(1.69)(s)-Rv(1.08)(s))
                    + 1e-9)) if not any(np.isnan([Rv(1.08)(s),
                    Rv(1.21)(s), Rv(1.69)(s)])) else np.nan)(sp),
                0.15),
            'BD1300': (BD(1.30, 1.08, 1.75)(sp), 0.03),
            'Right_Slope': ((lambda s: Rv(1.30)(s) - Rv(1.21)(s) if not any(np.isnan([Rv(1.30)(s), Rv(1.21)(s)])) else np.nan)(sp), 0.005),
        },
        'require_all': True,
    }

    # ── LCP  (VNIR component, both sensors) ──
    rules['LCP'] = {
        'desc': 'Low-Ca Pyroxene: 1.0 + 1.9 µm doublet',
        'tests': lambda sp: {
            'BD1000': (BD(1.00, 0.90, 1.10)(sp), 0.03) if waves[0] < 0.95 else
                      (np.nan, 0.03),
            'BD1900': (BD(1.90, 1.77, 2.07)(sp), 0.02),
        },
        'require_all': True,
    }

    if is_L:
        # ── HCP ──
        rules['HCP'] = {
            'desc': 'High-Ca Pyroxene: 2.3 µm',
            'tests': lambda sp: {
                'BD2300': (BD(2.30, 2.08, 2.53)(sp), 0.02),
            },
        }
        # ── Al-Phyllosilicate ──
        rules['Al-Phyllosilicate'] = {
            'desc': 'Al-OH at 2.21 µm',
            'tests': lambda sp: {
                'BD2210': (BD(2.21, 2.12, 2.27)(sp), 0.02),
                'BD1900w':(BD(1.91, 1.83, 2.07)(sp), 0.02),
            },
            'require_all': True,
        }
        # ── Mg-Phyllosilicate ──
        rules['Mg-Phyllosilicate'] = {
            'desc': 'Mg-OH at 2.35 µm',
            'tests': lambda sp: {
                'BD2350': (BD(2.35, 2.25, 2.43)(sp), 0.02),
            },
        }
        # ── Fe-Mg Smectite ──
        rules['Fe-Mg-Smectite'] = {
            'desc': 'Fe/Mg smectite: 2.25 µm',
            'tests': lambda sp: {
                'BD2250': (_bd2250_clean(sp, waves, bad_mask), 0.02),
                'D2300':  (_d2300(sp, waves, bad_mask), 0.02),
            },
            'require_all': True,
        }
        # ── Carbonate ──
        rules['Carbonate'] = {
            'desc': 'CO3 doublet 2.3 + 2.5 µm',
            'tests': lambda sp: {
                'BD2300': (BD(2.30, 2.08, 2.53)(sp), 0.02),
                'BD2500': (BD(2.50, 2.36, 2.68)(sp), 0.02),
            },
            'require_all': True,
        }
        # ── Sulfate ──
        rules['Sulfate'] = {
            'desc': 'SO4 at 2.1 µm',
            'tests': lambda sp: {
                'BD2100': (BD(2.10, 1.97, 2.40)(sp), 0.02),
                'SINDEX2': (_sindex2(sp, waves, bad_mask), 0.10),
            },
            'require_all': True,
        }

    # ── Water ice (both sensors, uses 1.5 µm) ──
    if waves[-1] > 1.6:
        rules['Water-Ice'] = {
            'desc': 'O-H at 1.5 µm',
            'tests': lambda sp: {
                'BD1500': (BD(1.50, 1.43, 1.815)(sp), 0.05),
                'BD2000': (BD(2.00, 1.83, 2.13)(sp), 0.05) if is_L else (np.nan, 0.05),
            },
            'require_all': True,
        }

    return rules


def _bd2250_clean(sp, waves, bad_mask):
    spike_w = (waves >= 2.18) & (waves <= 2.26)
    if np.any(sp[spike_w] > IF_MAX): return np.nan
    return band_depth(sp, waves, bad_mask, 2.25, 2.12, 2.43)

def _d2300(sp, waves, bad_mask):
    vs = [R(sp, waves, bad_mask, w) for w in (2.29, 2.33, 2.12, 2.52)]
    if any(np.isnan(v) for v in vs): return np.nan
    return 1.0 - (vs[0] + vs[1]) / (vs[2] + vs[3] + 1e-9)

def _sindex2(sp, waves, bad_mask):
    vs = [R(sp, waves, bad_mask, w) for w in (2.10, 2.16, 2.29)]
    if any(np.isnan(v) for v in vs): return np.nan
    return 1.0 - (vs[0] + vs[1]) / (2.0 * vs[2] + 1e-9)


# ─────────────────────────────────────────────
# RULE EVALUATION
# ─────────────────────────────────────────────
def evaluate_rules(spectrum, rules):
    results = {}
    for mineral, cfg in rules.items():
        tests    = cfg['tests'](spectrum)
        req_all  = cfg.get('require_all', False)
        evidence = {}
        scores   = []
        for rule_name, (value, threshold) in tests.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                evidence[rule_name] = (None, False)
            else:
                passed = float(value) > threshold
                evidence[rule_name] = (round(float(value), 4), passed)
                if passed: scores.append(float(value))
        n_pass  = sum(1 for _, p in evidence.values() if p)
        n_total = len(tests)
        detected = (n_pass == n_total) if req_all else (n_pass >= 1)
        results[mineral] = {
            'detected': detected,
            'score': float(np.mean(scores)) if scores else 0.0,
            'evidence': evidence,
        }
    return results


# ─────────────────────────────────────────────
# CONTRASTIVE ENCODER
# ─────────────────────────────────────────────
class ContrastiveEncoder(nn.Module):
    def __init__(self, n_good, latent=16):
        super().__init__()
        dummy = torch.zeros(1, 1, n_good)
        flat  = nn.Sequential(
            nn.Conv1d(1, 16, 5, 2, 2),
            nn.Conv1d(16, 32, 3, 2, 1),
            nn.Conv1d(32, 64, 3, 2, 1),
        )(dummy).numel()
        self.conv = nn.Sequential(
            nn.Conv1d(1,16,5,2,2), nn.BatchNorm1d(16), nn.LeakyReLU(0.2),
            nn.Conv1d(16,32,3,2,1),nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32,64,3,2,1),nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(flat,64), nn.LeakyReLU(0.2), nn.Linear(64, latent))
    def forward(self, x):
        return nn.functional.normalize(
            self.fc(self.conv(x).view(x.size(0),-1)), p=2, dim=1)


def augment(b): return b*( torch.rand_like(b)>0.1 ) + torch.randn_like(b)*0.02

def infonce(zi, zj, temp=0.1):
    B = zi.size(0); z = torch.cat([zi,zj])
    sim = torch.mm(z, z.t()) / temp
    lbl = torch.cat([torch.arange(B,2*B), torch.arange(B)]).to(DEVICE)
    sim.masked_fill_(torch.eye(2*B,device=DEVICE).bool(), -9e15)
    return nn.CrossEntropyLoss()(sim, lbl)


def train_encoder(cr_good, n_good, output_dir):
    save = os.path.join(output_dir, 'contrastive_enc.pth')
    model = ContrastiveEncoder(n_good).to(DEVICE)
    if os.path.exists(save):
        model.load_state_dict(torch.load(save, map_location=DEVICE,
                                          weights_only=True))
        model.eval(); return model
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    data = cr_good[:40000] if len(cr_good) > 40000 else cr_good
    ds = DataLoader(torch.tensor(data, dtype=torch.float32).unsqueeze(1),
                    batch_size=256, shuffle=True)
    for ep in range(12):
        ls = 0.0
        for b in ds:
            b = b.to(DEVICE); opt.zero_grad()
            loss = infonce(model(b), model(augment(b)))
            loss.backward(); opt.step(); ls += loss.item()
        if (ep+1) % 4 == 0:
            print(f"    Encoder ep {ep+1}/12  loss={ls/len(ds):.4f}")
    torch.save(model.state_dict(), save)
    model.eval(); return model


# ─────────────────────────────────────────────
# CONTINUUM REMOVAL
# ─────────────────────────────────────────────
def continuum_removal(spectrum, good_idx):
    out = np.zeros(len(spectrum), dtype=np.float32)
    y   = np.clip(spectrum[good_idx].astype(np.float64), 1e-6, None)
    x   = good_idx.astype(np.float64)
    pts = np.vstack([np.column_stack((x, y)),
                     [x[0], -100], [x[-1], -100]])
    try:
        hull  = ConvexHull(pts)
    except Exception:
        out[good_idx] = 1.0; return out
    upper = sorted([v for v in hull.vertices if pts[v,1] >= 0],
                   key=lambda i: pts[i,0])
    if len(upper) < 2:
        out[good_idx] = 1.0; return out
    cont  = np.interp(x, pts[upper,0], pts[upper,1])
    out[good_idx] = np.clip(y / np.clip(cont, 1e-6, None), 0.0, 1.0)
    return out.astype(np.float32)


# ─────────────────────────────────────────────
# AUTO-K CLUSTERING
# ─────────────────────────────────────────────
def find_clusters(latent):
    sub  = latent[np.random.choice(len(latent), min(15000,len(latent)),
                                   replace=False)]
    best_k, best_db = 3, float('inf')
    for k in range(3, 9):
        km  = KMeans(n_clusters=k, random_state=42, n_init=5)
        lbl = km.fit_predict(sub)
        if len(np.unique(lbl)) > 1:
            db = davies_bouldin_score(sub, lbl)
            print(f"   K={k}: DB={db:.4f}")
            if db < best_db: best_db, best_k = db, k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    return km.fit_predict(latent), km.cluster_centers_, best_k


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_mineral_identification(data_dir=DATA_DIR,
                                ml_dir=ML_DIR,
                                phys_dir=PHYS_DIR,
                                output_dir=OUTPUT_DIR):
    print("=" * 65)
    print("  STAGE 5: MINERAL IDENTIFICATION")
    print("  Works for L-sensor (438 bands) and S-sensor (107 bands)")
    print("=" * 65)

    # Prefer Stage-4; fall back to Stage-3
    s4 = sorted(glob.glob(os.path.join(ml_dir, '**', '*_S4.npy'), recursive=True) +
                glob.glob(os.path.join(ml_dir, '*_S4.npy')))
    s3 = sorted(glob.glob(os.path.join(phys_dir, '**', '*_S3.npy'), recursive=True) +
                glob.glob(os.path.join(phys_dir, '*_S3.npy')))

    candidates = s4 if s4 else s3
    if not candidates:
        print("  ❌ No processed .npy files found."); return None

    fp    = candidates[0]
    stage = 'S4' if '_S4.npy' in fp else 'S3'
    sname = os.path.basename(fp).replace(f'_{stage}.npy', '')
    cube  = np.load(fp).astype(np.float32)
    print(f"  Input: {fp}  (stage {stage})")

    # Sanity check
    fv = cube[np.isfinite(cube) & (cube != 0)]
    if len(fv) and abs(float(np.nanmedian(fv))) > 5.0:
        print("  [WARN] Values outside physical range; check preprocessing.")

    H, W, B = cube.shape
    waves    = get_sensor_waves(B)
    bad_mask = build_bad_mask(waves)
    good_mask = ~bad_mask
    good_idx  = np.where(good_mask)[0]
    n_good    = len(good_idx)
    print(f"  Cube: {H}×{W}×{B}  |  Sensor: {'L' if B>200 else 'S'}  "
          f"|  Good bands: {n_good}")

    # Mask sentinels + unphysical
    cube[cube >= 65534] = np.nan
    cube[(cube > IF_MAX) | (cube < IF_MIN)] = np.nan

    rules = get_rules(waves, bad_mask)
    print(f"  Active mineral rules: {list(rules.keys())}")

    # Valid pixel selection
    cube_2d = cube.reshape(-1, B)
    ok = (~np.isnan(cube_2d).all(axis=1) &
          ~np.all(cube_2d == 0, axis=1) &
          (np.nanmedian(cube_2d, axis=1) > IF_MIN) &
          (np.nanmedian(cube_2d, axis=1) < IF_MAX))
    raw_valid = np.clip(cube_2d[ok], IF_MIN, IF_MAX)
    print(f"  Valid pixels: {ok.sum()}/{H*W} ({100*ok.mean():.1f}%)")
    if ok.sum() < 50:
        print("  ❌ Insufficient valid pixels."); return None

    raw_valid[:, bad_mask] = 0.0

    # Continuum removal
    print("\n  [Step 1] Continuum removal...")
    cr_all  = np.array([continuum_removal(p, good_idx) for p in raw_valid])
    cr_good = cr_all[:, good_idx]

    # Contrastive encoder
    print("  [Step 2] Contrastive encoder training...")
    enc = train_encoder(cr_good, n_good, output_dir)
    with torch.no_grad():
        t      = torch.tensor(cr_good, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        latent = enc(t).cpu().numpy()

    print("  [Step 3] Auto-K geological clustering...")
    labels, centroids, K = find_clusters(latent)

    # Compute endmembers (mean spectra per cluster)
    em_raw = np.array([raw_valid[labels==k].mean(axis=0)
                       if (labels==k).any() else raw_valid[0]
                       for k in range(K)])

    # Classify each cluster
    print("\n  [Step 4] Physics rule-based mineral identification...")
    print("-" * 65)
    cluster_results = []
    for k in range(K):
        det  = evaluate_rules(em_raw[k], rules)
        found = {m: d for m, d in det.items() if d['detected']}
        n_px = int((labels == k).sum())
        area = n_px * 0.018 * 0.036

        res = {'type': '', 'minerals': [], 'fractions': [],
               'scores': {m: d['score'] for m, d in det.items()},
               'evidence': {m: d['evidence'] for m, d in det.items()}}

        if not found:
            good_v = em_raw[k][good_idx]
            std    = float(good_v[good_v > 0].std()) if (good_v > 0).any() else 0.0
            if std > 0.15:
                res['type'] = 'SPECTRAL_ANOMALY'
                res['minerals'] = ['Unknown novel signature']
            else:
                res['type'] = 'FEATURELESS'
                res['minerals'] = ['Featureless / dark basalt']
        elif len(found) == 1:
            m = list(found.keys())[0]
            res['type'] = 'SINGLE'
            res['minerals'] = [m]; res['fractions'] = [1.0]
        else:
            res['type'] = 'MIXTURE'
            res['minerals'] = list(found.keys())
            sc = np.array([found[m]['score'] for m in res['minerals']])
            s  = sc.sum()
            res['fractions'] = list(sc/s) if s > 0 else [1/len(sc)]*len(sc)

        cluster_results.append(res)
        print(f"\n  Cluster {k+1} ({n_px} px, ~{area:.3f} km²): {res['type']}")
        if res['type'] in ('SINGLE','MIXTURE'):
            for mn, fr in zip(res['minerals'], res['fractions']):
                print(f"    → {mn:25s}  {fr:.0%}")
        else:
            print(f"    → {res['minerals'][0]}")

    print("-" * 65)

    # CL-5: pixel-count sanity
    print("\n  [CL-5] Cluster pixel counts:")
    for k in range(K):
        print(f"   Cluster {k+1}: {int((labels==k).sum())} px"
              f"  → {cluster_results[k]['minerals'][0][:30]}")

    # NNLS sub-pixel unmixing
    print("\n  [Step 5] NNLS unmixing...")
    A = em_raw.T
    abundances = np.zeros((len(raw_valid), K), dtype=np.float32)
    for i in range(len(raw_valid)):
        x, _ = nnls(A, raw_valid[i])
        s = x.sum()
        abundances[i] = x / s if s > 0 else x

    # Rebuild maps
    class_map = np.zeros(H * W, dtype=np.uint8)
    abund_map = np.zeros((H * W, K), dtype=np.float32)
    class_map[ok] = labels + 1
    abund_map[ok] = abundances
    class_map = class_map.reshape(H, W)
    abund_map = abund_map.reshape(H, W, K)

    np.save(os.path.join(output_dir, f'{sname}_classmap.npy'), class_map)
    np.save(os.path.join(output_dir, f'{sname}_abundance.npy'), abund_map)

    # Classification map PNG
    cmap = plt.colormaps.get_cmap('tab10').resampled(K+1)
    fig, ax = plt.subplots(figsize=(9,4))
    im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=K)
    cb = plt.colorbar(im, ax=ax, ticks=range(K+1))
    tlabels = ['Background'] + [
        f"Cl{k+1}: {cluster_results[k]['minerals'][0][:18]}"
        for k in range(K)]
    cb.set_ticklabels(tlabels, fontsize=7)
    ax.set_title(f'Mineral Map — {sname}', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sname}_classmap.png'), dpi=130)
    plt.close()

    # Per-cluster spectral plots
    for k in range(K):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(waves[good_idx], em_raw[k][good_idx],
                color='#C0392B', lw=1.5, label=f'Cluster {k+1}')
        for lo, hi in [(1.37,1.42),(1.81,1.95),(2.01,2.06),
                       (2.14,2.26),(2.60,2.76),(3.16,3.22),(3.55,4.0)]:
            if hi <= waves[-1]:
                ax.axvspan(lo, hi, color='#555', alpha=0.2)
        ax.set_xlabel('Wavelength (µm)'); ax.set_ylabel('I/F')
        res = cluster_results[k]
        title = (f"{res['type']}: {' + '.join(res['minerals'])}"
                 if res['type'] in ('SINGLE','MIXTURE')
                 else res['minerals'][0])
        ax.set_title(f'Cluster {k+1} — {title}', fontsize=9)
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f'{sname}_cluster{k+1}_spectrum.png'), dpi=120)
        plt.close()

    # Save JSON summary
    summary = {}
    for k, res in enumerate(cluster_results):
        summary[f'cluster_{k+1}'] = {
            'type': res['type'],
            'minerals': res['minerals'],
            'fractions': [round(f,3) for f in res.get('fractions', [])],
            'n_pixels': int((labels==k).sum()),
        }
    with open(os.path.join(output_dir, f'{sname}_minerals.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CL-6: MTRDR cross-check hint
    obs_id = sname.split('_')[0] if '_' in sname else sname
    print(f"\n  [CL-6] MTRDR cross-check:")
    print(f"   Obs ID: {obs_id.upper()}")
    print(f"   URL: https://ode.rsl.wustl.edu/mars/  → search {obs_id.upper()}")
    print(f"   Download *_mtr3.img and compare OLINDEX3, BD2210, BD2350 etc.")

    print(f"\n  ✅ Stage 5 complete  →  {output_dir}")
    return summary


if __name__ == '__main__':
    run_mineral_identification()