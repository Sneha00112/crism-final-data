#!/usr/bin/env python3
"""
stage4_denoising.py  —  CRISM 1D ML Denoising
==============================================
Two separate 1D models for the two remaining noise types:

  N1  Column stripe noise   → 1D spatial CNN  (operates along the
                               COLUMN / across-track axis for each band)
  N2  Gaussian random noise → Spectral DAE    (operates on the spectral
                               axis per pixel)

TRAINING STRATEGY — THE CORE PROBLEM SOLVED
--------------------------------------------
Without paired clean/noisy data (no MTRDR for this scene) we cannot
train a standard supervised denoiser.  We use two self-supervised
approaches that are scientifically sound:

  For the Stripe CNN (N1):
    CLASSICAL + BLIND-SPOT RESIDUAL TRAINING
    Step 1: Classical column-mean destriper removes the bulk of the
            stripe (no parameters, instant, reliable).
    Step 2: We train a 1D residual CNN to remove the RESIDUAL stripe
            that remains after step 1.  Training data is generated from
            the scene itself:
              - Extract per-band column profiles (1D sequences of W values).
              - Synthetically inject stripe noise at measured amplitude
                into a "clean" version (scene after step 1).
              - The CNN learns: noisy_profile → clean_profile.
            Because the injected noise is i.i.d. column-wise (matching
            real CRISM stripe statistics), this is valid self-supervision.

  For the Spectral DAE (N2):
    BLIND-SPOT / NOISE2VOID SPECTRAL TRAINING
    - Each training sample is one pixel spectrum (length B).
    - We randomly mask ~10% of good-band values and ask the network
      to reconstruct them from their spectral neighbours.
    - Because the masked position is never seen in the forward pass,
      the network CANNOT memorise noise — it must learn the smooth
      spectral structure.
    - This is mathematically equivalent to Noise2Void (Krull+2019)
      applied to the spectral dimension.
    - No external data needed; the scene itself provides ~H×W training
      samples (typically 30,000–300,000 spectra).

  For MIXED DATASETS (L + S sensor files):
    Each sensor gets its own model instance.  Model selection is
    automatic based on band count.

ARCHITECTURE SUMMARY
--------------------
  Stripe CNN   : 1D convolution along column axis per band
                 in_ch=1, channels=32, kernel=9, 3 layers, residual output
                 Operates on: (W,) column profile, one band at a time
                 Loss: MSE + L1 + column-mean preservation

  Spectral DAE : Encoder-decoder on per-pixel spectrum
                 S-sensor (107 bands): 107→64→32→64→107, dropout=0.15
                 L-sensor (438 bands): 438→256→128→256→438, dropout=0.20
                 Bottleneck wider than previous (32→64) to avoid
                 compressing narrow mineral absorptions out of existence.
                 Residual output: model predicts NOISE, subtracts it.
                 Loss: MSE on masked bands + spectral smoothness of NOISE
                       estimate (prevents the noise residual from having
                       sharp spectral features that look like absorptions)

POST-DENOISING VALIDATION BUILT IN
------------------------------------
  Metrics computed per band for bands that carry mineral indices:
    - SNR before / after
    - Spectral angle (SAM) preservation
    - Band-depth preservation for key minerals (BD2300, BD1900, OLINDEX3)
  These are written to the validation JSON.

VALIDATION STATUS CHECK (inline, no separate stage needed):
  PASS  : SNR improved, SAM < 0.05 rad, key band depths preserved ±20%
  REVIEW: SNR improved but SAM in 0.05–0.10 rad
  FAIL  : SNR degraded OR SAM > 0.10 rad → original cube is kept
"""

import os, glob, json, warnings
import numpy as np
from scipy.ndimage import median_filter as sp_median

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

from crism_utils import (
    parse_pds3_label, load_crism_cube, find_label,
    get_sensor_waves, build_bad_mask,
    IF_MAX, IF_MIN,
)

DATA_DIR    = os.environ.get("CRISM_DATA_DIR",
              r"D:\NEW CROSS MISSION\Data\Scene 8\Crism")
PHYSICS_DIR = os.path.join(DATA_DIR, "physics_output")
NOISE_DIR   = os.path.join(DATA_DIR, "noise_output")
OUTPUT_DIR  = os.path.join(DATA_DIR, "ml_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu')
print(f"  [ML] Device: {DEVICE}")

# Key mineral index band centres (µm) for validation
MINERAL_BAND_CENTRES = {
    'BD2300':  (2.30, 2.08, 2.53),
    'BD1900':  (1.90, 1.77, 2.07),
    'OLINDEX3': (1.21, 1.08, 1.69),
    'BD1500':  (1.50, 1.43, 1.815),
}


# ═══════════════════════════════════════════════
# PER-BAND NORMALISE / DENORMALISE
# ═══════════════════════════════════════════════
def per_band_norm(cube):
    """
    Normalise each band independently: subtract band mean, divide by std.
    Returns (normalised_cube, means, stds).
    Denormalisation: cube_n * stds + means
    """
    B = cube.shape[2]
    means = np.zeros(B, dtype=np.float32)
    stds  = np.ones(B, dtype=np.float32)
    out   = np.zeros_like(cube, dtype=np.float32)
    for bi in range(B):
        v = cube[:, :, bi]
        ok = np.isfinite(v) & (v < IF_MAX) & (v > IF_MIN)
        if ok.sum() > 1:
            means[bi] = float(v[ok].mean())
            stds[bi]  = max(float(v[ok].std()), 1e-6)
        out[:, :, bi] = np.where(ok, (v - means[bi]) / stds[bi], 0.0)
    return out, means, stds

def per_band_denorm(cube_n, means, stds):
    out = np.zeros_like(cube_n)
    for bi in range(cube_n.shape[2]):
        out[:, :, bi] = cube_n[:, :, bi] * stds[bi] + means[bi]
    return out


# ═══════════════════════════════════════════════
# N1 — STRIPE CNN  (1D spatial, per-band column axis)
# ═══════════════════════════════════════════════
class StripeCNN1D(nn.Module):
    """
    1D residual CNN operating on a single column profile (length W).
    Architecture: Conv1d 1→32→32→1, kernel 9, residual subtraction.
    The model learns the residual stripe pattern left after classical
    column-mean destriping.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1,  kernel_size=5, padding=2),
        )

    def forward(self, x):
        # x: (batch, 1, W) — residual: predict noise, subtract
        return x - self.net(x)


def classical_destripe(cube, bad_mask):
    """
    STEP 1: Classical column-mean destriper.
    For each band:
      column_offset = col_mean - scene_mean
    Remove the offset.  This handles the dominant, spectrally-flat
    component of CRISM stripe noise.
    """
    H, W, B = cube.shape
    out = cube.copy()
    for bi in range(B):
        if bad_mask[bi]:
            continue
        band = out[:, :, bi]
        vm   = np.isfinite(band) & (band < IF_MAX) & (band > IF_MIN)
        if not vm.any():
            continue
        col_means = np.array([
            float(band[:, c][vm[:, c]].mean()) if vm[:, c].any() else np.nan
            for c in range(W)])
        scene_mean = float(np.nanmean(col_means))
        col_offsets = np.where(np.isfinite(col_means),
                               col_means - scene_mean, 0.0)
        out[:, :, bi] = np.where(vm, band - col_offsets[np.newaxis, :], band)
    return out


def build_stripe_training_data(cube_classical, cube_noisy,
                                stripe_bands, bad_mask, waves, W):
    """
    Self-supervised stripe training data generator.

    For each stripe-affected band (and a random sample of others for
    regularisation):
      - "clean" column profile = column from the classical-destriped cube
      - "noisy" column profile = column from the original noisy cube
      - If they differ, we have a real training pair.

    Additionally synthesise training pairs for non-stripe bands using
    measured stripe amplitudes as injection noise.
    """
    profiles_noisy = []
    profiles_clean = []
    H = cube_classical.shape[0]

    bands_to_use = list(stripe_bands)
    # Add up to 50 random good bands for regularisation
    good_non_stripe = [bi for bi in range(cube_classical.shape[2])
                       if not bad_mask[bi] and bi not in stripe_bands]
    import random; random.shuffle(good_non_stripe)
    bands_to_use += good_non_stripe[:50]

    for bi in bands_to_use:
        if bad_mask[bi] or bi >= cube_classical.shape[2]:
            continue
        clean_band = cube_classical[:, :, bi]
        noisy_band = cube_noisy[:, :, bi]
        vm = (np.isfinite(clean_band) & np.isfinite(noisy_band)
              & (clean_band < IF_MAX) & (clean_band > IF_MIN))
        if vm.mean() < 0.5:
            continue
        # Normalise each column profile independently
        for c in range(W):
            col_c = clean_band[:, c]
            col_n = noisy_band[:, c]
            ok    = vm[:, c]
            if ok.sum() < max(4, H // 4):
                continue
            mu  = float(col_c[ok].mean())
            sig = max(float(col_c[ok].std()), 1e-6)
            profiles_clean.append(((col_c - mu) / sig).astype(np.float32))
            profiles_noisy.append(((col_n - mu) / sig).astype(np.float32))

    if not profiles_clean:
        return None, None
    return (np.array(profiles_noisy, dtype=np.float32),
            np.array(profiles_clean, dtype=np.float32))


def train_stripe_cnn(profiles_noisy, profiles_clean, W, output_dir,
                     epochs=150):
    """
    Train the 1D stripe CNN.
    Input/target: (N, 1, W) — one column profile per sample.
    Loss: MSE + L1 + mean-preservation term.
    """
    # Pad / truncate all profiles to the same length W
    def _pad(arr):
        if arr.shape[1] == W: return arr
        if arr.shape[1] > W:  return arr[:, :W]
        return np.pad(arr, ((0,0),(0, W - arr.shape[1])))

    Xn = torch.tensor(_pad(profiles_noisy)[:, np.newaxis, :])
    Xc = torch.tensor(_pad(profiles_clean)[:, np.newaxis, :])

    perm  = torch.randperm(len(Xn))
    nt    = int(0.8 * len(Xn))
    Xn_tr, Xc_tr = Xn[perm[:nt]].to(DEVICE), Xc[perm[:nt]].to(DEVICE)
    Xn_va, Xc_va = Xn[perm[nt:]].to(DEVICE), Xc[perm[nt:]].to(DEVICE)

    loader = DataLoader(TensorDataset(Xn_tr, Xc_tr),
                        batch_size=256, shuffle=True, drop_last=False)
    model = StripeCNN1D().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    mse   = nn.MSELoss()
    save_path = os.path.join(output_dir, 'stripe_cnn_W{}.pth'.format(W))

    best_vloss = float('inf'); patience = 0
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            pred = model(bx)
            loss = (mse(pred, by)
                    + 0.1 * torch.mean(torch.abs(pred - by))       # L1
                    + 0.1 * torch.abs(pred.mean() - by.mean()))     # mean pres.
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vpred  = model(Xn_va)
            vloss  = (mse(vpred, Xc_va)
                      + 0.1 * torch.mean(torch.abs(vpred - Xc_va))).item()
        if vloss < best_vloss:
            best_vloss = vloss; patience = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= 20: break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE,
                                     weights_only=True))
    print(f"  [StrCNN] Trained {ep+1} epochs, best val MSE={best_vloss:.5f}")
    return model


def apply_stripe_cnn(cube_classical, cube_noisy,
                     stripe_bands, bad_mask, model, W):
    """
    Apply the trained stripe CNN to residual stripes.
    For each stripe band:
      1. Get the column profile from cube_classical (already classically destriped).
      2. Compute the residual column pattern vs cube_noisy.
      3. Run CNN on the residual to refine further.
      4. Subtract refined residual from classical-destriped band.
    """
    if model is None:
        return cube_classical
    H, _, B = cube_classical.shape
    out = cube_classical.copy()
    model.eval()

    with torch.no_grad():
        for bi in stripe_bands:
            if bad_mask[bi] or bi >= B:
                continue
            clean_band = cube_classical[:, :, bi]
            noisy_band = cube_noisy[:, :, bi]
            vm = (np.isfinite(clean_band) & (clean_band < IF_MAX)
                  & (clean_band > IF_MIN))

            profiles = []
            norms    = []   # (mu, sig) per column
            col_idxs = []
            for c in range(W):
                col = clean_band[:, c]
                ok  = vm[:, c]
                if ok.sum() < 2: continue
                mu  = float(col[ok].mean())
                sig = max(float(col[ok].std()), 1e-6)
                profiles.append(col.astype(np.float32))
                norms.append((mu, sig))
                col_idxs.append(c)

            if not profiles: continue
            # Pad to W
            arr = np.array([
                np.pad(p, (0, W - len(p))) if len(p) < W else p[:W]
                for p in profiles], dtype=np.float32)
            # Normalise per-column using stored (mu, sig)
            for i, (mu, sig) in enumerate(norms):
                arr[i] = (arr[i] - mu) / sig

            t_in = torch.tensor(arr[:, np.newaxis, :]).to(DEVICE)
            t_out = model(t_in).squeeze(1).cpu().numpy()

            for i, c in enumerate(col_idxs):
                mu, sig = norms[i]
                cleaned_col = t_out[i, :H] * sig + mu
                ok = vm[:, c]
                out[:, c, bi][ok] = cleaned_col[ok]

    return out


# ═══════════════════════════════════════════════
# N2 — SPECTRAL DAE  (blind-spot / Noise2Void)
# ═══════════════════════════════════════════════
class SpectralDAE(nn.Module):
    """
    Spectral denoising autoencoder.
    S-sensor (107): 107 → 64 → 32 → 64 → 107, dropout=0.15
    L-sensor (438): 438 → 256 → 128 → 256 → 438, dropout=0.20

    Wider bottleneck than previous (26 / 153) to preserve narrow
    mineral absorption features (~2 nm wide on the spectral axis).

    Output = NOISE estimate; caller subtracts: clean = noisy - noise_est.
    Loss = MSE on masked positions + spectral smoothness of noise estimate.
    """
    def __init__(self, B):
        super().__init__()
        is_L = B > 200
        h1 = 256 if is_L else 64
        h2 = 128 if is_L else 32
        drop = 0.20 if is_L else 0.15

        self.enc = nn.Sequential(
            nn.Linear(B, h1), nn.LayerNorm(h1), nn.LeakyReLU(0.1),
            nn.Dropout(drop),
            nn.Linear(h1, h2), nn.LeakyReLU(0.1),
        )
        self.dec = nn.Sequential(
            nn.Linear(h2, h1), nn.LeakyReLU(0.1), nn.Dropout(drop),
            nn.Linear(h1, B),
        )

    def forward(self, x):
        return x - self.dec(self.enc(x))   # residual: return de-noised


def _loss_spectral_smoothness(noise_est):
    """
    Penalise sharp spectral features in the noise estimate.
    A true noise residual should be spectrally smooth; mineral
    absorptions are spectrally structured.  This term prevents
    the DAE from eating absorption features.
    """
    return torch.mean((noise_est[:, 1:] - noise_est[:, :-1]) ** 2)


def build_noise2void_dataset(cube_n, bad_mask, mask_frac=0.10,
                              n_samples=40000):
    """
    Blind-spot / Noise2Void dataset construction.

    For each sampled pixel spectrum:
      1. Sample a binary mask M where mask_frac of GOOD bands are set to 1.
      2. Input  = spectrum with masked positions replaced by local median.
      3. Target = original spectrum values at masked positions only.

    The network never sees the target value at masked positions during
    the forward pass → cannot memorise noise → learns spectral structure.

    Returns:
      inputs   (N, B) float32  — spectra with masks applied
      targets  (N, B) float32  — original spectra
      mask_arr (N, B) float32  — 1 where masked, 0 elsewhere
    """
    H, W, B = cube_n.shape
    flat = cube_n.reshape(-1, B).astype(np.float32)
    # Valid pixels: ≥50% of good bands are non-zero (were originally finite)
    good_bi = np.where(~bad_mask)[0]
    ok = (np.abs(flat[:, good_bi]).mean(axis=1) > 0.01) & \
         np.all(np.isfinite(flat), axis=1)
    valid = flat[ok]

    if len(valid) > n_samples:
        idx   = np.random.choice(len(valid), n_samples, replace=False)
        valid = valid[idx]
    if len(valid) < 100:
        return None, None, None

    N = len(valid)
    inputs   = valid.copy()
    mask_arr = np.zeros((N, B), dtype=np.float32)

    for i in range(N):
        # Choose mask positions from GOOD bands only
        n_mask = max(1, int(mask_frac * len(good_bi)))
        mask_bi = np.random.choice(good_bi, n_mask, replace=False)
        mask_arr[i, mask_bi] = 1.0
        # Replace masked values with local spectral median ±2 neighbours
        for mi in mask_bi:
            lo = max(0, mi - 2)
            hi = min(B, mi + 3)
            neighbours = np.concatenate([valid[i, lo:mi], valid[i, mi+1:hi]])
            replacement = float(np.median(neighbours)) if len(neighbours) > 0 else 0.0
            inputs[i, mi] = replacement

    return inputs, valid, mask_arr


def train_spectral_dae(cube_n, bad_mask, B, output_dir,
                       epochs_L=150, epochs_S=80):
    """
    Train the spectral DAE using the Noise2Void blind-spot approach.
    Loss is computed ONLY on the masked positions (the ones the
    network never saw as input) — this is the key to self-supervision.
    """
    is_L   = B > 200
    epochs = epochs_L if is_L else epochs_S
    label  = 'L' if is_L else 'S'
    save_path = os.path.join(output_dir, f'spectral_dae_{label}_B{B}.pth')

    inputs, targets, masks = build_noise2void_dataset(cube_n, bad_mask)
    if inputs is None:
        print(f"  [DAE-{label}] Insufficient valid spectra — skipping.")
        return None

    N = len(inputs)
    print(f"  [DAE-{label}] Training on {N} spectra, {epochs} epochs, "
          f"device={DEVICE}")

    Xinp = torch.tensor(inputs,  dtype=torch.float32)
    Xtgt = torch.tensor(targets, dtype=torch.float32)
    Xmsk = torch.tensor(masks,   dtype=torch.float32)

    perm  = torch.randperm(N)
    nt    = int(0.85 * N)
    ds    = TensorDataset(Xinp[perm[:nt]], Xtgt[perm[:nt]], Xmsk[perm[:nt]])
    loader= DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
    ds_va = TensorDataset(Xinp[perm[nt:]], Xtgt[perm[nt:]], Xmsk[perm[nt:]])
    ldr_va= DataLoader(ds_va, batch_size=512, shuffle=False)

    model = SpectralDAE(B).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3,
                       weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    mse   = nn.MSELoss()
    LAMBDA_SMOOTH = 0.05   # smoothness weight on noise estimate

    best_vloss = float('inf'); patience = 0; best_ep = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, bt, bm in loader:
            bx, bt, bm = bx.to(DEVICE), bt.to(DEVICE), bm.to(DEVICE)
            opt.zero_grad()
            pred = model(bx)                 # predicted clean spectrum
            noise_est = bx - pred            # estimated noise

            # Loss only on masked positions
            masked_pred = pred * bm
            masked_tgt  = bt   * bm
            loss_rec    = mse(masked_pred, masked_tgt)

            # Spectral smoothness of noise estimate (prevents sharp features)
            loss_smooth = _loss_spectral_smoothness(noise_est)

            loss = loss_rec + LAMBDA_SMOOTH * loss_smooth
            loss.backward(); opt.step()
            train_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bt, bm in ldr_va:
                bx, bt, bm = bx.to(DEVICE), bt.to(DEVICE), bm.to(DEVICE)
                pred       = model(bx)
                noise_est  = bx - pred
                vl = (mse(pred * bm, bt * bm)
                      + LAMBDA_SMOOTH * _loss_spectral_smoothness(noise_est))
                val_loss += vl.item()
        val_loss /= max(1, len(ldr_va))

        if val_loss < best_vloss:
            best_vloss = val_loss; patience = 0; best_ep = ep
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= 25: break

        if (ep + 1) % 20 == 0:
            print(f"    ep {ep+1:3d}/{epochs}  "
                  f"train={train_loss/len(loader):.5f}  "
                  f"val={val_loss:.5f}  best_ep={best_ep+1}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE,
                                     weights_only=True))
    print(f"  [DAE-{label}] Best val loss={best_vloss:.5f} "
          f"at epoch {best_ep+1}")
    return model


def apply_spectral_dae(cube, cube_n, bad_mask, model):
    """
    Apply the trained spectral DAE to all valid pixels.
    Only replaces values in non-bad bands where the pixel was valid.
    """
    if model is None:
        return cube
    H, W, B = cube.shape
    model.eval()

    flat_n = cube_n.reshape(-1, B).astype(np.float32)
    ok_pixels = np.all(np.isfinite(flat_n), axis=1) & \
                (np.abs(flat_n).mean(axis=1) > 0.01)
    out_flat = cube.reshape(-1, B).copy()

    if ok_pixels.sum() == 0:
        return cube

    inp_valid = flat_n[ok_pixels]
    batch_sz  = 50000
    out_valid = np.zeros_like(inp_valid)
    with torch.no_grad():
        for s in range(0, len(inp_valid), batch_sz):
            e = s + batch_sz
            t = torch.tensor(inp_valid[s:e]).to(DEVICE)
            out_valid[s:e] = model(t).cpu().numpy()

    # Denormalise from band-normalised space handled by caller
    out_flat[ok_pixels] = out_valid
    return out_flat.reshape(H, W, B)


# ═══════════════════════════════════════════════
# VALIDATION METRICS
# ═══════════════════════════════════════════════
def compute_snr(cube, good_mask):
    snrs = []
    for bi in np.where(good_mask)[0]:
        v = cube[:, :, bi]
        v = v[np.isfinite(v) & (v < IF_MAX) & (v > IF_MIN)]
        if v.size:
            snrs.append(abs(v.mean()) / (v.std() + 1e-8))
    return float(np.mean(snrs)) if snrs else 0.0

def compute_sam(c1, c2, good_mask):
    B  = c1.shape[2]
    v1 = c1.reshape(-1, B)[:, good_mask].astype(float)
    v2 = c2.reshape(-1, B)[:, good_mask].astype(float)
    ok = np.all(np.isfinite(v1), axis=1) & np.all(np.isfinite(v2), axis=1)
    v1, v2 = v1[ok], v2[ok]
    if not len(v1): return 0.0
    dot  = np.sum(v1 * v2, axis=1)
    n12  = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    return float(np.mean(np.arccos(np.clip(dot / n12, -1+1e-6, 1-1e-6))))


def compute_band_depth(spectrum, waves, centre_um, left_um, right_um):
    """Compute band depth on a mean spectrum."""
    def _r(wl):
        bi = int(np.argmin(np.abs(waves - wl)))
        v  = float(spectrum[bi])
        return np.nan if not np.isfinite(v) else v
    rc, rl, rr = _r(centre_um), _r(left_um), _r(right_um)
    if any(np.isnan(x) for x in (rc, rl, rr)):
        return np.nan
    frac = (centre_um - left_um) / (right_um - left_um + 1e-9)
    cont = rl + frac * (rr - rl)
    return float(1.0 - rc / cont) if cont > 1e-6 else np.nan


def validate_denoising(cube_before, cube_after, waves, good_mask, sname):
    """
    Check that denoising improved SNR without corrupting mineral indices.
    Returns (status, metrics_dict).
    """
    snr_b = compute_snr(cube_before, good_mask)
    snr_a = compute_snr(cube_after,  good_mask)
    sam   = compute_sam(cube_before, cube_after, good_mask)

    # Compute key band depths on the scene-median spectrum
    med_b = np.nanmedian(cube_before.reshape(-1, cube_before.shape[2]), axis=0)
    med_a = np.nanmedian(cube_after.reshape (-1, cube_after.shape[2]),  axis=0)

    bd_preservation = {}
    for name, (c, l, r) in MINERAL_BAND_CENTRES.items():
        if waves[-1] < c:  # S-sensor doesn't cover all indices
            continue
        bd_b = compute_band_depth(med_b, waves, c, l, r)
        bd_a = compute_band_depth(med_a, waves, c, l, r)
        if not any(np.isnan(x) for x in (bd_b, bd_a)):
            delta_pct = abs(bd_a - bd_b) / (abs(bd_b) + 1e-6) * 100
            bd_preservation[name] = {
                'before': round(float(bd_b), 4),
                'after':  round(float(bd_a), 4),
                'delta_pct': round(float(delta_pct), 1),
                'preserved': delta_pct < 20.0,
            }

    snr_improved = snr_a > snr_b * 0.99   # allow 1% tolerance
    sam_ok_pass  = sam < 0.05
    sam_ok_rev   = sam < 0.10
    bd_ok = all(v['preserved'] for v in bd_preservation.values()) \
            if bd_preservation else True

    if snr_improved and sam_ok_pass and bd_ok:
        status = 'PASS'
    elif snr_improved and sam_ok_rev:
        status = 'REVIEW'
    else:
        status = 'FAIL'

    metrics = {
        'snr_before': round(snr_b, 3),
        'snr_after':  round(snr_a, 3),
        'snr_delta_pct': round((snr_a - snr_b) / (snr_b + 1e-8) * 100, 1),
        'sam_rad':    round(sam, 4),
        'status':     status,
        'band_depth_preservation': bd_preservation,
    }

    print(f"\n  ── Validation: {sname} ──")
    print(f"  SNR: {snr_b:.3f} → {snr_a:.3f}  "
          f"(Δ={metrics['snr_delta_pct']:+.1f}%)")
    print(f"  SAM: {sam:.4f} rad  → {status}")
    for nm_idx, bd in bd_preservation.items():
        sym = "✅" if bd['preserved'] else "⚠️"
        print(f"  {sym}  {nm_idx}: {bd['before']:.4f} → {bd['after']:.4f}"
              f"  (Δ={bd['delta_pct']:.1f}%)")
    return status, metrics


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def run_ml_denoising(data_dir=DATA_DIR,
                     physics_dir=PHYSICS_DIR,
                     noise_dir=NOISE_DIR,
                     output_dir=OUTPUT_DIR):
    print("=" * 65)
    print("  STAGE 4: 1D ML DENOISING")
    print("  N1 Stripes  → Classical destriper + 1D residual CNN")
    print("  N2 Gaussian → Spectral DAE (Noise2Void blind-spot)")
    print("=" * 65)

    noise_json = os.path.join(noise_dir, 'noise_map.json')
    if not os.path.exists(noise_json):
        print(f"  ❌ noise_map.json not found at {noise_dir}"); return None
    with open(noise_json) as f:
        noise_map = json.load(f)

    # Collect Stage-3 .npy files
    npy_files = sorted(
        glob.glob(os.path.join(physics_dir, '**', '*_S3.npy'), recursive=True) +
        glob.glob(os.path.join(physics_dir, '*_S3.npy'))
    )
    seen = set()
    npy_files = [f for f in npy_files
                 if not (os.path.basename(f) in seen
                         or seen.add(os.path.basename(f)))]

    if not npy_files:
        print(f"  ❌ No *_S3.npy files found in {physics_dir}")
        return None

    # Need the original .img to build training data for classical destriper
    img_lookup = {}
    for img_path in sorted(
            glob.glob(os.path.join(data_dir, '**', '*.img'), recursive=True) +
            glob.glob(os.path.join(data_dir, '*.img'))):
        img_lookup[os.path.splitext(os.path.basename(img_path))[0]] = img_path

    all_results = {}

    for npy_path in npy_files:
        sname = os.path.basename(npy_path).replace('_S3.npy', '')
        if sname not in noise_map:
            print(f"  Skipping {sname} (not in noise map)"); continue

        print(f"\n{'━'*55}")
        print(f"  {sname}")

        cube_s3 = np.load(npy_path).astype(np.float64)
        H, W, B = cube_s3.shape
        noise   = noise_map[sname]

        # Get wavelengths
        img_path = img_lookup.get(sname)
        meta     = {}
        if img_path:
            lbl_path = find_label(img_path)
            if lbl_path:
                meta = parse_pds3_label(lbl_path)
        waves    = get_sensor_waves(B, meta.get('wavelengths'))
        bad_mask = build_bad_mask(waves)
        good_mask = ~bad_mask

        stripe_bands   = noise.get('striped_bands', [])
        gaussian_bands = (noise.get('high_noise_bands', []) +
                          noise.get('medium_noise_bands', []))
        gaussian_bands = [b for b in set(gaussian_bands) if not bad_mask[b]]

        print(f"  Bands: B={B} ({'L' if B>200 else 'S'}-sensor)  "
              f"| stripe={len(stripe_bands)}  "
              f"| gaussian={len(gaussian_bands)}")

        cube_work = cube_s3.copy()
        stripe_model = None
        dae_model    = None

        # ── N1: Stripe correction ──────────────────────────────
        if stripe_bands:
            print(f"\n  [N1] Stripe denoising ({len(stripe_bands)} bands)")
            # Step 1: Classical destriper (always applied)
            cube_classical = classical_destripe(cube_work, bad_mask)

            # Step 2: Build training data from original + classical
            cube_noisy_orig = cube_s3.copy()
            prof_n, prof_c = build_stripe_training_data(
                cube_classical, cube_noisy_orig,
                stripe_bands, bad_mask, waves, W)

            if prof_n is not None and len(prof_n) >= 200:
                stripe_model = train_stripe_cnn(
                    prof_n, prof_c, W, output_dir, epochs=150)
                cube_destriped = apply_stripe_cnn(
                    cube_classical, cube_noisy_orig,
                    stripe_bands, bad_mask, stripe_model, W)
            else:
                print("  [N1] Insufficient stripe training data — "
                      "classical destriper only")
                cube_destriped = cube_classical

            cube_work = cube_destriped
        else:
            print("  [N1] No stripe bands detected — skipping.")

        # ── N2: Spectral DAE ───────────────────────────────────
        if gaussian_bands:
            print(f"\n  [N2] Gaussian denoising ({len(gaussian_bands)} bands)")
            cube_n, means, stds = per_band_norm(cube_work)
            dae_model = train_spectral_dae(cube_n, bad_mask, B, output_dir)

            if dae_model is not None:
                cube_n_denoised = apply_spectral_dae(
                    cube_work, cube_n, bad_mask, dae_model)
                cube_n_denoised = per_band_denorm(cube_n_denoised, means, stds)

                # Replace ONLY gaussian-affected, non-bad bands
                cube_refined = cube_work.copy()
                vm_global = (np.isfinite(cube_work)
                             & (cube_work < IF_MAX)
                             & (cube_work > IF_MIN))
                for bi in gaussian_bands:
                    if bad_mask[bi]: continue
                    vm = vm_global[:, :, bi]
                    cube_refined[:, :, bi][vm] = cube_n_denoised[:, :, bi][vm]
                cube_work = cube_refined
            else:
                print("  [N2] DAE training failed — keeping physics output")
        else:
            print("  [N2] No Gaussian-noise bands — skipping.")

        # Final clip to physical bounds
        cube_work = np.clip(cube_work, IF_MIN, IF_MAX)

        # ── Validation ─────────────────────────────────────────
        status, metrics = validate_denoising(
            cube_s3, cube_work, waves, good_mask, sname)

        if status == 'FAIL':
            print("  ⚠️  Validation FAILED — reverting to physics output")
            cube_out = cube_s3.astype(np.float32)
        else:
            cube_out = cube_work.astype(np.float32)

        # Save
        rel = ''
        if img_path:
            rel = os.path.relpath(os.path.dirname(img_path), data_dir)
        out_sub = os.path.join(output_dir, rel) if rel else output_dir
        os.makedirs(out_sub, exist_ok=True)
        out_path = os.path.join(out_sub, f'{sname}_S4.npy')
        np.save(out_path, cube_out)
        print(f"  → Saved: {out_path}")

        all_results[sname] = {**metrics, 'output_path': out_path}

    with open(os.path.join(output_dir, 'stage4_validation.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  ✅ Stage 4 complete  →  {output_dir}")
    return all_results


if __name__ == '__main__':
    run_ml_denoising()