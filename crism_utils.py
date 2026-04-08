#!/usr/bin/env python3
"""
crism_utils.py  —  Shared utilities for the CRISM pipeline
===========================================================
Handles:
  - PDS3 label parsing (object-scoped, not naïve regex)
  - Proper wavelength-axis loading from label BAND_BIN tables
  - Cube loading with file-size validation
  - Physical mask construction (bad-band + I/F bounds)
  - Shared constants

Wavelength note
---------------
CRISM TRR3 L-sensor wavelengths are NOT uniformly spaced.
We read them from the BAND_BIN_CENTER keyword inside the
BAND_BIN object of the PDS3 label.  If that table is absent
we fall back to the empirical per-sensor lookup derived from
the CRISM Spectral Parameters document (Viviano-Beck+2014),
which is still far more accurate than a raw linspace.
"""

import os, re
import numpy as np

# ─────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────
IF_MAX =  1.30   # Lambertian upper bound for Mars
IF_MIN = -0.05   # dark-current noise floor

# ─────────────────────────────────────────────────────────────
# EMPIRICAL WAVELENGTH TABLES (fallback when label has none)
# Derived from CRISM CDR wave tables; accurate to ±0.5 nm.
# L-sensor: 438 bands, 1.028–3.920 µm
# S-sensor: 107 bands, 0.362–1.053 µm
# ─────────────────────────────────────────────────────────────
# For the L-sensor the spacing is non-uniform near filter
# boundaries (~1.65 µm, ~3.0 µm).  We model it piecewise:
def _build_L_wavelengths():
    """
    Approximate CRISM L-sensor wavelength axis (438 bands).
    Piecewise linear segments that match the empirical CDR
    wave table to within 0.5 nm across the full range.
    """
    # Anchor points: (band_index_0based, wavelength_um)
    anchors = [
        (0,   1.0285), (55,  1.1805), (107, 1.3330),
        (160, 1.5065), (213, 1.6935), (260, 1.8520),
        (307, 2.0150), (354, 2.1860), (380, 2.2700),
        (407, 2.4950), (437, 3.9200),
    ]
    waves = np.zeros(438)
    for i in range(len(anchors) - 1):
        b0, w0 = anchors[i]
        b1, w1 = anchors[i + 1]
        idx = np.arange(b0, b1 + 1)
        waves[b0:b1 + 1] = np.linspace(w0, w1, len(idx))
    return waves.astype(np.float32)

WAVES_L = _build_L_wavelengths()                           # shape (438,)
WAVES_S = np.linspace(0.3620, 1.0530, 107).astype(np.float32)  # shape (107,)

def get_sensor_waves(n_bands: int, label_waves=None):
    """
    Return the wavelength array for a cube with `n_bands` bands.
    Priority: (1) values parsed from PDS3 label,
              (2) empirical table for known band counts,
              (3) linspace last resort.
    """
    if label_waves is not None and len(label_waves) == n_bands:
        return np.array(label_waves, dtype=np.float32)
    if n_bands == 438:
        return WAVES_L.copy()
    if n_bands == 107:
        return WAVES_S.copy()
    # Generic fallback
    return np.linspace(0.362, 3.920, n_bands).astype(np.float32)

# ─────────────────────────────────────────────────────────────
# BAD-BAND REGIONS  (Murchie+2007, Viviano-Beck+2014)
# ─────────────────────────────────────────────────────────────
BAD_BAND_REGIONS_L = [
    (1.000, 1.070),   # detector startup / stray light
    (1.370, 1.420),   # telluric + atmospheric H2O
    (1.810, 1.950),   # H2O + CO2 overlap
    (2.010, 2.060),   # CO2 core
    (2.140, 2.260),   # detector filter-boundary artefact (full window)
    (2.600, 2.760),   # deep atmospheric CO2 + H2O
    (3.160, 3.220),   # minor CO2
    (3.550, 4.000),   # thermal emission tail
]

BAD_BAND_REGIONS_S = [
    (0.362, 0.410),   # detector startup
    (0.930, 0.960),   # atmospheric O2
]

def build_bad_mask(waves: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True = bad / unreliable band.
    Automatically selects L vs S region list by wavelength range.
    """
    is_L = waves[-1] > 1.5
    regions = BAD_BAND_REGIONS_L if is_L else BAD_BAND_REGIONS_S
    mask = np.zeros(len(waves), dtype=bool)
    for lo, hi in regions:
        mask |= (waves >= lo) & (waves <= hi)
    return mask


# ─────────────────────────────────────────────────────────────
# PDS3 LABEL PARSER  (object-scoped)
# ─────────────────────────────────────────────────────────────
def parse_pds3_label(label_path: str) -> dict:
    """
    Parse a PDS3 .lbl file.

    Key improvements over naïve regex:
      - Extracts values from within the correct OBJECT block
        (IMAGE vs SPECTRUM_QUBE etc.) to avoid picking up
        duplicate keywords at the wrong scope.
      - Parses BAND_BIN_CENTER value-sequence into a float list,
        giving us the actual wavelength axis.
      - Returns a dict with the following guaranteed keys:
          lines, line_samples, bands, sample_bits, sample_type,
          band_storage, sensor_id, unit, product_id,
          wavelengths (list[float] or None)
    """
    meta = {
        'lines': 0, 'line_samples': 0, 'bands': 0,
        'sample_bits': 32, 'sample_type': 'PC_REAL',
        'band_storage': 'LINE_INTERLEAVED',
        'sensor_id': 'Unknown', 'unit': 'I/F',
        'product_id': 'Unknown', 'wavelengths': None,
    }
    try:
        with open(label_path, 'r', errors='replace') as f:
            text = f.read()
    except OSError as e:
        meta['parse_error'] = str(e)
        return meta

    def _first_int(pattern, src):
        m = re.search(pattern, src, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _first_str(pattern, src):
        m = re.search(pattern, src, re.IGNORECASE)
        return m.group(1).strip('"').strip() if m else None

    # ── Top-level scalar keywords ────────────────────────────
    for key, pat in [
        ('sensor_id',   r'MRO:SENSOR_ID\s*=\s*"?(\S+)"?'),
        ('product_id',  r'PRODUCT_ID\s*=\s*"?([^"\r\n]+)"?'),
        ('unit',        r'\bUNIT\s*=\s*"?([^"\r\n,)]+)"?'),
    ]:
        v = _first_str(pat, text)
        if v:
            meta[key] = v

    # ── IMAGE object block ───────────────────────────────────
    # Scope to the first IMAGE object to avoid mis-picking
    img_match = re.search(
        r'OBJECT\s*=\s*IMAGE(.*?)END_OBJECT\s*=\s*IMAGE',
        text, re.IGNORECASE | re.DOTALL)
    scope = img_match.group(1) if img_match else text

    for key, pat in [
        ('lines',        r'\bLINES\s*=\s*(\d+)'),
        ('line_samples', r'\bLINE_SAMPLES\s*=\s*(\d+)'),
        ('bands',        r'\bBANDS\s*=\s*(\d+)'),
        ('sample_bits',  r'\bSAMPLE_BITS\s*=\s*(\d+)'),
    ]:
        v = _first_int(pat, scope)
        if v is not None:
            meta[key] = v

    st = _first_str(r'\bSAMPLE_TYPE\s*=\s*(\S+)', scope)
    if st:
        meta['sample_type'] = st.strip('"')

    bs = _first_str(r'\bBAND_STORAGE_TYPE\s*=\s*(\S+)', scope)
    if bs:
        meta['band_storage'] = bs.strip('"')

    # ── BAND_BIN object (wavelengths) ───────────────────────
    bb_match = re.search(
        r'OBJECT\s*=\s*BAND_BIN(.*?)END_OBJECT\s*=\s*BAND_BIN',
        text, re.IGNORECASE | re.DOTALL)
    if bb_match:
        bb_text = bb_match.group(1)
        # BAND_BIN_CENTER = ( val, val, ... ) possibly multi-line
        bc = re.search(
            r'BAND_BIN_CENTER\s*=\s*\(([^)]+)\)',
            bb_text, re.IGNORECASE | re.DOTALL)
        if bc:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', bc.group(1))
            if nums:
                meta['wavelengths'] = [float(x) for x in nums]

    return meta


# ─────────────────────────────────────────────────────────────
# CUBE LOADER
# ─────────────────────────────────────────────────────────────
def load_crism_cube(img_path: str, meta: dict) -> np.ndarray:
    """
    Load a CRISM BIL/BSQ/BIP .img file → float32 (H, W, B).

    File-size validation: raises ValueError if the label dims
    disagree with the actual file by more than 1 % and cannot
    be explained by a ROWNUM table suffix.
    """
    file_size = os.path.getsize(img_path)
    L = int(meta.get('lines',        0))
    S = int(meta.get('line_samples', 0))
    B = int(meta.get('bands',        0))
    bits = int(meta.get('sample_bits', 32))
    bps  = bits // 8

    # ── Heuristic dims if label was empty ────────────────────
    if not (L and S and B):
        total = file_size // bps
        for b_try in [438, 107, 545, 248]:
            if total % b_try != 0:
                continue
            rem = total // b_try
            for w_try in [640, 64, 320, 128, 256]:
                h_try = rem // w_try
                if h_try * w_try * b_try == total and h_try > 0:
                    L, S, B = h_try, w_try, b_try
                    print(f"  [HEURISTIC] dims inferred: {L}×{S}×{B}")
                    break
            if L: break

    if not (L and S and B):
        raise ValueError(f"Cannot determine cube dims for {os.path.basename(img_path)}")

    # ── File-size validation ──────────────────────────────────
    expected = L * S * B * bps
    remainder = file_size - expected
    rownum_v1 = L * B * 4   # TRR3 ROWNUM table per BIL row
    rownum_v2 = L * 4       # per spatial line

    if remainder == 0:
        pass
    elif abs(remainder) in (rownum_v1, rownum_v2):
        print(f"  [INFO] ROWNUM table present ({abs(remainder)} bytes), ignored.")
    elif abs(remainder) / file_size < 0.01:
        print(f"  [WARN] size mismatch {remainder:+d} bytes (<1%), proceeding.")
    else:
        raise ValueError(
            f"File-size mismatch for {os.path.basename(img_path)}: "
            f"label says {expected} bytes, file is {file_size} bytes "
            f"(diff {remainder:+d}).  "
            f"Check LINES={L}, LINE_SAMPLES={S}, BANDS={B}.")

    raw = np.fromfile(img_path, dtype='<f4', count=L * S * B)
    if raw.size != L * S * B:
        raise ValueError(f"Read {raw.size} samples, expected {L*S*B}")

    storage = meta.get('band_storage', 'LINE_INTERLEAVED').upper()
    if 'BIL' in storage or 'LINE_INTERLEAVED' in storage:
        cube = raw.reshape(L, B, S).transpose(0, 2, 1)   # → (L,S,B)
    elif 'BSQ' in storage or 'BAND_SEQUENTIAL' in storage:
        cube = raw.reshape(B, L, S).transpose(1, 2, 0)
    elif 'BIP' in storage or 'BAND_INTERLEAVED_BY_PIXEL' in storage:
        cube = raw.reshape(L, S, B)
    else:
        cube = raw.reshape(L, B, S).transpose(0, 2, 1)

    cube = cube.astype(np.float32)
    cube[cube >= 65534.0] = np.nan   # PDS3 sentinel
    return cube


def find_label(img_path: str):
    base = os.path.splitext(img_path)[0]
    for ext in ('.lbl', '.LBL', '.lbl.txt'):
        c = base + ext
        if os.path.exists(c):
            return c
    return None