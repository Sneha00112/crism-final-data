#!/usr/bin/env python3
"""
run_pipeline.py  —  CRISM Full Pipeline Orchestrator
=====================================================
Runs all five stages in sequence.

Usage
-----
  # Default: uses DATA_DIR set in each stage file
  python run_pipeline.py

  # Override data directory
  CRISM_DATA_DIR="D:/NEW CROSS MISSION/Data/Scene 3/Crism" python run_pipeline.py

  # Run only specific stages (comma-separated)
  python run_pipeline.py --stages 1,2,3

  # Skip ML denoising (run physics only)
  python run_pipeline.py --stages 1,2,3,5

Stage dependency graph
----------------------
  Stage 1 (EDA)           → eda_output/eda_summary.json
  Stage 2 (Noise)         → noise_output/noise_map.json
                            needs: Stage 1 output
  Stage 3 (Physics)       → physics_output/*_S3.npy
                            needs: Stage 1 + 2 outputs
  Stage 4 (ML Denoising)  → ml_output/*_S4.npy
                            needs: Stage 2 + 3 outputs
  Stage 5 (Minerals)      → mineral_output/*
                            needs: Stage 3 or 4 output

Sensor support
--------------
  L-sensor: 438 bands, 1.028–3.920 µm   (file suffix: l_trr3)
  S-sensor: 107 bands, 0.362–1.053 µm   (file suffix: s_trr3)
  Both can appear in the same data directory and are handled
  automatically by band-count detection.
"""

import sys, argparse, time, traceback

def run(stages):
    results = {}

    if 1 in stages:
        print("\n" + "═"*65)
        print(" RUNNING STAGE 1: EDA")
        print("═"*65)
        try:
            from Stage1_eda import run_eda
            t0 = time.time()
            all_results, df = run_eda()
            results[1] = 'OK' if all_results else 'EMPTY'
            print(f"\n  Stage 1 done in {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc(); results[1] = 'ERROR'

    if 2 in stages:
        print("\n" + "═"*65)
        print(" RUNNING STAGE 2: NOISE CHARACTERISATION")
        print("═"*65)
        try:
            from Stage2_noise import run_noise_characterisation
            t0 = time.time()
            nm = run_noise_characterisation()
            results[2] = 'OK' if nm else 'EMPTY'
            print(f"\n  Stage 2 done in {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc(); results[2] = 'ERROR'

    if 3 in stages:
        print("\n" + "═"*65)
        print(" RUNNING STAGE 3: PHYSICS CORRECTION")
        print("═"*65)
        try:
            from Stage3_physics import run_physics_correction
            t0 = time.time()
            logs = run_physics_correction()
            results[3] = 'OK' if logs else 'EMPTY'
            print(f"\n  Stage 3 done in {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc(); results[3] = 'ERROR'

    if 4 in stages:
        print("\n" + "═"*65)
        print(" RUNNING STAGE 4: ML DENOISING (1D CNN + Spectral DAE)")
        print("═"*65)
        try:
            from Stage4_denoising import run_ml_denoising
            t0 = time.time()
            res = run_ml_denoising()
            results[4] = 'OK' if res else 'EMPTY'
            print(f"\n  Stage 4 done in {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc(); results[4] = 'ERROR'

    if 5 in stages:
        print("\n" + "═"*65)
        print(" RUNNING STAGE 5: MINERAL IDENTIFICATION")
        print("═"*65)
        try:
            from Stage5_minerals import run_mineral_identification
            t0 = time.time()
            summary = run_mineral_identification()
            results[5] = 'OK' if summary else 'EMPTY'
            print(f"\n  Stage 5 done in {time.time()-t0:.1f}s")
        except Exception:
            traceback.print_exc(); results[5] = 'ERROR'

    print("\n" + "═"*65)
    print(" PIPELINE SUMMARY")
    print("═"*65)
    labels = {
        1: 'Stage 1 EDA',
        2: 'Stage 2 Noise characterisation',
        3: 'Stage 3 Physics correction',
        4: 'Stage 4 ML denoising (1D)',
        5: 'Stage 5 Mineral identification',
    }
    for s, stat in results.items():
        icon = "✅" if stat == 'OK' else ("⚠️" if stat == 'EMPTY' else "❌")
        print(f"  {icon}  {labels[s]:40s} [{stat}]")

    all_ok = all(v == 'OK' for v in results.values())
    print(f"\n  {'ALL STAGES PASSED' if all_ok else 'SOME STAGES NEED ATTENTION'}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRISM pipeline runner')
    parser.add_argument('--stages', type=str, default='1,2,3,4,5',
                        help='Comma-separated stage numbers, e.g. 1,2,3,4,5')
    args = parser.parse_args()
    stage_list = [int(s.strip()) for s in args.stages.split(',')]
    run(stage_list)