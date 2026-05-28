#!/usr/bin/env python3
"""
Compute Mean Squared Error (MSE) between two audio files.

Usage:
    python audio_mse.py file1.wav file2.wav

Requirements:
    pip install librosa numpy
"""

import sys
import os
from glob import glob
import argparse
import numpy as np
import librosa

def compute_mse(audio1, audio2, sr1, sr2):
    """Compute MSE between two audio signals, handling sample rate and length differences."""

    # Resample if sample rates differ
    if sr1 != sr2:
        print(f"Warning: Sample rates differ ({sr1} Hz vs {sr2} Hz). Resampling to {sr1} Hz.")
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    # Ensure mono by averaging channels if stereo
    if len(audio1.shape) > 1:
        audio1 = np.mean(audio1, axis=0)
    if len(audio2.shape) > 1:
        audio2 = np.mean(audio2, axis=0)

    # Trim or pad to match lengths
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    if len(audio1) != len(audio2):
        print("Warning: Audio files were trimmed to match the shortest length.")

    # Compute MSE
    mse = np.mean((audio1 - audio2) ** 2)
    return mse

def calculate_mse(file1, file2):
    audio1, sr1 = librosa.load(file1, sr=None)
    audio2, sr2 = librosa.load(file2, sr=None)
    return compute_mse(audio1, audio2, sr1, sr2)


def extract_id(filename):
    base = os.path.basename(filename)
    if base.endswith('_nat.wav'):
        return base[:-8]
    elif base.endswith('_adv.wav'):
        return base[:-8]
    return None


def compare_directories(nat_dir, adv_dir):
    nat_files = glob(os.path.join(nat_dir, '*_nat.wav'))
    adv_files = glob(os.path.join(adv_dir, '*_adv.wav'))

    nat_map = {extract_id(f): f for f in nat_files}
    adv_map = {extract_id(f): f for f in adv_files}

    common_ids = set(nat_map.keys()) & set(adv_map.keys())
    if not common_ids:
        raise ValueError("No matching ID pairs found between directories")

    mses = []
    for id_ in sorted(common_ids):
        nat_path = nat_map[id_]
        adv_path = adv_map[id_]
        try:
            mse = calculate_mse(nat_path, adv_path)
            mses.append(mse)
            print(f"{id_}: {mse:.10f}")
        except Exception as e:
            print(f"Error computing MSE for {id_}: {e}")

    mean_mse = np.mean(mses) if mses else float('nan')
    print(f"\nMean MSE: {mean_mse:.10f}")
    if mean_mse > 0:
        print(f"Mean MSE (dB): {10 * np.log10(mean_mse):.2f}")
    else:
        print("Mean MSE (dB): -inf")
    return mean_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MSE between audio files or directories")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--files', nargs=2, metavar=('FILE1', 'FILE2'), help='Two audio files to compare')
    group.add_argument('--dirs', nargs=2, metavar=('NAT_DIR', 'ADV_DIR'), help='Directories with *_nat.wav and *_adv.wav files')

    args = parser.parse_args()

    try:
        if args.files:
            mse = calculate_mse(args.files[0], args.files[1])
            print(f"\nMSE between '{args.files[0]}' and '{args.files[1]}': {mse:.10f}")
            print(f"MSE (dB): {10 * np.log10(mse):.2f}" if mse > 0 else "MSE (dB): -inf")
        else:
            compare_directories(args.dirs[0], args.dirs[1])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)