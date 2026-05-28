import argparse
import os
import numpy as np
from scipy.io import wavfile
from glob import glob

def calculate_snr(original_path, adversarial_path):
    rate1, original = wavfile.read(original_path)
    rate2, adversarial = wavfile.read(adversarial_path)

    if rate1 != rate2:
        raise ValueError(f"Sample rates must match: {original_path} vs {adversarial_path}")
    if original.shape != adversarial.shape:
        raise ValueError(f"Audio shapes must match: {original_path} vs {adversarial_path}")

    original = original.astype(np.float64)
    adversarial = adversarial.astype(np.float64)

    noise = adversarial - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)

def extract_id(filename):
    """Extract ID from filename: <ID>_nat.wav or <ID>_adv.wav"""
    base = os.path.basename(filename)
    if base.endswith('_nat.wav'):
        return base[:-8]  # Remove '_nat.wav'
    elif base.endswith('_adv.wav'):
        return base[:-8]  # Remove '_adv.wav'
    return None

def compare_directories(nat_dir, adv_dir):
    nat_files = glob(os.path.join(nat_dir, '*_nat.wav'))
    adv_files = glob(os.path.join(adv_dir, '*_adv.wav'))

    # Build ID to path mappings
    nat_map = {extract_id(f): f for f in nat_files}
    adv_map = {extract_id(f): f for f in adv_files}

    common_ids = set(nat_map.keys()) & set(adv_map.keys())
    if not common_ids:
        raise ValueError("No matching ID pairs found between directories")

    snrs = []
    for id_ in sorted(common_ids):
        nat_path = nat_map[id_]
        adv_path = adv_map[id_]
        snr = calculate_snr(nat_path, adv_path)
        snrs.append(snr)
        print(f"{id_}: {snr:.2f} dB")

    mean_snr = np.mean(snrs)
    print(f"\nMean SNR: {mean_snr:.2f} dB")
    return mean_snr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SNR between audio files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs=2, metavar=("ORIGINAL", "ADVERSARIAL"),
                      help="Two WAV files to compare")
    group.add_argument("--dirs", nargs=2, metavar=("NAT_DIR", "ADV_DIR"),
                      help="Directories with *_nat.wav and *_adv.wav files")

    args = parser.parse_args()

    if args.files:
        snr = calculate_snr(args.files[0], args.files[1])
        print(f"SNR: {snr:.2f} dB")
    else:
        compare_directories(args.dirs[0], args.dirs[1])