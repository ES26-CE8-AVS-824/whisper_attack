"""
Data preparation for the VCTK corpus (CSTR-Edinburgh/vctk) from
HuggingFace.

Produces a CSV index compatible with robust_speech.data.dataio.dataio_prepare.

CSV columns
-----------
ID        : unique utterance identifier  (e.g. p225_001)
duration  : duration in seconds          (float as string)
wav       : absolute path to 16 kHz WAV  (written under <data_folder>/audio/)
spk_id    : speaker identifier           (e.g. p225)
wrd       : normalised transcript

Usage
-----
Called automatically by run_attack.py via the `dataset_prepare_fct` key in
attack_configs/whisper/pgd_vctk.yaml.  Can also be run standalone:

    python vctk_prepare.py \\
        --data_folder /path/to/VCTK \\
        --save_folder /path/to/VCTK/csv \\
        --split vctk-100
"""

import argparse
import csv
import logging
import os

import torchaudio

logger = logging.getLogger(__name__)

SAMPLERATE = 16000
NUM_SAMPLES = 100
# Official VCTK dataset on HuggingFace (CSTR-Edinburgh).
# Recordings are 48 kHz; this script resamples them to 16 kHz on save.
HF_DATASET_ID = "CSTR-Edinburgh/vctk"


# ---------------------------------------------------------------------------
# Public entry point (called by run_attack.py / fit_attacker.py)
# ---------------------------------------------------------------------------

def prepare_vctk(
    data_folder,
    te_splits,
    save_folder,
    skip_prep=False,
    num_samples=NUM_SAMPLES,
    hf_dataset=HF_DATASET_ID,
    sample_seed=None,
):
    """Prepare VCTK data for adversarial attack evaluation.

    Downloads up to `num_samples` utterances from the VCTK corpus on
    HuggingFace, saves them as 16 kHz mono WAV files under
    ``<data_folder>/audio/``, and writes a CSV index for each split name
    listed in `te_splits`.

    Arguments
    ---------
    data_folder : str
        Root directory where audio files will be stored.
    te_splits : list[str]
        Test split names (e.g. ``["vctk-100"]``).  A file
        ``<split>.csv`` is written to `save_folder` for every entry.
    save_folder : str
        Directory where CSV files are written.
    skip_prep : bool
        When *True*, skip preparation if all expected CSV files already exist.
    num_samples : int
        Maximum number of utterances to include (default 100).
    hf_dataset : str
        HuggingFace dataset identifier for VCTK
        (default ``"CSTR-Edinburgh/vctk"``).
    sample_seed : int | None
        Optional seed used to randomize the sampled subset. When omitted,
        a different subset may be selected on each preparation run.
    """
    os.makedirs(save_folder, exist_ok=True)

    if skip_prep and _all_csvs_exist(te_splits, save_folder):
        logger.info("Skipping VCTK preparation — CSV files already present.")
        return

    audio_dir = os.path.join(data_folder, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    existing_rows = _load_existing_rows(te_splits, save_folder, num_samples)
    if existing_rows is not None:
        logger.info(
            "Reusing %d existing VCTK samples from local CSV/audio cache.",
            len(existing_rows),
        )
        for split in te_splits:
            csv_path = os.path.join(save_folder, split + ".csv")
            if not os.path.isfile(csv_path):
                _write_csv(csv_path, existing_rows)
                logger.info(
                    "VCTK CSV written to %s (%d rows).",
                    csv_path,
                    len(existing_rows),
                )
        return

    rows = _download_vctk_samples(
        audio_dir,
        num_samples,
        hf_dataset,
        sample_seed=sample_seed,
    )

    for split in te_splits:
        csv_path = os.path.join(save_folder, split + ".csv")
        _write_csv(csv_path, rows)
        logger.info(
            "VCTK CSV written to %s (%d rows).", csv_path, len(rows)
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _all_csvs_exist(splits, save_folder):
    """Return True when every expected CSV file is already on disk."""
    return all(
        os.path.isfile(os.path.join(save_folder, s + ".csv"))
        for s in splits
    )


def _load_existing_rows(splits, save_folder, num_samples):
    """Return cached rows when a prepared CSV and all referenced WAVs exist."""
    for split in splits:
        csv_path = os.path.join(save_folder, split + ".csv")
        rows = _read_csv_rows(csv_path)
        if rows is None:
            continue

        rows = rows[:num_samples]
        if len(rows) < num_samples:
            logger.info(
                "Existing VCTK CSV %s has only %d rows; downloading more data.",
                csv_path,
                len(rows),
            )
            continue

        missing_wavs = [row["wav"] for row in rows if not os.path.isfile(row["wav"])]
        if missing_wavs:
            logger.info(
                "Existing VCTK CSV %s references %d missing WAV files; downloading data again.",
                csv_path,
                len(missing_wavs),
            )
            continue

        return rows

    return None


def _read_csv_rows(csv_path):
    """Read prepared CSV rows if the file exists and has the expected columns."""
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_fields = ["ID", "duration", "wav", "spk_id", "wrd"]
        if reader.fieldnames != expected_fields:
            logger.info(
                "Ignoring cached VCTK CSV %s because its columns do not match the expected format.",
                csv_path,
            )
            return None
        return list(reader)


def _download_vctk_samples(audio_dir, num_samples, hf_dataset, sample_seed=None):
    """Stream samples from HuggingFace, save WAV files, and return CSV rows.

    The dataset is shuffled before the first `num_samples` utterances are
    selected so the prepared subset is randomized.

    Returns
    -------
    list[dict]
        One dict per accepted utterance with keys:
        ``ID``, ``duration``, ``wav``, ``spk_id``, ``wrd``.
    """
    import numpy as np
    import torch
    from datasets import load_dataset

    if sample_seed is None:
        logger.info(
            "Downloading and randomly sampling %d samples from %s.",
            num_samples,
            hf_dataset,
        )
    else:
        logger.info(
            "Downloading and randomly sampling %d samples from %s with seed %s.",
            num_samples,
            hf_dataset,
            sample_seed,
        )

    # Randomized subset selection requires materializing the dataset locally.
    # trust_remote_code is required by some HuggingFace datasets.
    ds = load_dataset(
        hf_dataset,
        split="train",
        streaming=False,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=sample_seed)

    rows = []
    saved = 0
    for sample in ds:
        if saved >= num_samples:
            break

        audio = sample["audio"]
        array = audio["array"]
        src_sr = audio["sampling_rate"]

        # --- utterance ID --------------------------------------------------
        utt_id = (
            sample.get("file_id")
            or sample.get("id")
            or f"vctk_{saved:04d}"
        )
        # Sanitise for use as a filename
        utt_id = str(utt_id).replace("/", "_").replace(" ", "_")

        # --- speaker ID ----------------------------------------------------
        spk_id = (
            sample.get("speaker_id")
            or sample.get("spk_id")
            or "unknown"
        )

        # --- transcript ----------------------------------------------------
        wrd = (
            sample.get("text")
            or sample.get("transcription")
            or ""
        ).strip()

        if not wrd:
            logger.warning(
                "Empty transcript for sample %d (%s); skipping.", saved, utt_id
            )
            continue

        # --- save 16 kHz WAV ----------------------------------------------
        wav_path = os.path.join(audio_dir, utt_id + ".wav")
        if not os.path.exists(wav_path):
            tensor = torch.from_numpy(
                np.array(array, dtype="float32")
            ).unsqueeze(0)  # shape: (1, T)
            if src_sr != SAMPLERATE:
                tensor = torchaudio.transforms.Resample(
                    src_sr, SAMPLERATE
                )(tensor)
            torchaudio.save(wav_path, tensor, SAMPLERATE)

        duration = _get_duration(wav_path)
        rows.append(
            {
                "ID": utt_id,
                "duration": f"{duration:.4f}",
                "wav": wav_path,
                "spk_id": str(spk_id),
                "wrd": wrd,
            }
        )
        saved += 1

    logger.info("Collected %d VCTK samples.", len(rows))
    return rows


def _get_duration(wav_path):
    """Return duration in seconds for a WAV file."""
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate


def _write_csv(csv_path, rows):
    """Write *rows* to a CSV file compatible with robust_speech dataio."""
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "duration", "wav", "spk_id", "wrd"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Prepare a 100-sample VCTK CSV for whisper_attack."
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        help="Root folder where audio/ sub-directory will be created.",
    )
    parser.add_argument(
        "--save_folder",
        required=True,
        help="Directory where the CSV file(s) are written.",
    )
    parser.add_argument(
        "--split",
        default="vctk-100",
        help="Name of the split (determines CSV filename). Default: vctk-100.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of utterances to download. Default: 100.",
    )
    parser.add_argument(
        "--hf_dataset",
        default=HF_DATASET_ID,
        help="HuggingFace dataset ID. Default: CSTR-Edinburgh/vctk.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Optional seed for randomized subset selection.",
    )
    parser.add_argument(
        "--skip_prep",
        action="store_true",
        help="Skip preparation if CSV already exists.",
    )
    args = parser.parse_args()

    prepare_vctk(
        data_folder=args.data_folder,
        te_splits=[args.split],
        save_folder=args.save_folder,
        skip_prep=args.skip_prep,
        num_samples=args.num_samples,
        hf_dataset=args.hf_dataset,
        sample_seed=args.sample_seed,
    )
