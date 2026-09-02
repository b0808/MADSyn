import argparse
import json
import os

import nibabel as nib
from tqdm import tqdm


MODALITY_SUFFIXES = {
    "t1c": "-t1c.nii.gz",
    "t1n": "-t1n.nii.gz",
    "t2w": "-t2w.nii.gz",
    "t2f": "-t2f.nii.gz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute foreground mean/std statistics for BraTS-style volumes.")
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing subject subdirectories.")
    parser.add_argument("--output_file", type=str, default="./data/dataset_stats.json")
    return parser.parse_args()


def get_brats_subjects(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    subjects = []
    for subject_name in sorted(os.listdir(root_dir)):
        subject_dir = os.path.join(root_dir, subject_name)
        if not os.path.isdir(subject_dir):
            continue

        subject = {}
        for modality, suffix in MODALITY_SUFFIXES.items():
            file_path = os.path.join(subject_dir, f"{subject_name}{suffix}")
            if os.path.exists(file_path):
                subject[modality] = file_path

        if len(subject) == len(MODALITY_SUFFIXES):
            subjects.append(subject)

    return subjects


def compute_stats(subjects):
    stats = {}
    for subject in tqdm(subjects, desc="Computing statistics"):
        for path in subject.values():
            image = nib.load(path)
            data = image.get_fdata()
            foreground = data[data > 0]

            if foreground.size == 0:
                mean, std = 0.0, 1.0
            else:
                mean = float(foreground.mean())
                std = float(foreground.std())

            stats[os.path.abspath(path)] = {"mean": mean, "std": std}

    return stats


def main():
    args = parse_args()
    subjects = get_brats_subjects(args.data_root)
    if not subjects:
        raise RuntimeError(f"No complete subjects found in {args.data_root}")

    stats = compute_stats(subjects)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"Saved statistics for {len(subjects)} subjects to {args.output_file}")


if __name__ == "__main__":
    main()
