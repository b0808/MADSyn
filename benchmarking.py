import argparse
import csv
import glob
import json
import os

import nibabel as nib
import numpy as np
import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from src.models.DiT_model_random import DiT_class as DiT


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
CONDITIONS = (
    ("t1c", (0,)),
    ("t1n", (1,)),
    ("t2f", (2,)),
    ("t2w", (3,)),
    ("t1c+t1n", (0, 1)),
    ("t1c+t2f", (0, 2)),
    ("t1c+t2w", (0, 3)),
    ("t1n+t2f", (1, 2)),
    ("t1n+t2w", (1, 3)),
    ("t2f+t2w", (2, 3)),
    ("t1c+t1n+t2f", (0, 1, 2)),
    ("t1c+t1n+t2w", (0, 1, 3)),
    ("t1c+t2f+t2w", (0, 2, 3)),
    ("t1n+t2f+t2w", (1, 2, 3)),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal MRI synthesis inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a Lightning checkpoint.")
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing subject folders.")
    parser.add_argument("--stats_file", type=str, required=True, help="JSON file produced by src/datamodules/convert.py.")
    parser.add_argument("--save_root", type=str, default="./outputs/generated_volumes")
    parser.add_argument("--output_file", type=str, default="./outputs/metrics.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, cuda:0, or cpu. Defaults automatically.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--crop", type=int, nargs=4, default=(40, 200, 20, 220), metavar=("H0", "H1", "W0", "W1"))
    parser.add_argument("--no_save_volumes", action="store_true", help="Compute metrics without writing NIfTI outputs.")
    return parser.parse_args()


def find_stats_key(path, stats):
    if path in stats:
        return path

    filename = os.path.basename(path)
    for key in stats:
        if key.endswith(filename):
            return key

    raise KeyError(f"Could not find normalization statistics for {filename}")


def normalize_unit(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


def load_subject(subject_dir, stats, crop):
    subject_id = os.path.basename(os.path.normpath(subject_dir))
    paths = {mod: os.path.join(subject_dir, f"{subject_id}-{mod}.nii.gz") for mod in MODALITIES}
    volumes = {}
    affines = {}

    for mod, path in paths.items():
        image = nib.load(path)
        volumes[mod] = image.get_fdata().astype(np.float32)
        affines[mod] = image.affine

    h0, h1, w0, w1 = crop
    z_scores = []
    for mod in MODALITIES:
        key = find_stats_key(paths[mod], stats)
        crop_volume = volumes[mod][h0:h1, w0:w1]
        mean = stats[key]["mean"]
        std = stats[key]["std"]
        z_scores.append(np.clip((crop_volume - mean) / (std + 1e-8), -5, 5))

    input_tensor = torch.from_numpy(np.stack(z_scores, axis=0).transpose(3, 0, 1, 2)).float()
    mask = volumes["t1c"][h0:h1, w0:w1] > 0
    return subject_id, input_tensor, volumes, affines, mask


def compute_metrics(target, prediction, device):
    target = target.to(device)
    prediction = prediction.to(device)

    q_low, q_high = torch.quantile(target, 0.005), torch.quantile(target, 0.995)
    p_low, p_high = torch.quantile(prediction, 0.005), torch.quantile(prediction, 0.995)
    target = torch.clamp(target, min=q_low, max=q_high)
    prediction = torch.clamp(prediction, min=p_low, max=p_high)

    target = normalize_unit(target)
    prediction = normalize_unit(prediction)
    ssim = structural_similarity_index_measure(prediction, target, data_range=1.0)
    psnr = peak_signal_noise_ratio(prediction, target, data_range=1.0)
    return float(ssim), float(psnr)


@torch.no_grad()
def run_subject(subject_dir, model, stats, args, device):
    subject_id, input_tensor, volumes, affines, mask = load_subject(subject_dir, stats, args.crop)
    h0, h1, w0, w1 = args.crop
    rows = []

    for condition_name, condition_indices in CONDITIONS:
        output_buffers = [[] for _ in MODALITIES]
        for start in range(0, input_tensor.shape[0], args.batch_size):
            batch = input_tensor[start:start + args.batch_size].to(device)
            predictions = model.inf_ddim(batch, list(condition_indices), args.num_steps)
            for mod_idx, prediction in enumerate(predictions):
                output_buffers[mod_idx].append(prediction.cpu())

        condition_dir = os.path.join(args.save_root, condition_name.replace("+", "_"), subject_id)
        if not args.no_save_volumes:
            os.makedirs(condition_dir, exist_ok=True)

        row = {"Subject_ID": subject_id, "Condition_Modality": condition_name}
        for mod_idx, mod_name in enumerate(MODALITIES):
            if mod_idx in condition_indices:
                prediction_volume = torch.from_numpy(volumes[mod_name]).float()
                suffix = "org"
            else:
                crop_prediction = torch.cat(output_buffers[mod_idx], dim=0).squeeze(1).permute(1, 2, 0)
                crop_prediction = normalize_unit(crop_prediction)
                prediction_volume = torch.zeros(volumes[mod_name].shape, dtype=torch.float32)
                prediction_volume[h0:h1, w0:w1, :] = crop_prediction * torch.from_numpy(mask)
                suffix = "gen"

            if not args.no_save_volumes:
                save_path = os.path.join(condition_dir, f"{subject_id}-{mod_name}_{suffix}.nii.gz")
                nib.save(nib.Nifti1Image(prediction_volume.numpy(), affines[mod_name]), save_path)

            target = torch.from_numpy(volumes[mod_name]).float().unsqueeze(0).unsqueeze(0)
            prediction = prediction_volume.unsqueeze(0).unsqueeze(0)
            ssim, psnr = compute_metrics(target, prediction, device)
            row[f"SSIM_{mod_name}"] = f"{ssim:.4f}"
            row[f"PSNR_{mod_name}"] = f"{psnr:.4f}"

        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def main():
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    with open(args.stats_file, "r") as fp:
        stats = json.load(fp)

    subject_dirs = sorted(path for path in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(path))
    subject_dirs = subject_dirs[args.start_index:args.end_index]
    if not subject_dirs:
        raise RuntimeError(f"No subject directories found in {args.data_root}")

    print(f"Loading checkpoint from {args.model_path}")
    model = DiT.load_from_checkpoint(args.model_path, strict=False).to(device).eval()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    fieldnames = (
        "Subject_ID",
        "Condition_Modality",
        "SSIM_t1c",
        "SSIM_t1n",
        "SSIM_t2f",
        "SSIM_t2w",
        "PSNR_t1c",
        "PSNR_t1n",
        "PSNR_t2f",
        "PSNR_t2w",
    )

    write_header = not os.path.exists(args.output_file)
    with open(args.output_file, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for index, subject_dir in enumerate(subject_dirs, start=1):
            subject_id = os.path.basename(subject_dir)
            print(f"[{index}/{len(subject_dirs)}] {subject_id}")
            rows = run_subject(subject_dir, model, stats, args, device)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
