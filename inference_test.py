import argparse
import json
import os
import nibabel as nib
import numpy as np
import torch

from src.models.DiT_model_random import DiT_class as DiT

# The output order is fixed by the model's decoder/target training
OUTPUT_MODALITIES = ["t1c", "t1n", "t2f", "t2w"]

def parse_args():
    parser = argparse.ArgumentParser(description="Modality-Agnostic MRI Synthesis Inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Lightning checkpoint.")
    parser.add_argument("--inputs", type=str, nargs='+', required=True, help="List of input .nii.gz files (1 to 4 files).")
    parser.add_argument("--save_dir", type=str, default="./outputs/generated", help="Directory to save the 4 outputs.")
    parser.add_argument("--stats_file", type=str, default=None, help="Optional: JSON file for normalization stats.")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--crop", type=int, nargs=4, default=(40, 200, 20, 220), metavar=("H0", "H1", "W0", "W1"))
    
    return parser.parse_args()

def normalize_unit(tensor):
    """Scales tensor to [0, 1] range."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def get_stats(file_path, stats_dict):
    """Attempt to find mean/std in stats file by filename."""
    if not stats_dict:
        return None, None
    filename = os.path.basename(file_path)
    for key in stats_dict:
        if key.endswith(filename):
            return stats_dict[key]["mean"], stats_dict[key]["std"]
    return None, None

def main():
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    
    if len(args.inputs) > 4:
        raise ValueError("You can provide a maximum of 4 input modalities.")
    
    print(f"Received {len(args.inputs)} agnostic input(s).")

    # Load stats if provided
    stats = {}
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file, "r") as fp:
            stats = json.load(fp)

    h0, h1, w0, w1 = args.crop
    reference_affine = None
    reference_shape = None
    mask = None

    # We create a 4-channel tensor. 
    # We will place the provided inputs in the first N channels.
    # The model's internal logic will duplicate and shuffle them.
    volumes = []
    
    for i, file_path in enumerate(args.inputs):
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        
        if reference_affine is None:
            reference_affine = img.affine
            reference_shape = data.shape
            # Create a brain mask based on the first volume
            mask = (data[h0:h1, w0:w1] > 0)

        mean, std = get_stats(file_path, stats)
        crop_volume = data[h0:h1, w0:w1, :]
        
        # On-the-fly normalization if stats aren't found (since inputs are agnostic)
        if mean is None:
            mean = crop_volume[crop_volume > 0].mean()
            std = crop_volume[crop_volume > 0].std()

        # Z-score normalization
        z_volume = np.clip((crop_volume - mean) / (std + 1e-8), -5, 5)
        volumes.append(z_volume)

    # 1. Build the Input Tensor (Depth x 4 x H x W)
    depth = reference_shape[2]
    input_tensor = np.zeros((depth, 4, h1-h0, w1-w0), dtype=np.float32)
    
    # Fill the available channels sequentially (0, 1, ... up to N)
    available_indices = list(range(len(volumes)))
    
    for idx, z_volume in enumerate(volumes):
        # Transpose: W,H,D -> D,H,W
        input_tensor[:, idx, :, :] = z_volume.transpose(2, 0, 1)

    input_tensor = torch.from_numpy(input_tensor).float()

    # 2. Load Model
    print("Loading model...")
    model = DiT.load_from_checkpoint(args.model_path, strict=False).to(device).eval()
    
    # 3. Run Inference
    # We tell the model which channels have real data (e.g., [0, 1] if 2 inputs provided).
    # Your model's code will automatically duplicate from these indices and shuffle them!
    print(f"Running inference (Model will automatically duplicate/shuffle indices {available_indices})...")
    
    # output_buffers will store the 4 fixed output modalities
    output_buffers = [[] for _ in range(4)]
    
    with torch.no_grad():
        for start in range(0, input_tensor.shape[0], args.batch_size):
            batch = input_tensor[start : start + args.batch_size].to(device)
            
            # Predict
            predictions = model.inf_ddim_test(batch, available_indices, args.num_steps)
            
            # Collect the 4 outputs
            for mod_idx in range(4):
                output_buffers[mod_idx].append(predictions[mod_idx].cpu())

    # 4. Reconstruct and Save all 4 Volumes
    os.makedirs(args.save_dir, exist_ok=True)
    mask_tensor = torch.from_numpy(mask)

    print("Saving all 4 generated output modalities...")
    for mod_idx, mod_name in enumerate(OUTPUT_MODALITIES):
        # Concatenate batches, remove channel dim, put Depth back to axis 2
        crop_prediction = torch.cat(output_buffers[mod_idx], dim=0).squeeze(1).permute(1, 2, 0)
        
        # Scale to [0,1] range
        crop_prediction = normalize_unit(crop_prediction)
        
        # Reconstruct into full 3D shape
        full_volume = torch.zeros(reference_shape, dtype=torch.float32)
        full_volume[h0:h1, w0:w1, :] = crop_prediction * mask_tensor
        
        # Save
        save_path = os.path.join(args.save_dir, f"generated_{mod_name}.nii.gz")
        nib.save(nib.Nifti1Image(full_volume.numpy(), reference_affine), save_path)
        print(f"  -> Saved {mod_name.upper()} to: {save_path}")

    print("Inference Complete!")

if __name__ == "__main__":
    main()