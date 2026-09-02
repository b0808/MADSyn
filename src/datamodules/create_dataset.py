import os
import glob
import torch
import torchio as tio
import SimpleITK as sitk

from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import json

def find_brats_subjects(root_folder,m):
    """
    Scans folders to find BraTS subjects.
    Assumes structure: root_folder / SubjectID / SubjectID-modality.nii.gz
    """
    subjects_list = []
    
    # 1. Get all subfolders (each subfolder is a patient)
    # e.g., ValidationData/BraTS-GLI-00001-000
    patient_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    for folder in patient_folders:
        patient_id = os.path.basename(folder)
        
        # 2. Define the expected filenames for the 4 modalities
        # Adjust these suffixes if your files are named differently!
        # Example: BraTS-GLI-00001-000-t1c.nii.gz
        paths = {
            't1n':  os.path.join(folder, f"{patient_id}-t1n.nii.gz"), # T1 Native
            't1c': os.path.join(folder, f"{patient_id}-t1c.nii.gz"), # T1 Contrast
            't2w':  os.path.join(folder, f"{patient_id}-t2w.nii.gz"), # T2 Weighted
            't2f': os.path.join(folder, f"{patient_id}-t2f.nii.gz"), # FLAIR
            'seg': os.path.join(folder, f"{patient_id}-seg.nii.gz") # seg
        }
        
        # 3. Check if all 4 files actually exist before adding to the list
        if all(os.path.exists(p) for p in paths.values()):
            subjects_list.append(paths)
        else:
            print(f"Skipping {paths}: Missing one or more files.")
            
    return subjects_list

class BraTS_2D_Dataset(Dataset):
    def __init__(self, subjects_list, cfg, k, slice_range=(40, 140)):
        self.subjects = subjects_list
        self.cfg = cfg
        self.slice_min, self.slice_max = slice_range
        self.modality_order = ['t1c', 't1n', 't2f', 't2w', "seg"]
        self.k = k
        stats_file_train = self.cfg.get("stats_file_train", os.path.join(self.cfg.path.pathBase, "dataset_stats_train.json"))
        with open(stats_file_train, 'r') as f:
                self.stats_train = json.load(f)
        stats_file_val = self.cfg.get("stats_file_val", os.path.join(self.cfg.path.pathBase, "dataset_stats_val.json"))
        with open(stats_file_val, 'r') as f:
                self.stats_val = json.load(f)
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index):
        paths = self.subjects[index]
        
        # Pick random slice index
        if self.k == "train":
            slice_idx = torch.randint(self.slice_min, self.slice_max, (1,)).item()
            stats = self.stats_train
        else:
            slice_idx = (self.slice_min + self.slice_max) // 2
            stats = self.stats_val
        
        slices = []
        
        for mod in self.modality_order:
            path = paths[mod]  # This is the string path
            
            img_obj = nib.load(path)
            
            img_slice = np.array(img_obj.dataobj[40:200, 20:220, slice_idx], dtype=np.float32)

            # Normalization is computed per slice for memory-efficient loading.
            if mod != "seg":
                mask = img_slice > 0
                if mask.any():
                    mean = stats[path]['mean']
                    std = stats[path]['std']
                    img_slice = (img_slice - mean) / (std + 1e-8)
                    img_slice = np.clip(img_slice, -5, 5)
                # If mask is empty, img_slice remains as is (likely zeros)
            
            slices.append(img_slice)
        
        # Stack to (Channels, H, W)
        stacked_image = np.stack(slices, axis=0)
        
        # Return Tensor
        return torch.from_numpy(stacked_image).float()

# The entry point function calling the class above
def Train(folder_path, cfg):
    subjects_list = find_brats_subjects(folder_path,"train")
    return BraTS_2D_Dataset(subjects_list, cfg,k="train")

def Val(folder_path, cfg):
    subjects_list = find_brats_subjects(folder_path,"val")
    return BraTS_2D_Dataset(subjects_list, cfg,k="val")
