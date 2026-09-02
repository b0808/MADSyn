# MADSyn
Code for the paper titled "Modality-Agnostic Diffusion Model for Missing Modality Synthesis in Brain MRI Scans".

## Environment

Create the conda environment:

```bash
conda env create -f environment_MADSyn.yml
conda activate MADSyn
```

The code expects Python dependencies from `environment_MADSyn.yml`. If you use a different environment, make sure PyTorch, PyTorch Lightning, Hydra, nibabel, torchmetrics, SimpleITK, TorchIO, and albumentations are installed.

## Data Layout

Prepare BraTS-style subject folders. Each subject folder should contain one NIfTI file per modality:

```text
data/
  train/
    BraTS-GLI-00000-000/
      BraTS-GLI-00000-000-t1c.nii.gz
      BraTS-GLI-00000-000-t1n.nii.gz
      BraTS-GLI-00000-000-t2f.nii.gz
      BraTS-GLI-00000-000-t2w.nii.gz
      BraTS-GLI-00000-000-seg.nii.gz
  val/
    BraTS-GLI-00001-000/
      BraTS-GLI-00001-000-t1c.nii.gz
      BraTS-GLI-00001-000-t1n.nii.gz
      BraTS-GLI-00001-000-t2f.nii.gz
      BraTS-GLI-00001-000-t2w.nii.gz
      BraTS-GLI-00001-000-seg.nii.gz
  test/
    BraTS-GLI-00002-000/
      BraTS-GLI-00002-000-t1c.nii.gz
      BraTS-GLI-00002-000-t1n.nii.gz
      BraTS-GLI-00002-000-t2f.nii.gz
      BraTS-GLI-00002-000-t2w.nii.gz
      BraTS-GLI-00002-000-seg.nii.gz
  sample_test/
    BraTS-GLI-00003-000/
      BraTS-GLI-00003-000-1.nii.gz
      BraTS-GLI-00003-000-2.nii.gz
```

The default paths are:

```text
DATA_DIR=./data
TRAIN_DATA_DIR=./data/train
VAL_DATA_DIR=./data/val
DATASET_STATS_FILE=./data/dataset_stats.json
LOG_DIR=./logs
```

You can either export these variables or override them through Hydra on the command line.

## Compute Normalization Statistics

Compute foreground mean and standard deviation values before training or inference:

```bash
python -m src.datamodules.convert \
  --data_root ./data/train \
  --output_file ./data/dataset_stats_train.json

python -m src.datamodules.convert \
  --data_root ./data/val \
  --output_file ./data/dataset_stats_val.json

python -m src.datamodules.convert \
  --data_root ./data/test \
  --output_file ./data/dataset_stats_test.json

python -m src.datamodules.convert \
  --data_root ./data/smaple_test \
  --output_file ./data/dataset_stats_smaple_test.json
```
```

## Training

Run the default training configuration:

```bash
python run.py \
  data_dir=./data \
  datamodule.cfg.train_data_dir=./data/train \
  datamodule.cfg.val_data_dir=./data/val \
  datamodule.cfg.stats_file_train=./data/dataset_stats_train.json \
  datamodule.cfg.stats_file_test=./data/dataset_stats_val.json \
  log_dir=./logs
```

Useful overrides:

```bash
python run.py trainer.gpus=[0] datamodule.cfg.batch_size=2 trainer.max_epochs=1000
python run.py trainer.gpus=[0,1] trainer.strategy=ddp datamodule.cfg.num_workers=4
python run.py debug=true
```

Checkpoints are written under the configured log directory through the Hydra and PyTorch Lightning checkpoint callbacks.

## Inference (Benchmarking & Evaluation)

Run synthesis and metric computation across all simulated missing conditions with a trained checkpoint:

```bash
python benchmarking.py \
  --model_path ./checkpoints/last_fold-1.ckpt \
  --data_root ./data/test \
  --stats_file ./data/dataset_stats_test.json \
  --save_root ./outputs/generated_volumes \
  --output_file ./outputs/metrics.csv \
  --batch_size 128 \
  --num_steps 500
```

The evaluation script simulates all 14 supported conditioning sets:

```text
t1c, t1n, t2f, t2w,
t1c+t1n, t1c+t2f, t1c+t2w,
t1n+t2f, t1n+t2w, t2f+t2w,
t1c+t1n+t2f, t1c+t1n+t2w,
t1c+t2f+t2w, t1n+t2f+t2w
```

It writes synthesized NIfTI volumes to `--save_root` and appends SSIM/PSNR metrics to `--output_file`. To compute metrics without saving volumes, add:

```bash
--no_save_volumes
```

To process only part of a split:

```bash
python inference.py \
  --model_path ./checkpoints/last_fold-1.ckpt \
  --data_root ./data/val \
  --stats_file ./data/dataset_stats_test.json \
  --start_index 0 \
  --end_index 10
```

## Modality-Agnostic Inference (Real-World Usage)

To generate missing modalities in a real-world scenario where you only have partial scans, use `inference_test.py`. 

This script is **modality-agnostic**—you can pass any 1 to 4 available MRI NIfTI files without explicitly labeling which modality is which. The model automatically recognizes the input modalities and generates the correct missing ones. 


```bash
python inference_test.py \
  --model_path ./checkpoints/last_fold-1.ckpt \
  --save_dir ./outputs/patient_001_results \
  --stats_file ./data/dataset_stats_sample_test.json \
  --inputs data/sample_test/BraTS-GLI-00002-000/BraTS-GLI-00002-000-1.nii.gz data/test/BraTS-GLI-00002-000/BraTS-GLI-00002-000-2.nii.gz \
  --batch_size 128 \
  --num_steps 500
```

The script will automatically reconstruct and save all 4 output modalities into the target `--save_dir`:
* `generated_t1c.nii.gz`
* `generated_t1n.nii.gz`
* `generated_t2f.nii.gz`
* `generated_t2w.nii.gz`

## Notes

- The repository does not include datasets or trained checkpoints.
- Local paths can be configured through environment variables, a `.env` file, or Hydra command-line overrides.
- Generated outputs, logs, datasets, checkpoints, cache files, and local environment files are ignored by `.gitignore`.
