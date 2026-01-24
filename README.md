# ADLM practical - Implicit Neural Representations and the Effect of Latent Size and Resolution

**Team:** Tiannan Zheng | Khadim Sarr | Michael Borisov  
**Supervisors:** Nil Stolt | Matan Atad

### Overview

This project studies implicit neural representations (INRs) on MedMNIST data with two pipelines:

1. latent size/spatial resolution sweeps
2. vector-quantized INRs with codebooks

### Repository layout

- `train.py`: latent size/spatial resolution sweeps.
- `train_vq.py`: codebook/VQ pipeline (Lightning).
- `src/model.py`: MLP decoder, spatial latents, and VQ models.
- `src/trainer.py`: Lightning training module for VQINR.
- `src/dataset.py`: pixel-coordinate dataset, full-image dataset, and MedMNIST loader.
- `src/utils.py`: evaluation, plotting, and experiment folder helpers.
- `config/`: JSON configs for reproducible runs.

### Configuration

Training is controlled via JSON configs. Example files live in `config/`.

Latent size/resolution (train.py) key knobs:

- `LATENT_SIZES`: list of total latent parameters per image to compare.
- `LATENT_SPATIAL_DIMS`: list of spatial grid sizes `s` to compare per `LATENT_SIZE`.
- `LATENT_FEATURE_DIM`: decoder input width after the conv (fixed for fair `s` comparisons).
  For `s>1`, a shared 3x3 conv is applied to each image's latent grid before sampling per-pixel latents;
  for `s=1`, the conv is disabled and the model reduces to a single latent vector per image.
- `NUM_EPOCHS`, `BATCH_SIZE`, `VIZ_INTERVAL` (epochs), `NUM_WORKERS`, and dataset settings.

Codebook/VQ (train_vq.py) key knobs:

- `K_IMAGES` and `DATASET_NAMES`: total images and dataset sources.
- `LATENT_DIM`, `NUM_CODES`, `NUM_LATENT_VECTORS`: VQ codebook and residual quantization settings.
- `HIDDEN_SIZE`, `NUM_LAYERS`, `ACTIVATION`, `NUM_FREQS`: decoder architecture and positional encoding.
- `NUM_EPOCHS`, `VIZ_INTERVAL` (epochs), `BATCH_SIZE`, `NUM_WORKERS`, and `LR`.

### Running

From the repo root:

```bash
python train.py -config config/latent_size_only_300.json -output-folder "experiments"
```

```bash
python train_vq.py -config config/vqinr_300.json -output-folder "experiments"
```

Flags:

- `-config` (required): path to the JSON config.
- `-debug` (default: `0`): `1` logs every epoch; `0` logs at `VIZ_INTERVAL` epochs.
- `-use-amp-tf32` (default: `1`): enables AMP/TF32 when CUDA is available.
- `-output-folder` (default: `.`): base folder for all outputs.

### Outputs

Latent size/resolution outputs:

Each `(latent_size, s)` pair gets its own run folder:

- `run_latent_<size>_s<s>/config.json`
- reconstructions: `recons_refs_epoch_<epoch>.png`
- t-SNE: `tsne_2d_epoch_<epoch>.png`

Summary plots in the output folder:

- `latent_size_comparison_sweep.png` (only when `LATENT_SPATIAL_DIMS` is `[1]`)
- `latent_all_runs_psnr_epochs.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)
- `latent_size_<size>_spatial_sweep.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)
- `latent_spatial_comparison.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)

Codebook/VQ outputs:

Each VQ run writes to:

- `run_vq_<config_name>/config.json`
- `training_metrics_psnr.png`
- `training_metrics_codebook.png`
- `evolution_<dataset>.png` (tracked images across epochs)
