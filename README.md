# ADLM practical - Implicit Neural Representations and the Effect of Latent Size and Resolution

**Team:** Tiannan Zheng | Khadim Sarr | Michael Borisov  
**Supervisors:** Nil Stolt | Matan Atad

### Overview

This project studies how latent size and spatial latent resolution affect implicit neural representations (INRs) on MedMNIST data. A single training script reads a JSON config, runs a sweep over latent sizes and spatial resolutions, and logs reconstructions, t-SNE plots, and aggregate PSNR curves.

### Repository layout

- `train.py`: main training entry point (config-driven).
- `src/model.py`: decoder MLP and spatial latent wrapper.
- `src/dataset.py`: pixel-coordinate dataset and MedMNIST loader.
- `src/utils.py`: evaluation, plotting, and experiment folder helpers.
- `config/`: JSON configs for reproducible runs.

### Configuration

Training is controlled via a JSON config. Example files live in `config/`.

Key knobs:

- `LATENT_SIZES`: list of total latent parameters per image to compare.
- `LATENT_SPATIAL_DIMS`: list of spatial grid sizes `s` to compare per `LATENT_SIZE`.
- `LATENT_FEATURE_DIM`: decoder input width after the conv (fixed for fair `s` comparisons).
  For `s>1`, a shared 3x3 conv is applied to each image's latent grid before sampling per-pixel latents;
  for `s=1`, the conv is disabled and the model reduces to a single latent vector per image.
- `NUM_EPOCHS`, `BATCH_SIZE`, `VIZ_INTERVAL`, `NUM_WORKERS`, and dataset settings.

### Running

From the repo root:

```bash
python train.py -config config/latent_size_only_3000.json -output-folder "experiments"
```

Flags:

- `-config` (required): path to the JSON config.
- `-debug` (default: `0`): `1` logs every 100 steps; `0` logs at `VIZ_INTERVAL`.
- `-use-amp-tf32` (default: `1`): enables AMP/TF32 when CUDA is available.
- `-output-folder` (default: `.`): base folder for all outputs.

### Outputs

Each `(latent_size, s)` pair gets its own run folder:

- `run_latent_<size>_s<s>/config.json`
- reconstructions: `recons_refs_step_<step>.png`
- t-SNE: `tsne_2d_step_<step>.png`

Summary plots in the output folder:

- `latent_size_comparison_sweep.png` (only when `LATENT_SPATIAL_DIMS` is `[1]`)
- `latent_all_runs_psnr_steps.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)
- `latent_size_<size>_spatial_sweep.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)
- `latent_spatial_comparison.png` (when `LATENT_SPATIAL_DIMS` is not `[1]`)
