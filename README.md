## Depth-Aware Image Dehazing

Investigating Explicit Depth Integration Strategies for Image Dehazing

**Author:** Sejal Barshikar  
**Course:** CS7150 Deep Learning, Spring 2026

## Project Overview

This project investigates whether explicitly incorporating depth information improves deep learning-based image dehazing. We propose and compare three depth integration strategies:

1. **Depth Concatenation** - Depth as 4th input channel
2. **Depth-Guided Attention** - Depth modulates spatial attention
3. **Joint Multi-Task Learning** - Shared encoder with dehazing and depth decoders

## Installation

```bash
git clone https://github.com/sej07/depth-aware-dehazing.git
cd depth-aware-dehazing
pip install -r requirements.txt
```

## Model Weights

Download trained model weights from Google Drive:
[\https://drive.google.com/drive/folders/1L-VIvVxQgD68U3U1cVZ8UfeQ6yQvIDih?usp=drive_link](https://drive.google.com/drive/folders/1L-VIvVxQgD68U3U1cVZ8UfeQ6yQvIDih?usp=drive_link)

Extract into the `experiments/` folder. The structure should be:
```text
experiments/
├── aodnet_baseline/
│   └── checkpoints/
│       └── best.pth
├── ffanet_baseline/
│   └── checkpoints/
│       └── best.pth
├── aodnet_depth_concat/
│   └── checkpoints/
│       └── best.pth
├── depth_attention/
│   └── checkpoints/
│       └── best.pth
└── depth_joint/
└── checkpoints/
└── best.pth
```

## Demo

Run the demo notebook to see inference on sample images:

```bash
jupyter notebook demo.ipynb
```

## Project Structure
```text
depth-aware-dehazing/
├── src/
│   ├── models/         # Model architectures (AOD-Net, FFA-Net, depth methods)
│   ├── datasets/       # Dataset loaders (SOTS, OTS, O-HAZE, I-HAZE)
│   ├── losses/         # Loss functions
│   └── evaluation/     # Metrics (PSNR, SSIM)
├── scripts/            # Training and evaluation scripts
├── experiments/        # Trained model checkpoints
├── demo_images/        # Sample hazy images for demo
├── outputs/            # Evaluation results and visualizations
├── demo.ipynb          # Demo notebook (inference only)
├── requirements.txt    # Python dependencies
└── README.md
```

## Datasets

This project uses the following datasets:

### RESIDE (REalistic Single Image DEhazing)
- **OTS (Outdoor Training Set):** 8,646 synthetic hazy-clean pairs for training
- **SOTS (Synthetic Objective Testing Set):** 500 synthetic pairs for validation

Download from: [https://sites.google.com/view/reside-dehaze-datasets](https://sites.google.com/view/reside-dehaze-datasets)

### O-HAZE
- 45 real outdoor hazy-clean pairs

Download from: [https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)

### I-HAZE
- 30 real indoor hazy-clean pairs

Download from: [https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)

### Data Structure

After downloading, organize the data as follows:
```text
data/
├── reside/
│   ├── OTS/
│   │   ├── hazy/
│   │   └── clear/
│   └── SOTS/
│       └── outdoor/
│           ├── hazy/
│           └── gt/
├── ohaze/
│   ├── hazy/
│   └── GT/
└── ihaze/
├── hazy/
└── GT/
```
**Note:** For the demo notebook, only sample images in `demo_images/` are needed. Full datasets are required only for training.

## Results

| Model | Params | SOTS (PSNR/SSIM) | O-HAZE (PSNR/SSIM) | I-HAZE (PSNR/SSIM) |
|-------|--------|------------------|--------------------|--------------------|
| AOD-Net (baseline) | 1.7K | 20.55 / 0.85 | 16.43 / 0.63 | 15.76 / 0.67 |
| FFA-Net (baseline) | 1.5M | 19.47 / 0.58 | 15.15 / 0.39 | 15.07 / 0.53 |
| Depth Concat | 1.7K | 20.80 / 0.74 | 15.67 / 0.44 | 14.96 / 0.58 |
| Depth Attention | 1.6M | 19.42 / 0.55 | 15.62 / 0.35 | 15.00 / 0.50 |
| Joint Multi-Task | 10.8M | 18.52 / 0.55 | 15.72 / 0.37 | 14.19 / 0.49 |

## Key Findings

1. **Naive depth integration does not universally improve dehazing** - AOD-Net baseline outperforms all depth methods on real-world haze datasets.

2. **Depth is conditionally beneficial based on haze density:**

| Haze Level | Avg PSNR Change | Improved | Degraded |
|------------|-----------------|----------|----------|
| Light (A=0.08) | -1.78 dB | 12 | 129 |
| Medium (A=0.12) | -0.11 dB | 46 | 52 |
| Dense (A=0.16) | +1.75 dB | 77 | 12 |
| Very Dense (A=0.20) | +2.10 dB | 99 | 9 |

3. **Depth quality is not the bottleneck** - Oracle depth from clean images showed no improvement (20.80 vs 20.78 dB), indicating the limitation lies in integration strategy.

## Acknowledgments

- [MiDaS](https://github.com/isl-org/MiDaS) for monocular depth estimation
- [RESIDE](https://sites.google.com/view/raborwang/projects/reside) benchmark for datasets
- [AOD-Net](https://github.com/Boyiliee/AOD-Net) and [FFA-Net](https://github.com/zhilin007/FFA-Net) original implementations
