# MoireMix

## 🚀 Reproducing Training Experiments

### 🛠 1. Environment & Dependencies

* **PyTorch Framework:** `torch`, `torchvision`
* **Model Architecture:** `timm` (v0.9 or later is recommended; required for loading ViT models)
* **Image Processing & Computation:** `Pillow`, `numpy`

### 📁 2. Directory Structure

Ensure that the following scripts and modules are placed in the same working directory before execution.

```text
onthefly_root/ (Working Directory)
  ├── run_experiment.sh               # Main shell script for execution
  ├── train_onthefly.py               # Python script for training
  ├── pixmix_utils.py                 # Utility functions for PixMix
  ├── calibration_tools.py            # Accuracy and calibration tools
  ├── mixing_presets.py               # Preset definitions for offline mixing
  ├── aug/                            # Directory for data augmentation implementations
  │   ├── layermix.py
  │   ├── diffusemix.py
  │   ├── ipmix.py
  │   ├── gridmask.py
  │   └── official_defaults_imagenet_vitb224.py
  └── mixing_image_generators/        # Directory for on-the-fly (online) image generation
      ├── __init__.py
      ├── afa.py
      ├── base.py
      ├── colorbackground.py
      ├── coloredfractal.py
      ├── coloredmoire.py
      ├── deadleaves.py
      ├── fourier2019.py
      ├── moire.py
      ├── perlin.py
      └── stripe.py
```
*(Note: The `__pycache__` directory generated during execution is omitted from this tree.)*

### 🗂 3. Dataset Path Configuration

Generic placeholder paths are set within `run_experiment.sh` and `mixing_presets.py`. Please modify the following paths according to your local execution environment:

* **Working Directory Setting:**
    * `WORK_DIR=${WORK_DIR:-"$(pwd)"}` (Relative paths from the script execution location or environment variables are recommended)
* **ImageNet Paths** (Specified in `COMMON_ARGS` within `run_experiment.sh`):
    * `--data-standard`: ImageNet Train set (e.g., `/path/to/ImageNet-1K/train`)
    * `--data-val`: ImageNet Validation set (e.g., `/path/to/ImageNet-1K/val`)
    * `--imagenet-r-dir`: ImageNet-R set (e.g., `/path/to/ImageNet-R`)
    * `--imagenet-c-dir`: ImageNet-C set (e.g., `/path/to/ImageNet-C`)
* **CIFAR Paths:**
    * `CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-"/path/to/cifar"}`
* **Offline Mixing Image Paths** (Specify when running PixMix, LayerMix, DiffuseMix, or IPMix offline):
    * Update variables such as `PIXMIX_FVIS_DIR="/path/to/mixingsets/fractals_and_fvis/first_layers_resized256_onevis"` in `run_experiment.sh`.
    * Ensure to also update the various paths listed in the `PRESETS` dictionary within `mixing_presets.py`, which are currently set to `/path/to/mixingsets/...`.

### 💻 4. Execution Commands

Execute training using `run_experiment.sh`. Checkpoints and logs will be saved in automatically generated directories (e.g., `./experiments/vit_base_...`).

**Basic Format**
```bash
bash run_experiment.sh {mode} [GPU_ID] [epochs] [warmup_epochs]
```

* **Executing Standard Baseline:**
    * Run on GPU 0 for 100 epochs with 5 warmup epochs.
```bash
bash run_experiment.sh standard 0 100 5
```
* **Executing On-the-fly Generation (Moire):**
    * Run on GPU 1 for 90 epochs.
```bash
bash run_experiment.sh moire 1 90
```
* **Executing Offline Mix (PixMix + Preset Specification):**
    * Using the `fractals` preset.
```bash
bash run_experiment.sh pixmix fractals 0 100
```
* **Sequential Execution of All Implemented Experiments:**
    * Run all defined modes sequentially on GPU 0.
```bash
bash run_experiment.sh all 0
```

---

## 📊 Evaluation & Analysis

### 1. Standard Evaluation & PGD Robustness Evaluation (ImageNet-val / C / R & PGD)

* **Script:** `run_eval_best.sh`
* **Evaluation Targets:** ImageNet validation, ImageNet-C, ImageNet-R, PGD attack
* **Strict PGD Conditions:** $L_\infty$ norm, 50 steps, $\epsilon = 1/255$
* **Preparation:** Modify the generic paths (`PGD_DATA_VAL`, `IMAGENET_C_DIR`, etc.) in the script to match your local environment, or specify them via environment variables at runtime.

**Example Commands:**
```bash
# Run comprehensive evaluation in the background (nohup) with specified environment variables
nohup env \
PGD_ONLY=0 PGD_NORM=linf PGD_STEPS=50 \
PGD_EPS=0.00392156862745098 PGD_ALPHA=0.000980392156862745 \
PGD_BATCH=16 PGD_WORKERS=8 \
bash run_eval_best.sh ./experiments/vit_base_stripe_online_100ep/model_best.pth.tar 0 \
> ./experiments/vit_base_stripe_online_100ep/nohup_eval_full_plus_pgd_gpu0.out 2>&1 &

# Run standard evaluation simply in the foreground (GPU 0)
bash run_eval_best.sh ./experiments/vit_base_moire_online_100ep/model_best.pth.tar 0
```

---

### 2. ImageNet-Moire Benchmark Creation & Evaluation

* **Objective:** Verify model robustness against real-world Moiré noise.
* **Base Method:** Edit ImageNet-val images using the mathematical formulation derived from the [AIM 2019 Challenge (arXiv:1911.02498)](https://arxiv.org/abs/1911.02498).

#### 🗂 2.1 Dataset Generation

* **Script:** `make_moire_c_aim.py`
* **Target Data:** ImageNet-validation (1,000 classes)
* **Generation Condition:** Specify Severity 4

**Generation Command Example:**
```bash
python ./tools/make_moire_c_aim.py \
  --imagenet-val /path/to/ImageNet-1K/val \
  --out /path/to/benchmarks/ImageNet-C-moire/moire_aim \
  --severities 4 \
  --num-workers 8
```

#### 💻 2.2 Evaluation Execution

* **Script:** `run_eval_moire_c.sh`
* **Preparation:** Set `IMAGENET_C_MOIRE_DIR` in the script to the output path from step 2.1, or specify it via an environment variable.

**Evaluation Command Examples:**
```bash
# Evaluate AFA Mix model (GPU 0)
bash run_eval_moire_c.sh ./experiments/vit_base_afa_online_100ep/model_best.pth.tar 0

# Evaluate Baseline (Standard) model (GPU 0)
bash run_eval_moire_c.sh ./experiments/vit_base_standard_100ep/model_best.pth.tar 0

# Evaluate Moire Mix model (GPU 3)
bash run_eval_moire_c.sh ./experiments/vit_base_moire_online_100ep/model_best.pth.tar 3
```

---

## 🔍 Frequency Domain Analysis (Fourier Heatmap)

* **Objective:** Analyze and visualize model robustness in the frequency domain utilizing Fourier Heatmaps.
* **Reference:** *A Fourier Perspective on Model Robustness in Computer Vision* [Yin et al., NeurIPS 2019]
* **Base Implementation:** Customized for Vision Transformers (ViT) based on [gatheluck/FourierHeatmap](https://github.com/gatheluck/FourierHeatmap).
* **Script:** `fhmap/apps/eval_fhmap_vit.py`
* **Key Execution Parameters:**
    * `weightpath`: Path to the target model's weight file (Required).
    * `eps`: Perturbation magnitude (e.g., `30.0`).
    * `num_samples`: Number of image samples used for validation (e.g., `100`).
    * `ignore_edge_size`: Edge size to exclude from the heatmap evaluation (e.g., `96`).
    * `batch_size`: Batch size for evaluation (e.g., `32`).

**Execution Command Example:**
```bash
# Run analysis in the background (nohup) by setting the environment variable (PYTHONPATH) (GPU 0)
nohup env PYTHONPATH=. python -u fhmap/apps/eval_fhmap_vit.py \
    --config-name eval_fhmap_vit \
    ignore_edge_size=96 \
    num_samples=100 \
    batch_size=32 \
    eps=30.0 \
    weightpath="./experiments/vit_base_moire_online_100ep/model_best.pth.tar" \
    > log_vit_moire_eps30_imagesample100_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
```


### 💻 4. Execution Commands

Execute training using `run_experiment.sh`. Checkpoints and logs will be saved in automatically generated directories (e.g., `./experiments/vit_base_...`).

**Basic Format**
```bash
bash run_experiment.sh {mode} [GPU_ID] [epochs] [warmup_epochs]
```

#### 📌 Supported Experiment Modes
You can specify one of the following modes for the `{mode}` argument to run various baselines or mixing strategies:

* **Standard & Augmentation Baselines:**
  * `standard`, `cutout`, `mixup`, `cutmix`, `autoaugment`, `randaugment`, `augmix`, `gridmask`
* **On-the-fly (Online) Mixing:** *(Generates mixing images dynamically during the training loop)*
  * `moire`, `fractal` (or `coloredfractal`), `deadleaves`, `perlin`, `stripe`, `fourier2019`, `afa`
* **Offline Mixing:** *(Requires pre-generated mixing datasets)*
  * `pixmix`, `diffusemix`, `layermix`, `ipmix`
* **Run All:**
  * `all` *(Sequentially executes all defined experiments)*

#### 💡 Basic Execution Examples

**1. On-the-fly DeadLeaves Mixing**
```bash
bash run_experiment.sh deadleaves 0 100 5
```

**2. Standard MixUp Baseline**
```bash
bash run_experiment.sh mixup 1 100 5
```

**3. Standard Baseline**
```bash
bash run_experiment.sh standard 0 100 5
```

#### ⚠️ Special Execution: Specifying Custom Mixing Sets (Offline Mixing)
For offline mixing modes like `pixmix`, the script relies on pre-generated texture datasets. You can explicitly override the default dataset path by exporting the `MIXING_SET_DIR` environment variable before execution. 

This is particularly useful when evaluating your own generated datasets (e.g., a custom Moiré database) without modifying the preset files.

**Example: Running PixMix with a custom MoiréMix dataset in the background (nohup)**
```bash
# Specify the path to your custom mixing images
export MIXING_SET_DIR="/path/to/mixingsets/10000_MoireMix_20260218"

# Execute in the background and output logs to a specific file
nohup bash run_experiment.sh pixmix 4 100 0 \
  > ./experiments/nohup_pixmix_moireDB_100ep_$(date +%Y%m%d_%H%M).log 2>&1 &
```

#### 🔬 Ablation Studies (Hyperparameter Tuning)
You can seamlessly conduct ablation studies by appending specific configuration arguments to the `run_experiment.sh` command. These extra arguments are directly passed to the underlying `train_onthefly.py` script, allowing you to override default hyperparameters such as the model architecture, the number of Moiré centers, and frequency ranges.

**Example 1: Ablation on the Number of Moiré Centers ($N$)**
Run fast 30-epoch ablation experiments using the lightweight `vit_tiny` architecture, restricting the maximum number of Moiré centers to 1 or 2.
```bash
# Set max Moiré centers to 1 (N=1) on GPU 0
nohup bash ./run_experiment.sh moire 0 30 0 --arch vit_tiny --online-moire-centers-max 1 \
  > ./experiments/nohup_moire_N1_vittiny_gpu0_30ep_$(date +%Y%m%d_%H%M).log 2>&1 &

# Set max Moiré centers to 2 (N=2) on GPU 1
nohup bash ./run_experiment.sh moire 1 30 0 --arch vit_tiny --online-moire-centers-max 2 \
  > ./experiments/nohup_moire_N2_vittiny_gpu1_30ep_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Example 2: Ablation on the Frequency Range**
Investigate the impact of different Moiré frequency bands (e.g., Low vs. Mid frequencies) on model robustness.
```bash
# Low Frequency Band (1-33) on GPU 0
nohup bash ./run_experiment.sh moire 0 30 0 --arch vit_tiny --online-moire-freq-min 1 --online-moire-freq-max 33 \
  > ./experiments/nohup_moire_freqLow_vittiny_gpu0_30ep_$(date +%Y%m%d_%H%M).log 2>&1 &

# Mid Frequency Band (34-66) on GPU 3
nohup bash ./run_experiment.sh moire 3 30 0 --arch vit_tiny --online-moire-freq-min 34 --online-moire-freq-max 66 \
  > ./experiments/nohup_moire_freqMid_vittiny_gpu3_30ep_$(date +%Y%m%d_%H%M).log 2>&1 &
```
