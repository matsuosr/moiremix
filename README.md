# moiremix

# 🚀 学習実験の再現 (Training Experiments)

### 🛠 1. 動作環境・必要ライブラリ

- **PyTorch系**: `torch`, `torchvision`
- **モデル系**: `timm` (v0.9以上推奨。ViTモデルのロードに必須)
- **画像処理・数値計算**: `Pillow`, `numpy`

### 📁 2. フォルダ構成

実行前に以下のスクリプトおよびモジュール群が同一階層に配置されていることを確認してください。

`onthefly_root/ (作業ディレクトリ)
  ├── run_experiment.sh               # 実行用メインシェルスクリプト
  ├── train_onthefly.py               # 学習用Pythonスクリプト
  ├── pixmix_utils.py                 # PixMix用ユーティリティ
  ├── calibration_tools.py            # 精度・キャリブレーション補助
  ├── mixing_presets.py               # オフラインMix用プリセット定義
  ├── aug/                            # データ拡張の実装ディレクトリ
  │   ├── layermix.py
  │   ├── diffusemix.py
  │   ├── ipmix.py
  │   ├── gridmask.py
  │   └── official_defaults_imagenet_vitb224.py
  └── mixing_image_generators/        # オンザフライ(Online)画像生成用ディレクトリ
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
      └── stripe.py`

*(注: 実行時に生成される `__pycache__` ディレクトリは構成ツリーから省略しています。)*

### 🗂 3. 設定すべきデータセットパス

`run_experiment.sh` および `mixing_presets.py` にはプレースホルダとなる汎用的なパスが設定されています。ご自身の実行環境に合わせて以下のパスを修正してください。

- **作業ディレクトリ設定**:
    - `WORK_DIR=${WORK_DIR:-"$(pwd)"}` （スクリプトの実行場所からの相対パスや環境変数を推奨）
- **ImageNet用パス** (`run_experiment.sh` の `COMMON_ARGS` 内で指定):
    - `-data-standard`: ImageNet Trainセット (例: `/path/to/ImageNet-1K/train`)
    - `-data-val`: ImageNet Valセット (例: `/path/to/ImageNet-1K/val`)
    - `-imagenet-r-dir`: ImageNet-Rセット (例: `/path/to/ImageNet-R`)
    - `-imagenet-c-dir`: ImageNet-Cセット (例: `/path/to/ImageNet-C`)
- **CIFAR用パス**:
    - `CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-"/path/to/cifar"}`
- **オフラインMix用画像パス** (PixMix、LayerMix、DiffuseMix、IPMixをオフラインで実行する場合に指定):
    - `run_experiment.sh` 内の `PIXMIX_FVIS_DIR="/path/to/mixingsets/fractals_and_fvis/first_layers_resized256_onevis"` など
    - `mixing_presets.py` 内の `PRESETS` 辞書に記載されている各種パスも `/path/to/mixingsets/...` となっているので合わせて変更してください。

### 💻 4. 実行コマンド例

`run_experiment.sh` を用いて学習を実行します。チェックポイントやログは自動生成されるディレクトリ（例: `./experiments/vit_base_...`）に保存されます。

**基本フォーマット**

Bash

`bash run_experiment.sh {モード名} [GPU_ID] [エポック数] [ウォームアップ数]`

- **標準的なベースライン (Standard) の実行**:
    - GPU 0, 100エポック, ウォームアップ5エポックで実行
    - `bash run_experiment.sh standard 0 100 5`
- **オンザフライ生成 (Moire) の実行**:
    - GPU 1, 90エポックで実行
    - `bash run_experiment.sh moire 1 90`
- **オフラインMix (PixMix + プリセット指定) の実行**:
    - `fractals` プリセットを使用
    - `bash run_experiment.sh pixmix fractals 0 100`
- **実装されている全実験の順次実行**:
    - GPU 0 で定義されているすべてのモードを順次実行
    - `bash run_experiment.sh all 0`

---

# 📊 評価および解析実験 (Evaluation & Analysis)

### 1. 標準評価・PGD堅牢性評価 (ImageNet-val / C / R & PGD)

- **使用スクリプト**: `run_eval_best.sh`
- **評価対象**: ImageNet validation, ImageNet-C, ImageNet-R, PGD攻撃
- **PGD実行条件 (厳守)**: $L_\infty$ ノルム, 50 steps, $\epsilon = 1/255$
- **事前準備**: スクリプト内の汎用パス（`PGD_DATA_VAL`, `IMAGENET_C_DIR` 等）を実環境に合わせて修正、または実行時に環境変数で指定

**【実行コマンド例】**

Bash

`# 環境変数を指定してバックグラウンド(nohup)で一括評価
nohup env \
PGD_ONLY=0 PGD_NORM=linf PGD_STEPS=50 \
PGD_EPS=0.00392156862745098 PGD_ALPHA=0.000980392156862745 \
PGD_BATCH=16 PGD_WORKERS=8 \
bash run_eval_best.sh ./experiments/vit_base_stripe_online_100ep/model_best.pth.tar 0 \
> ./experiments/vit_base_stripe_online_100ep/nohup_eval_full_plus_pgd_gpu0.out 2>&1 &

# フォアグラウンドでシンプルに実行 (GPU 0)
bash run_eval_best.sh ./experiments/vit_base_moire_online_100ep/model_best.pth.tar 0`

---

### 2. ImageNet-Moire ベンチマーク作成と評価

- **目的**: 実世界のモアレノイズに対するモデルの堅牢性検証
- **ベース手法**: AIM 2019 Challenge (arXiv:1911.02498) の数式を用いてImageNet-valを編集

### 🗂 2-1. データセットの生成

- **使用スクリプト**: `make_moire_c_aim.py`
- **対象データ**: ImageNet-validation (1,000クラス)
- **生成条件**: Severity 4 を指定

**【生成コマンド例】**

Bash

`python ./tools/make_moire_c_aim.py \
  --imagenet-val /path/to/ImageNet-1K/val \
  --out /path/to/benchmarks/ImageNet-C-moire/moire_aim \
  --severities 4 \
  --num-workers 8`

### 💻 2-2. 評価の実行

- **使用スクリプト**: `run_eval_moire_c.sh`
- **事前準備**: スクリプト内の `IMAGENET_C_MOIRE_DIR` を「2-1」の出力先パスに設定、または環境変数で指定

**【評価コマンド例】**

Bash

`# AFA Mix モデルの評価 (GPU 0)
bash run_eval_moire_c.sh ./experiments/vit_base_afa_online_100ep/model_best.pth.tar 0

# Baseline (Standard) モデルの評価 (GPU 0)
bash run_eval_moire_c.sh ./experiments/vit_base_standard_100ep/model_best.pth.tar 0

# Moire Mix モデルの評価 (GPU 3)
bash run_eval_moire_c.sh ./experiments/vit_base_moire_online_100ep/model_best.pth.tar 3`

# 周波数解析

- **目的**: Fourier Heatmapを用いた、周波数領域におけるモデルの堅牢性（ロバスト性）の解析と可視化
- **参考文献**: *A Fourier Perspective on Model Robustness in Computer Vision* [Yin+, NeurIPS2019]
- **ベース実装**: [gatheluck/FourierHeatmap](https://github.com/gatheluck/FourierHeatmap) をもとにViT向けにカスタマイズ
- **使用スクリプト**: `fhmap/apps/eval_fhmap_vit.py`
- **主要な実行パラメータ**:
    - `weightpath`: 評価対象モデルの重みファイルパス（必須）
    - `eps`: 摂動強度（例: `30.0`）
    - `num_samples`: 検証に使用する画像サンプル数（例: `100`）
    - `ignore_edge_size`: ヒートマップで評価から除外するエッジサイズ（例: `96`）
    - `batch_size`: バッチサイズ（例: `32`）

**【実行コマンド例】**

Bash

`# 環境変数(PYTHONPATH)を設定し、バックグラウンド(nohup)で解析を実行 (GPU 0)
nohup env PYTHONPATH=. python -u fhmap/apps/eval_fhmap_vit.py \
    --config-name eval_fhmap_vit \
    ignore_edge_size=96 \
    num_samples=100 \
    batch_size=32 \
    eps=30.0 \
    weightpath="./experiments/vit_base_moire_online_100ep/model_best.pth.tar" \
    > log_vit_moire_eps30_imagesample100_$(date +%Y%m%d_%H%M%S).txt 2>&1 &`
