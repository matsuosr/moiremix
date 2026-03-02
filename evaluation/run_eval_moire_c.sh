#!/bin/bash
# Usage: bash run_eval_moire_c.sh /path/to/model_best.pth.tar <gpu_id>

if [ $# -lt 2 ]; then
  echo "Usage: $0 /path/to/model_best.pth.tar <gpu_id>"
  exit 1
fi

CKPT_PATH="$1"
GPU_ID="$2"

if [ ! -f "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH"
  exit 1
fi

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found."
  exit 1
fi

# 環境固有のパスを汎用的なプレースホルダに変更
IMAGENET_C_MOIRE_DIR=${IMAGENET_C_MOIRE_DIR:-/path/to/benchmarks/ImageNet-C-moire}
EVAL_LOG_FILE=${EVAL_LOG_FILE:-""}
EVAL_ARCH=${EVAL_ARCH:-vit_base}

conda activate pixmix

# 作業ディレクトリをスクリプトの実行場所からの相対パスや環境変数を利用するように変更
cd "${WORK_DIR:-$(pwd)}" || exit 1

if [ -n "$EVAL_LOG_FILE" ]; then
  LOG_FILE="$EVAL_LOG_FILE"
else
  CKPT_DIR=$(dirname "$CKPT_PATH")
  CKPT_BASE=$(basename "$CKPT_PATH")
  LOG_FILE="${CKPT_DIR}/${CKPT_BASE}_moire_c_eval_gpu${GPU_ID}.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "" | tee -a "$LOG_FILE" >/dev/null
echo "[$(date '+%F %T')] Starting Moire-C evaluation for $CKPT_PATH (GPU $GPU_ID)" | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] IMAGENET_C_DIR=$IMAGENET_C_MOIRE_DIR" | tee -a "$LOG_FILE"

# データセットのパスを汎用的なプレースホルダに変更
CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
  --num-classes 1000 \
  --arch "$EVAL_ARCH" \
  --gpu 0 \
  --batch-size-val 256 \
  --workers 8 \
  --data-val /path/to/ImageNet-1K/val \
  --imagenet-c-dir "$IMAGENET_C_MOIRE_DIR" \
  --evaluate \
  --eval-only-imagenet-c \
  --resume "$CKPT_PATH" \
  --save "$(dirname "$CKPT_PATH")" \
  | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] Moire-C evaluation finished. Log: $LOG_FILE" | tee -a "$LOG_FILE"