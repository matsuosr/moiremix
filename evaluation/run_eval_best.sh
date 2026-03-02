#!/bin/bash
# Usage examples:
#   bash run_eval_best.sh /path/to/model_best.pth.tar 0           # full val/R/C + PGD (default)
#   PGD_ONLY=1 bash run_eval_best.sh /path/to/model_best.pth.tar 0 # PGD only
#   PGD_ENABLE=0 bash run_eval_best.sh /path/to/model_best.pth.tar 0 # skip PGD, keep standard eval

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

# 環境固有のパスを汎用的なプレースホルダや相対パスに変更
PGD_ENABLE=${PGD_ENABLE:-1}
PGD_TAG=${PGD_TAG:-""}
PGD_SEED=${PGD_SEED:-1}
PGD_DATA_VAL=${PGD_DATA_VAL:-/path/to/ImageNet-1K/val}
PGD_BATCH=${PGD_BATCH:-16}
PGD_WORKERS=${PGD_WORKERS:-8}
PGD_ARCH=${PGD_ARCH:-vit_base}
PGD_EPS=${PGD_EPS:-0.00392156862745098}
PGD_ALPHA=${PGD_ALPHA:-0.000980392156862745}
PGD_STEPS=${PGD_STEPS:-50}
PGD_NORM=${PGD_NORM:-linf}
PGD_RESULTS_NAME=${PGD_RESULTS_NAME:-metrics.json}
PGD_LOG_DIR=${PGD_LOG_DIR:-./logs/pgd}
PGD_ONLY=${PGD_ONLY:-0}
EVAL_LOG_FILE=${EVAL_LOG_FILE:-""}
IMAGENET_C_DIR=${IMAGENET_C_DIR:-/path/to/ImageNet-C}

conda activate pixmix

# 作業ディレクトリをスクリプトの実行場所からの相対パスや環境変数を利用するように変更
cd "${WORK_DIR:-$(pwd)}" || exit 1

if [ -n "$EVAL_LOG_FILE" ]; then
  LOG_FILE="$EVAL_LOG_FILE"
else
  CKPT_DIR=$(dirname "$CKPT_PATH")
  CKPT_BASE=$(basename "$CKPT_PATH")
  if [[ "$CKPT_BASE" == *_model_best.pth.tar ]]; then
    PREFIX=${CKPT_BASE%_model_best.pth.tar}
    LOG_FILE="${CKPT_DIR}/${PREFIX}_model_best_eval_gpu${GPU_ID}.log"
  else
    LOG_FILE="${CKPT_DIR}/${CKPT_BASE}_eval_gpu${GPU_ID}.log"
  fi
fi
mkdir -p "$(dirname "$LOG_FILE")"

SUPPORTS_PGD_ONLY=0
python train_onthefly.py -h >/tmp/train_help.txt 2>&1 && grep -q -- "--pgd-only" /tmp/train_help.txt && SUPPORTS_PGD_ONLY=1
rm -f /tmp/train_help.txt

PGD_ARGS=()
if [ "$PGD_ENABLE" -eq 1 ]; then
  PGD_ARGS+=(--pgd-eval)
  PGD_ARGS+=(--pgd-data-val "$PGD_DATA_VAL")
  PGD_ARGS+=(--pgd-batch-size "$PGD_BATCH")
  PGD_ARGS+=(--pgd-workers "$PGD_WORKERS")
  PGD_ARGS+=(--pgd-eps "$PGD_EPS")
  PGD_ARGS+=(--pgd-alpha "$PGD_ALPHA")
  PGD_ARGS+=(--pgd-steps "$PGD_STEPS")
  PGD_ARGS+=(--pgd-norm "$PGD_NORM")
  PGD_ARGS+=(--pgd-seed "$PGD_SEED")
  PGD_ARGS+=(--pgd-log-dir "$PGD_LOG_DIR")
  PGD_ARGS+=(--pgd-results-name "$PGD_RESULTS_NAME")
  PGD_ARGS+=(--pgd-tag "$PGD_TAG")
  if [ "$PGD_ONLY" -eq 1 ]; then
    if [ "$SUPPORTS_PGD_ONLY" -eq 1 ]; then
      PGD_ARGS+=(--pgd-only)
    else
      echo "[$(date '+%F %T')] --pgd-only not supported by train_onthefly.py, falling back to full evaluation." | tee -a "$LOG_FILE"
    fi
  fi
fi

echo "[run_eval_best.sh] PGD: only=${PGD_ONLY} enable=${PGD_ENABLE} norm=${PGD_NORM} steps=${PGD_STEPS} eps=${PGD_EPS} alpha=${PGD_ALPHA} batch=${PGD_BATCH} workers=${PGD_WORKERS}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE" >/dev/null
echo "[$(date '+%F %T')] Starting evaluation for $CKPT_PATH (GPU $GPU_ID)" | tee -a "$LOG_FILE"

# データセットのパスを汎用的なプレースホルダに変更
EVAL_ARGS=(--num-classes 1000 --arch "$PGD_ARCH" --gpu 0 --batch-size-val 256 --workers 8 \
  --data-val /path/to/ImageNet-1K/val --imagenet-r-dir /path/to/ImageNet-R \
  --imagenet-c-dir "$IMAGENET_C_DIR" --evaluate --resume "$CKPT_PATH" --save "$(dirname "$CKPT_PATH")")

if [ "$PGD_ONLY" -eq 1 ] && [ "$SUPPORTS_PGD_ONLY" -eq 1 ]; then
  # こちらも同様にプレースホルダに変更
  EVAL_ARGS=(--num-classes 1000 --arch "$PGD_ARCH" --gpu 0 --batch-size-val 256 --workers 8 \
    --data-val /path/to/ImageNet-1K/val --imagenet-r-dir /path/to/ImageNet-R \
    --imagenet-c-dir "$IMAGENET_C_DIR" --evaluate --resume "$CKPT_PATH" --save "$(dirname "$CKPT_PATH")")
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
  "${EVAL_ARGS[@]}" \
  "${PGD_ARGS[@]}" | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] Evaluation finished. Log: $LOG_FILE" | tee -a "$LOG_FILE"