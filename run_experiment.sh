#!/bin/bash
# ------------------------------------------------------------------
# PixMix On-the-fly Experiments Runner
# Usage: bash run_experiment.sh {standard|moire|fractal|all} [gpu_id] [epochs] [warmup]
#        bash run_experiment.sh pixmix [fractals|fvis] [gpu_id] [epochs] [warmup]
# Example: bash run_experiment.sh moire 0
# ------------------------------------------------------------------

MODE=${1:-"help"}
shift 1
PIXMIX_PRESET_NAME=""
if [ "$MODE" = "pixmix" ]; then
  if [ $# -ge 1 ] && ! [[ "$1" =~ ^[0-9]+$ ]]; then
    PIXMIX_PRESET_NAME="$1"
    shift 1
  fi
fi
GPU_ID=${1:-0}
EPOCHS=${2:-100}
WARMUP=${3:-5}
shift 3
EXTRA_ARGS=("$@")

# 作業ディレクトリ（スクリプトの実行場所からの相対パスや環境変数を利用）
WORK_DIR=${WORK_DIR:-"$(pwd)"}

PRESET_FRACTALS=${PRESET_FRACTALS:-"fractals"}
PIXMIX_FVIS_DIR="/path/to/mixingsets/fractals_and_fvis/first_layers_resized256_onevis"
IPMIX_PRESET=${IPMIX_PRESET:-"ipmix_best"}
DIFFUSEMIX_PRESET=${DIFFUSEMIX_PRESET:-"diffusemix_best"}
LAYERMIX_PRESET=${LAYERMIX_PRESET:-"layermix_best"}
IPMIX_ALL_OPS=${IPMIX_ALL_OPS:-0}
DATASET=${DATASET:-"imagenet"}

# データセットパス（利用者が書き換えることを前提としたプレースホルダー）
CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-"/path/to/cifar"}

ESTIMATE_ITERS=${ESTIMATE_ITERS:-0}
ESTIMATE_EXIT=${ESTIMATE_EXIT:-0}
RUN_TAG=${RUN_TAG:-""}
NO_TIMESTAMP=${NO_TIMESTAMP:-0}

DATA_ARGS="--dataset $DATASET"
if [ "$DATASET" = "cifar10" ] || [ "$DATASET" = "cifar100" ]; then
  DATA_ARGS="$DATA_ARGS --data $CIFAR_DATA_ROOT"
fi

ESTIMATE_ARGS=""
if [ "$ESTIMATE_ITERS" -gt 0 ]; then
  ESTIMATE_ARGS="--estimate-epoch-time-iters $ESTIMATE_ITERS"
  if [ "$ESTIMATE_EXIT" = "1" ]; then
    ESTIMATE_ARGS="$ESTIMATE_ARGS --estimate-epoch-time-exit"
  fi
fi

NUM_CLASSES_ARG="--num-classes 1000"
if [ "$DATASET" = "cifar10" ]; then
  NUM_CLASSES_ARG="--num-classes 10"
elif [ "$DATASET" = "cifar100" ]; then
  NUM_CLASSES_ARG="--num-classes 100"
fi

IPMIX_ALL_OPS_FLAG=""
if [ "$IPMIX_ALL_OPS" = "1" ]; then
  IPMIX_ALL_OPS_FLAG="--all-ops"
fi

if [ "$PIXMIX_PRESET_NAME" = "fvis" ]; then
  MIXING_SET_DIR="$PIXMIX_FVIS_DIR"
fi
PIX_MIXING_SET_ARG=()
if [ -n "$MIXING_SET_DIR" ]; then
  PIX_MIXING_SET_ARG=(--mixing-set "$MIXING_SET_DIR")
else
  PIX_MIXING_SET_ARG=(--mixing-set-preset "$PRESET_FRACTALS")
fi

IPMIX_SET_ARG=()
if [ -n "$IPMIX_SET_DIR" ]; then
  IPMIX_SET_ARG=(--mixing-set "$IPMIX_SET_DIR")
else
  IPMIX_SET_ARG=(--mixing-set-preset "$IPMIX_PRESET")
fi

DIFFUSEMIX_SET_ARG=()
if [ -n "$DIFFUSEMIX_SET_DIR" ]; then
  DIFFUSEMIX_SET_ARG=(--mixing-set "$DIFFUSEMIX_SET_DIR")
else
  DIFFUSEMIX_SET_ARG=(--mixing-set-preset "$DIFFUSEMIX_PRESET")
fi

DIFFUSEMIX_FRACTAL_PRESET=${DIFFUSEMIX_FRACTAL_PRESET:-"$DIFFUSEMIX_PRESET"}
DIFFUSEMIX_FRACTAL_ARG=()
if [ -n "$DIFFUSEMIX_FRACTAL_SET_DIR" ]; then
  DIFFUSEMIX_FRACTAL_ARG=(--diffusemix-fractal-set "$DIFFUSEMIX_FRACTAL_SET_DIR")
else
  DIFFUSEMIX_FRACTAL_ARG=(--diffusemix-fractal-preset "$DIFFUSEMIX_FRACTAL_PRESET")
fi

LAYERMIX_SET_ARG=()
if [ -n "$LAYERMIX_SET_DIR" ]; then
  LAYERMIX_SET_ARG=(--mixing-set "$LAYERMIX_SET_DIR")
else
  LAYERMIX_SET_ARG=(--mixing-set-preset "$LAYERMIX_PRESET")
fi

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found." >&2
  exit 1
fi
conda activate pixmix
cd "$WORK_DIR" || exit 1

echo "Running mode=$MODE on GPU=$GPU_ID for epochs=$EPOCHS (warmup=$WARMUP)"

COMMON_ARGS="$NUM_CLASSES_ARG --arch vit_base --gpu 0 --seed 42 \
--data-standard /path/to/ImageNet-1K/train \
--data-val /path/to/ImageNet-1K/val \
--imagenet-r-dir /path/to/ImageNet-R \
--imagenet-c-dir /path/to/ImageNet-C \
--epochs $EPOCHS --warmup-epochs $WARMUP --batch-size 256 --batch-size-val 256 \
--optimizer adamw --lr 3e-3 --weight-decay 0.05 \
--mixup-alpha 0.0 --cutmix-alpha 0.0 --log-first-epoch-time $DATA_ARGS $ESTIMATE_ARGS"

make_save_dir() {
  local base="$1"
  local suffix=""
  if [ -n "$RUN_TAG" ]; then
    suffix="_$RUN_TAG"
  elif [ "$NO_TIMESTAMP" = "1" ]; then
    suffix=""
  else
    suffix="_$(date +%Y%m%d_%H%M)"
  fi
  local dir="${base}${suffix}"
  if [ -d "$dir" ]; then
    dir="${dir}_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "$dir"
}

write_command() {
  local cmd_file="$1"
  {
    echo "#!/bin/bash"
    echo "set -eu"
    printf '%q ' bash ./run_experiment.sh "$MODE"
    if [ "$MODE" = "pixmix" ] && [ -n "$PIXMIX_PRESET_NAME" ]; then
      printf '%q ' "$PIXMIX_PRESET_NAME"
    fi
    printf '%q ' "$GPU_ID" "$EPOCHS" "$WARMUP" "${EXTRA_ARGS[@]}"
    echo
  } > "$cmd_file"
  chmod +x "$cmd_file"
}

write_env() {
  local env_file="$1"
  {
    printf 'MIXING_SET_DIR=%s\n' "${MIXING_SET_DIR-}"
    printf 'IPMIX_SET_DIR=%s\n' "${IPMIX_SET_DIR-}"
    printf 'DIFFUSEMIX_SET_DIR=%s\n' "${DIFFUSEMIX_SET_DIR-}"
    printf 'DIFFUSEMIX_FRACTAL_SET_DIR=%s\n' "${DIFFUSEMIX_FRACTAL_SET_DIR-}"
    printf 'LAYERMIX_SET_DIR=%s\n' "${LAYERMIX_SET_DIR-}"
    printf 'RUN_TAG=%s\n' "${RUN_TAG-}"
    printf 'NO_TIMESTAMP=%s\n' "${NO_TIMESTAMP-}"
    printf 'IPMIX_ALL_OPS=%s\n' "${IPMIX_ALL_OPS-}"
  } > "$env_file"
}

setup_run_logs() {
  local log_file="$1"
  mkdir -p "$(dirname "$log_file")"
  exec > >(tee -a "$log_file") 2>&1
}

run_standard() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_standard_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Standard] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Standard] Finished."
}

run_cutout() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_cutout_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Cutout] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --cutout \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Cutout] Finished."
}

run_mixup() {
  MIXUP_ALPHA=${MIXUP_ALPHA:-0.8}
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_mixup_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[MixUp] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixup-alpha "$MIXUP_ALPHA" \
    --cutmix-alpha 0.0 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[MixUp] Finished."
}

run_cutmix() {
  CUTMIX_ALPHA=${CUTMIX_ALPHA:-1.0}
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_cutmix_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[CutMix] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixup-alpha 0.0 \
    --cutmix-alpha "$CUTMIX_ALPHA" \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[CutMix] Finished."
}

run_autoaugment() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_autoaugment_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[AutoAugment] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --auto-augment \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[AutoAugment] Finished."
}

run_randaugment() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_randaugment_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[RandAugment] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --rand-augment \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[RandAugment] Finished."
}

run_augmix() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_augmix_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[AugMix] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --augmix \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[AugMix] Finished."
}

run_gridmask() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_gridmask_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[GridMask] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --gridmask \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[GridMask] Finished."
}

run_pixmix_offline() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_pixmix_offline_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[PixMix Offline] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    "${PIX_MIXING_SET_ARG[@]}" \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[PixMix Offline] Finished."
}

run_diffusemix() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_diffusemix_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  if [ "${DIFFUSEMIX_SET_ARG[0]}" = "--mixing-set" ]; then
    local diffusemix_set_path="${DIFFUSEMIX_SET_ARG[1]}"
    local diffusemix_set_name
    diffusemix_set_name="$(basename "$diffusemix_set_path")"
    echo "[run_experiment.sh] mixing_set_path=$diffusemix_set_path mixing_set_name=$diffusemix_set_name"
  else
    echo "[run_experiment.sh] mixing_set_path= mixing_set_name= mixing_set_preset=${DIFFUSEMIX_SET_ARG[1]}"
  fi
  echo "=========================================================="
  echo "[DiffuseMix] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method diffusemix \
    "${DIFFUSEMIX_SET_ARG[@]}" \
    "${DIFFUSEMIX_FRACTAL_ARG[@]}" \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[DiffuseMix] Finished."
}

run_ipmix() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_ipmix_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  if [ "${IPMIX_SET_ARG[0]}" = "--mixing-set" ]; then
    local ipmix_set_path="${IPMIX_SET_ARG[1]}"
    local ipmix_set_name
    ipmix_set_name="$(basename "$ipmix_set_path")"
    echo "[run_experiment.sh] mixing_set_path=$ipmix_set_path mixing_set_name=$ipmix_set_name"
  else
    echo "[run_experiment.sh] mixing_set_path= mixing_set_name= mixing_set_preset=${IPMIX_SET_ARG[1]}"
  fi
  echo "=========================================================="
  echo "[IPMix] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  if [ "${IPMIX_SET_ARG[0]}" = "--mixing-set" ]; then
    local ipmix_set_path="${IPMIX_SET_ARG[1]}"
    local ipmix_set_name
    ipmix_set_name="$(basename "$ipmix_set_path")"
    echo "[run_experiment.sh] mixing_method=ipmix mixing_set_path=$ipmix_set_path mixing_set_name=$ipmix_set_name"
  else
    echo "[run_experiment.sh] mixing_method=ipmix mixing_set_preset=${IPMIX_SET_ARG[1]}"
  fi
  if [ "$IPMIX_ALL_OPS" = "1" ]; then
    echo "[run_experiment.sh] ipmix_all_ops_enabled=1"
  else
    echo "[run_experiment.sh] ipmix_all_ops_enabled=0"
  fi
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method ipmix \
    "${IPMIX_SET_ARG[@]}" \
    $IPMIX_ALL_OPS_FLAG \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[IPMix] Finished."
}

run_layermix() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_layermix_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  if [ "${LAYERMIX_SET_ARG[0]}" = "--mixing-set" ]; then
    local layermix_set_path="${LAYERMIX_SET_ARG[1]}"
    local layermix_set_name
    layermix_set_name="$(basename "$layermix_set_path")"
    echo "[run_experiment.sh] mixing_set_path=$layermix_set_path mixing_set_name=$layermix_set_name"
  else
    echo "[run_experiment.sh] mixing_set_path= mixing_set_name= mixing_set_preset=${LAYERMIX_SET_ARG[1]}"
  fi
  echo "=========================================================="
  echo "[LayerMix] Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method layermix \
    "${LAYERMIX_SET_ARG[@]}" \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[LayerMix] Finished."
}

run_moire() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_moire_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Moire] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend moire \
    --online-moire-freq-min 1 \
    --online-moire-freq-max 100 \
    --online-moire-centers-min 1 \
    --online-moire-centers-max 3 \
    --online-moire-margin 0.08 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Moire] Finished."
}

run_fractal() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_fractal_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Fractal] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend fractal \
    --online-fractal-iters 40000 \
    --online-fractal-instances-min 1 \
    --online-fractal-instances-max 3 \
    --online-fractal-scale-min 0.4 \
    --online-fractal-scale-max 0.85 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Fractal] Finished."
}

run_deadleaves() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_deadleaves_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[DeadLeaves] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend deadleaves \
    --online-deadleaves-variant shapes \
    --online-deadleaves-shapes-min 250 \
    --online-deadleaves-shapes-max 400 \
    --online-deadleaves-radius-min 4.0 \
    --online-deadleaves-radius-max 40.0 \
    --online-deadleaves-bg uniform \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[DeadLeaves] Finished."
}

run_perlin() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_perlin_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Perlin] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend perlin \
    --online-perlin-mode fbm \
    --online-perlin-octaves-min 4 \
    --online-perlin-octaves-max 7 \
    --online-perlin-scale-min 32.0 \
    --online-perlin-scale-max 96.0 \
    --online-perlin-persistence-min 0.45 \
    --online-perlin-persistence-max 0.6 \
    --online-perlin-lacunarity-min 1.8 \
    --online-perlin-lacunarity-max 2.2 \
    --online-perlin-perlin-scale 64.0 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Perlin] Finished."
}

run_stripe() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_stripe_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Stripe] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend stripe \
    --online-stripe-freq-min 1 \
    --online-stripe-freq-max 100 \
    --online-stripe-amp 0.5 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Stripe] Finished."
}

run_fourier2019() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_fourier2019_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[Fourier2019] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend fourier2019 \
    --online-fourier2019-mode uniform \
    --online-fourier2019-r-min 1 \
    --online-fourier2019-r-max 50 \
    --online-fourier2019-amp 0.5 \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[Fourier2019] Finished."
}

run_afa() {
  SAVE_DIR=$(make_save_dir "./experiments/vit_base_afa_online_${EPOCHS}ep")
  RUN_ID=$(basename "$SAVE_DIR")
  LOG_FILE="$SAVE_DIR/${RUN_ID}.train.log"
  CMD_FILE="$SAVE_DIR/${RUN_ID}.command.sh"
  ENV_FILE="$SAVE_DIR/${RUN_ID}.env.txt"
  setup_run_logs "$LOG_FILE"
  write_command "$CMD_FILE"
  write_env "$ENV_FILE"
  echo "[run_experiment.sh] LOG_FILE=$LOG_FILE"
  echo "[run_experiment.sh] CMD_FILE=$CMD_FILE"
  echo "[run_experiment.sh] ENV_FILE=$ENV_FILE"
  echo "=========================================================="
  echo "[AFA] On-the-fly Experiment Started on GPU $GPU_ID"
  echo "=========================================================="
  echo "[run_experiment.sh] EXTRA_ARGS=${EXTRA_ARGS[*]}"
  echo "[run_experiment.sh] SAVE_DIR=$SAVE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_onthefly.py \
    $COMMON_ARGS \
    --mixing-method pixmix \
    --online-mixing \
    --online-backend afa \
    --online-afa-lambda 0.05 \
    --online-afa-f-min 1 \
    --online-afa-f-max 224 \
    --online-afa-per-channel \
    "${EXTRA_ARGS[@]}" \
    --save "$SAVE_DIR"
  echo "[AFA] Finished."
}

case "$MODE" in
  standard)
    run_standard
    ;;
  cutout)
    run_cutout
    ;;
  mixup)
    run_mixup
    ;;
  cutmix)
    run_cutmix
    ;;
  autoaugment)
    run_autoaugment
    ;;
  randaugment)
    run_randaugment
    ;;
  augmix)
    run_augmix
    ;;
  gridmask)
    run_gridmask
    ;;
  pixmix)
    run_pixmix_offline
    ;;
  diffusemix)
    run_diffusemix
    ;;
  layermix)
    run_layermix
    ;;
  ipmix)
    run_ipmix
    ;;
  moire)
    run_moire
    ;;
  fractal|coloredfractal)
    run_fractal
    ;;
  deadleaves)
    run_deadleaves
    ;;
  perlin)
    run_perlin
    ;;
  stripe)
    run_stripe
    ;;
  fourier2019)
    run_fourier2019
    ;;
  afa)
    run_afa
    ;;
  all)
    run_standard
    run_cutout
    run_mixup
    run_cutmix
    run_autoaugment
    run_augmix
    run_gridmask
    run_pixmix_offline
    run_diffusemix
    run_layermix
    run_ipmix
    run_moire
    run_fractal
    run_deadleaves
    run_perlin
    run_stripe
    run_fourier2019
    run_afa
    ;;
  *)
    echo "Error: Unknown mode '$MODE'"
    echo "Usage: bash run_experiment.sh {standard|cutout|mixup|cutmix|autoaugment|augmix|gridmask|pixmix|diffusemix|layermix|ipmix|moire|fractal|coloredfractal|deadleaves|perlin|stripe|fourier2019|afa|all} [gpu_id] [epochs] [warmup]"
    echo "       bash run_experiment.sh pixmix [fractals|fvis] [gpu_id] [epochs] [warmup]"
    exit 1
    ;;
esac