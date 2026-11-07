#!/usr/bin/env bash
# FC-Former MHIF on CAVE x4: download the data, train, test. Needs HyperSpectralCollection (hisr).
#
#   bash scripts/run_cave_x4.sh            # download + train + test the trained model
#   bash scripts/run_cave_x4.sh download   # data from Hugging Face (elsting/MHIF) into data/MHIF/
#   bash scripts/run_cave_x4.sh train      # NUM_GPUS=4 bash scripts/run_cave_x4.sh train for 4 GPUs
#   bash scripts/run_cave_x4.sh test       # WEIGHTS=<file or checkpoint dir>, default weights/fcformer_cave_x4.pth
#
# The released CAVE weights take RGB in the reverse order of the elsting/MHIF files (flip_guide=true);
# any other weights are tested with flip_guide=false.
#
# Extra arguments after the step are passed to the config, e.g. max_epochs=500.
set -euo pipefail
cd "$(dirname "$0")/.."

STEP=${1:-all}
shift || true
NUM_GPUS=${NUM_GPUS:-1}
CFG=(--config-path="$PWD/configs" --config-name=fcformer_cave_x4)
TRAIN_DIR="$PWD/results/hisr/cave_x4/fcformer_cave_x4_train"

download() {
  local files=("cave/train_cave(with_up)x4_rgb.h5" "cave/test_cave(with_up)x4_rgb.h5")
  if command -v hf >/dev/null; then
    hf download elsting/MHIF --repo-type dataset --local-dir data/MHIF "${files[@]}"
  else
    huggingface-cli download elsting/MHIF --repo-type dataset --local-dir data/MHIF "${files[@]}"
  fi
}

train() {
  need accelerate_mhif HyperSpectralCollection
  local multi=()
  if [ "$NUM_GPUS" -gt 1 ]; then multi=(--multi_gpu); fi
  # Trained from scratch on these files, the RGB order needs no flip.
  accelerate launch ${multi[@]+"${multi[@]}"} --num_processes="$NUM_GPUS" "$(which accelerate_mhif)" "${CFG[@]}" \
    args.pretrained=null args.flip_guide=false hydra.run.dir="$TRAIN_DIR" "$@"
}

latest_checkpoint() {
  # Checkpoints are directories checkpoints/model_<epoch>_<iter>...; take the newest one.
  local ckpt
  ckpt=$(ls -td "$TRAIN_DIR"/checkpoints/model_*/ 2>/dev/null | head -1)
  echo "${ckpt%/}"
}

need() {
  command -v "$1" >/dev/null || { echo "$1 not found: install $2 (see README)" >&2; exit 1; }
}

test_model() {
  local weights=$1; shift
  case "$weights" in /*) ;; *) weights="$PWD/$weights" ;; esac
  local flip=()
  [ "$weights" = "$PWD/weights/fcformer_cave_x4.pth" ] || flip=(args.flip_guide=false)
  need naive_mhif HyperSpectralCollection
  naive_mhif "${CFG[@]}" eval=true args.pretrained="$weights" ${flip[@]+"${flip[@]}"} \
    hydra.run.dir="$PWD/results/hisr/cave_x4/fcformer_cave_x4_test" "$@"
}

case "$STEP" in
  download) download ;;
  train) train "$@" ;;
  test) test_model "${WEIGHTS:-weights/fcformer_cave_x4.pth}" "$@" ;;
  all)
    download
    train "$@"
    test_model "${WEIGHTS:-$(latest_checkpoint)}"
    ;;
  *) echo "unknown step: $STEP (download | train | test | all)" >&2; exit 1 ;;
esac
