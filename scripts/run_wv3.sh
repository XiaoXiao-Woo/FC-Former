#!/usr/bin/env bash
# FC-Former pansharpening on WorldView-3: download the data, train, test.
#
#   bash scripts/run_wv3.sh            # download + train + test the trained model
#   bash scripts/run_wv3.sh download   # data from Hugging Face (elsting/PanCollection) into data/
#   bash scripts/run_wv3.sh train      # NUM_GPUS=4 bash scripts/run_wv3.sh train for 4 GPUs
#   bash scripts/run_wv3.sh test       # WEIGHTS=<file or checkpoint dir>, default weights/fcformer_wv3.pth
#
# Extra arguments after the step are passed to the config, e.g. max_epochs=500.
set -euo pipefail
cd "$(dirname "$0")/.."

STEP=${1:-all}
shift || true
NUM_GPUS=${NUM_GPUS:-1}
CFG=(--config-path="$PWD/configs" --config-name=fcformer_wv3)
TRAIN_DIR="$PWD/results/pansharpening/wv3/fcformer_wv3_train"

download() {
  local files=(training_data/train_wv3_9714.h5 test_data/test_wv3_multiExm1.h5 test_data/test_wv3_OrigScale_multiExm1.h5)
  if command -v hf >/dev/null; then
    hf download elsting/PanCollection --repo-type dataset --local-dir data "${files[@]}"
  else
    huggingface-cli download elsting/PanCollection --repo-type dataset --local-dir data "${files[@]}"
  fi
}

train() {
  need accelerate_pansharpening PanCollection
  local multi=()
  if [ "$NUM_GPUS" -gt 1 ]; then multi=(--multi_gpu); fi
  accelerate launch ${multi[@]+"${multi[@]}"} --num_processes="$NUM_GPUS" "$(which accelerate_pansharpening)" "${CFG[@]}" \
    args.pretrained=null hydra.run.dir="$TRAIN_DIR" "$@"
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
  need naive_pansharpening PanCollection
  # Full resolution (HQNR, D_lambda, D_s), then reduced resolution (SAM, ERGAS, ...).
  naive_pansharpening "${CFG[@]}" eval=true args.pretrained="$weights" \
    hydra.run.dir="$PWD/results/pansharpening/wv3/fcformer_wv3_test_full" "$@"
  naive_pansharpening "${CFG[@]}" eval=true args.pretrained="$weights" \
    "dataset.wv3_test_path=[\${data_root}/test_data/test_wv3_multiExm1.h5]" base.test.test=reduce \
    hydra.run.dir="$PWD/results/pansharpening/wv3/fcformer_wv3_test_reduce" "$@"
}

case "$STEP" in
  download) download ;;
  train) train "$@" ;;
  test) test_model "${WEIGHTS:-weights/fcformer_wv3.pth}" "$@" ;;
  all)
    download
    train "$@"
    test_model "${WEIGHTS:-$(latest_checkpoint)}"
    ;;
  *) echo "unknown step: $STEP (download | train | test | all)" >&2; exit 1 ;;
esac
