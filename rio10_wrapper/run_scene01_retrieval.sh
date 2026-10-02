#!/bin/bash
# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Runs the learned-retrieval + MASt3R-own-pointmap + PnP pipeline on scene01 only, as a
# sanity-check against the paper's reported RIO10 "MASt3R" numbers before scaling to all 10 scenes.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAST3R_ROOT="$(dirname "$SCRIPT_DIR")"
ACE_G_ROOT="$(cd "$MAST3R_ROOT/../ace-g" && pwd)"

source "$ACE_G_ROOT/helper_scripts/setup.sh"
export PYTHONPATH="$MAST3R_ROOT:$MAST3R_ROOT/dust3r:$SCRIPT_DIR:${PYTHONPATH:-}"

RIO10_ROOT="$MAST3R_ROOT/../ace-g/rio10"
CKPT_DIR="$MAST3R_ROOT/checkpoints"
OUT_DIR="$SCRIPT_DIR/results/scene01_retrieval"
PAIRS_FILE="$SCRIPT_DIR/results/pairs/scene01_retrieval_pairs.txt"
TOPK=10
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"  # queries per inference() call; raise on a bigger/freer GPU

mkdir -p "$OUT_DIR" "$(dirname "$PAIRS_FILE")"

python "$SCRIPT_DIR/pairs_retrieval.py" \
    --root "$RIO10_ROOT" --scene scene01 --topk "$TOPK" \
    --retrieval_model "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" \
    --backbone_weights "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --output "$PAIRS_FILE"

python "$SCRIPT_DIR/run_visloc_rio10.py" \
    --root "$RIO10_ROOT" --scene scene01 --pairs_file "$PAIRS_FILE" --topk "$TOPK" \
    --weights "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --pnp_mode poselib --pnp_max_points 5000 --query_batch_size "$QUERY_BATCH_SIZE" \
    --output_dir "$OUT_DIR" \
    --results_csv "$SCRIPT_DIR/results/scene01_retrieval_per_scene.csv" \
    "$@"
