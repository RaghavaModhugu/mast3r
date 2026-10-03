#!/bin/bash
# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Runs the oracle-retrieval (GT camera-center distance, no learned retrieval model, no depth) +
# MASt3R-own-pointmap + PnP pipeline across all 10 RIO10 scenes, as an upper bound on retrieval
# quality. Writes one row per scene to results/oracle_per_scene.csv and a paper-style pooled table.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAST3R_ROOT="$(dirname "$SCRIPT_DIR")"
ACE_G_ROOT="$(cd "$MAST3R_ROOT/../ace-g" && pwd)"

source "$ACE_G_ROOT/helper_scripts/setup.sh"
export PYTHONPATH="$MAST3R_ROOT:$MAST3R_ROOT/dust3r:$SCRIPT_DIR:${PYTHONPATH:-}"

RIO10_ROOT="$MAST3R_ROOT/../ace-g/rio10"
CKPT_DIR="$MAST3R_ROOT/checkpoints"
OUT_DIR="$SCRIPT_DIR/results/oracle"
RESULTS_CSV="$SCRIPT_DIR/results/oracle_per_scene.csv"
TOPK=10
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"  # queries per inference() call; raise on a bigger/freer GPU

mkdir -p "$OUT_DIR" "$OUT_DIR/pairs"
# RESUME=1 keeps the existing per-scene CSV and skips scenes that already wrote raw_errors.csv
[ "${RESUME:-0}" = 1 ] || rm -f "$RESULTS_CSV"

for i in $(seq -w 1 10); do
    SCENE="scene$i"
    PAIRS_FILE="$OUT_DIR/pairs/${SCENE}_oracle_pairs.txt"

    if [ -f "$OUT_DIR/${SCENE}_raw_errors.csv" ]; then
        echo "=== $SCENE: already done, skipping ==="
        continue
    fi

    echo "=== $SCENE: building oracle pairs ==="
    python "$SCRIPT_DIR/pairs_oracle.py" \
        --root "$RIO10_ROOT" --scene "$SCENE" --topk "$TOPK" \
        --output "$PAIRS_FILE"

    echo "=== $SCENE: running matching + PnP ==="
    python "$SCRIPT_DIR/run_visloc_rio10.py" \
        --root "$RIO10_ROOT" --scene "$SCENE" --pairs_file "$PAIRS_FILE" --topk "$TOPK" \
        --weights "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
        --pnp_mode poselib --pnp_max_points 5000 --query_batch_size "$QUERY_BATCH_SIZE" \
        --output_dir "$OUT_DIR" \
        --results_csv "$RESULTS_CSV" \
        "$@"
done

echo "=== paper-style pooled table (oracle retrieval, all 10 scenes) ==="
python "$SCRIPT_DIR/make_paper_table.py" --output_dir "$OUT_DIR" --label "MASt3R (oracle, ours)"
