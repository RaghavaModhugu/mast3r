#!/bin/bash
# New file, part of the RIO10 wrapper. Does not modify any existing mast3r/dust3r file.
#
# Launches the interactive RIO10 relocalization debugging viewer (viz_app.py) on this node,
# bound to all interfaces so it's reachable over SSH port-forwarding. From your own machine:
#   ssh -L 7860:<this-node-hostname>:7860 <cluster-login>
# then browse http://localhost:7860 .
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAST3R_ROOT="$(dirname "$SCRIPT_DIR")"
ACE_G_ROOT="$(cd "$MAST3R_ROOT/../ace-g" && pwd)"

source "$ACE_G_ROOT/helper_scripts/setup.sh"
export PYTHONPATH="$MAST3R_ROOT:$MAST3R_ROOT/dust3r:$SCRIPT_DIR:${PYTHONPATH:-}"

RIO10_ROOT="$MAST3R_ROOT/../ace-g/rio10"
CKPT_DIR="$MAST3R_ROOT/checkpoints"
SERVER_PORT="${SERVER_PORT:-7860}"

python "$SCRIPT_DIR/viz_app.py" \
    --root "$RIO10_ROOT" \
    --weights "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --retrieval_model "$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" \
    --output_dir "$SCRIPT_DIR/results/oracle" \
    --pairs_dir "$SCRIPT_DIR/results" \
    --texture_change_csv "$SCRIPT_DIR/analysis/texture_change_features.csv" \
    --local_network --server_port "$SERVER_PORT" \
    "$@"
