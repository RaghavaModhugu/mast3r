#!/usr/bin/env bash
#SBATCH --job-name=rio10_mast3r_retrieval_all_scenes
#SBATCH --partition=u22
#SBATCH --nodelist=node05
#SBATCH --chdir=/scratch/nagendra.raghava/localization_on_dynamic_scenes/mast3r/rio10_wrapper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nagendra.raghava/localization_on_dynamic_scenes/mast3r/rio10_wrapper/results/logs/%x_%j.out
#SBATCH --error=/scratch/nagendra.raghava/localization_on_dynamic_scenes/mast3r/rio10_wrapper/results/logs/%x_%j.err
#
# Learned-retrieval + MASt3R-own-pointmap + PnP relocalization across all 10 RIO10 scenes
# (the paper's "MASt3R" Unseen baseline, arxiv 2510.00978 -- no SfM, no dataset depth).
#
# --nodelist=node05: /scratch/nagendra.raghava/... is local to node05, not a cluster-wide shared
# filesystem -- confirmed via `hostname`/`sinfo` this session (node05 has the l40s GPU feature and
# is where the checkpoints/rio10_wrapper code/env were set up and validated). A job scheduled on
# any other node can't see any of that. Same reasoning as the pre-existing
# ace-g/helper_scripts/train_temporal_fusion_alternating_epoch_smoke.sbatch's own --nodelist pin,
# just for a different node since this repo's data lives on a different one.
#
# Delegates to run_all_scenes_retrieval.sh by absolute path rather than reimplementing its loop:
# SLURM stages *this* submitted script into a spool directory before executing it, so any
# ${BASH_SOURCE[0]}-based path derivation INSIDE this file would resolve wrong -- but a script this
# file calls by an explicit absolute path (as below) is a separate, non-staged invocation and
# resolves its own BASH_SOURCE correctly. Verified this session by running
# run_scene01_retrieval.sh end-to-end from a fully clean `env -i` shell (no inherited PATH/env),
# simulating exactly how sbatch invokes a job script.
#
# Expected runtime: ~9.5-10.5h (34,415 total query frames across all 10 scenes, ~1s/query matching
# + PnP, plus per-scene ASMK retrieval-pairs building). Validated end-to-end on scene01 first
# (2069 queries: median 9.2cm / 2.06 deg, beating the paper's own full-10-scene aggregate of
# 17cm / 5.5 deg -- see rio10_wrapper/results/scene01_retrieval_per_scene.csv).
#
# Submit with:
#   sbatch rio10_wrapper/run_rio10_retrieval_all_scenes.sbatch
set -euo pipefail

exec bash /scratch/nagendra.raghava/localization_on_dynamic_scenes/mast3r/rio10_wrapper/run_all_scenes_retrieval.sh
