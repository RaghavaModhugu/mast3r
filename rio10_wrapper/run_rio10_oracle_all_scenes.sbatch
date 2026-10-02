#!/usr/bin/env bash
#SBATCH --job-name=rio10_mast3r_oracle_all_scenes
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
# Oracle-retrieval (GT pose: camera-center distance + viewing-direction angle, no learned
# retrieval model, no depth) + MASt3R-own-pointmap + PnP relocalization across all 10 RIO10
# scenes, as an upper bound on retrieval quality vs. the companion learned-retrieval run
# (run_rio10_retrieval_all_scenes.sbatch).
#
# --nodelist=node05: see run_rio10_retrieval_all_scenes.sbatch's comment -- same reasoning,
# /scratch/nagendra.raghava/... is node05-local, not cluster-shared.
#
# Delegates to run_oracle_all_scenes.sh by absolute path -- see
# run_rio10_retrieval_all_scenes.sbatch's comment on why (SLURM spool-staging only affects the
# top-level submitted script, not scripts it calls by absolute path afterward).
#
# Expected runtime: ~9.5h (34,415 total query frames across all 10 scenes, ~1s/query matching +
# PnP; cheaper than the retrieval job since oracle pairs are pure pose arithmetic, no ASMK/model
# load for pairs building). Validated end-to-end on scene01 (50 queries) after fixing the oracle
# ranking to use pose (position + viewing-direction angle) instead of position alone -- pure
# distance-ranked pairs had a median viewing-direction cosine similarity of -0.14 (facing away)
# and produced 100% PnP failure (inf error on every query); the pose-based fix gets median cosine
# similarity to 0.94 and median 17.5cm / 3.8 deg on that same smoke test. See
# rio10_wrapper/pairs_oracle.py and its git history for the full debugging trail.
#
# Submit with:
#   sbatch rio10_wrapper/run_rio10_oracle_all_scenes.sbatch
set -euo pipefail

exec bash /scratch/nagendra.raghava/localization_on_dynamic_scenes/mast3r/rio10_wrapper/run_oracle_all_scenes.sh
