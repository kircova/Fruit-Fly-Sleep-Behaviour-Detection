#!/usr/bin/env bash
# Reproduce every number and figure in the README from the raw clips.
#   FLYACT_DATA_ROOT=/path/to/clips scripts/reproduce.sh
set -euo pipefail
: "${FLYACT_DATA_ROOT:?set FLYACT_DATA_ROOT to the folder that holds the raw clips}"

flyact index          # de-duplicate the raw files               -> data/manifest.csv
flyact preprocess     # track the fly, write fly-centred crops   -> cache/
flyact sessions       # recover recordings, assign folds         -> data/manifest.csv

flyact baselines                                        # no neural network
flyact train --config configs/triplet_2d.yaml           # the model
for ablation in static_2d static_tsm tsm_triplet res224 random_split; do
  flyact train --config "configs/ablations/${ablation}.yaml"
done

flyact report                                           # docs/results.md, docs/figures/
flyact train --config configs/triplet_2d.yaml --final   # runs/triplet_2d/final.pt
