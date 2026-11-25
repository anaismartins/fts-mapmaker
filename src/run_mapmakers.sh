#!/usr/bin/env bash

set -euo pipefail

sim_type="fossil"
sim_name="fossil_sim_v40_noiseless_nside128"
mapmaker_type="cg"

# --run-name only labels the profiling log; outputs go to ../output/<mapmaker>/<sim_type>/<sim_name>/
run_name="${mapmaker_type}_${sim_name}"

MPLBACKEND=Agg NUMBA_NUM_THREADS=8 python ${mapmaker_type}_mapmaker.py --run-name "$run_name" --sim-type "$sim_type" --plots "debug" --noise --sim-name "$sim_name"

if [[ "$mapmaker_type" == "legacy" || "$mapmaker_type" == "cg" ]]; then
    path="../output/${mapmaker_type}/${sim_type}/${sim_name}/"
else
    path="../output/${mapmaker_type}/${sim_type}/${sim_name}/maps/"
fi
cd "$path"
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 "${mapmaker_type}_${sim_name}.gif"