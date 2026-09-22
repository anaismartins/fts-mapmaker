#!/usr/bin/env bash

set -euo pipefail

sim_type="firas"
mapmaker_type="legacy"
version="v3"
mode="debug" #"release"

run_name="${mapmaker_type}_${sim_type}_${version}"

MPLBACKEND=Agg NUMBA_NUM_THREADS=8 python ${mapmaker_type}_mapmaker.py --run-name "$run_name" --sim-type "$sim_type" --plots "debug" --firas_ss

if [[ "$mapmaker_type" == "legacy" || "$mapmaker_type" == "cg" ]]; then
    path="../output/${mapmaker_type}/${sim_type}/"
else
    path="../output/${mapmaker_type}/${sim_type}/maps/"
fi
cd "$path"
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 ${mapmaker_type}_${sim_type}_${add_on}.gif