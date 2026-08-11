#!/usr/bin/env bash

set -euo pipefail

sim_type="fossil"
mapmaker_type="binned"
version="v11"

run_name="${mapmaker_type}_${sim_type}_${version}"

python ${mapmaker_type}_mapmaker.py --run-name "$run_name" --sim-type "$sim_type"

if [[ "$mapmaker_type" == "binned" || "$mapmaker_type" == "cg" ]]; then
    path="../output/${mapmaker_type}/${sim_type}/"
else
    path="../output/${mapmaker_type}/${sim_type}/maps/"
fi
cd "$path"
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 ${mapmaker_type}_${sim_type}.gif