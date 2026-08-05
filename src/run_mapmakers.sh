#!/usr/bin/env bash

set -euo pipefail

sim_type="firas"
mapmaker_type="white_noise"
version="v3"

run_name="${mapmaker_type}_${sim_type}_${version}"

python ${mapmaker_type}_mapmaker.py --run-name "$run_name" --sim-type "$sim_type"

cd ../output/${mapmaker_type}/${sim_type}
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 ${mapmaker_type}_${sim_type}.gif