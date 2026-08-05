#!/usr/bin/env bash

set -euo pipefail

sim_type="firas"
run_name="binned_${sim_type}_v6"

python binned_mapmaker.py --run-name "$run_name" --sim-type "$sim_type"

cd ../output/binned/${sim_type}
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 binned_${sim_type}.gif