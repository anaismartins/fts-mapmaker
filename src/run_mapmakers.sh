#!/usr/bin/env bash
# usage: ./run_mapmakers.sh <sim_name> [binned|cg|legacy]
# Outputs go to ../output/<mapmaker>/<sim_type>/<sim_name>/.

set -euo pipefail

sim_name="${1:?usage: ./run_mapmakers.sh <sim_name> [binned|cg|legacy]}"
mapmaker_type="${2:-cg}"

MPLBACKEND=Agg NUMBA_NUM_THREADS=8 python ${mapmaker_type}_mapmaker.py "$sim_name" --plots "debug"

if [[ "$mapmaker_type" == "binned" ]]; then
    cd ../output/${mapmaker_type}/*/"$sim_name"/maps/
else
    cd ../output/${mapmaker_type}/*/"$sim_name"/
fi
echo "Creating GIF from PNG files..."
convert *.png -delay 20 -loop 0 "${mapmaker_type}_${sim_name}.gif"
