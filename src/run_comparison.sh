#!/usr/bin/env bash

sim_type="firas"
version="v1"

run_name="comparison_${sim_type}_${version}"

if [[ "$sim_type" == "fossil" ]]; then
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --run-name "$run_name"
else
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --cg-dummy --run-name "$run_name"
fi