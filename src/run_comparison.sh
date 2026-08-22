#!/usr/bin/env bash

sim_type="firas"
version="v0"
mode="release"

run_name="comparison_${sim_type}_${version}"

if [[ "$sim_type" == "fossil" ]]; then
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --run-name "$run_name" --plots "$mode"
else
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --run-name "$run_name" --plots "$mode"
fi