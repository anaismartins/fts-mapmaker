#!/usr/bin/env bash

sim_type="fossil"
sim_name="fossil_sim_v40_noiseless_nside128"
mode="release"

run_name="comparison_${sim_name}"

MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --sim-name "$sim_name" --run-name "$run_name" --plots "$mode"