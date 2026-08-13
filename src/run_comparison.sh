#!/usr/bin/env bash

sim_type="firas"

if [[ "$sim_type" == "fossil" ]]; then
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type"
else
    MPLBACKEND=Agg python compare.py --sim-type "$sim_type" --cg-dummy
fi