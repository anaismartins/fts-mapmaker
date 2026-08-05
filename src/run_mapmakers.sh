#!/usr/bin/env bash

set -euo pipefail

run_name="binned_fossil_v10"

python binned_mapmaker.py --run-name "$run_name" --sim-type "firas"