#!/usr/bin/env bash
# usage: ./run_comparison.sh <sim_name>

sim_name="${1:?usage: ./run_comparison.sh <sim_name>}"
plots="none"  # "none", "debug" or "paper_only"

MPLBACKEND=Agg python compare.py "$sim_name" --plots "$plots"
