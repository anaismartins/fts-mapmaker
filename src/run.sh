#!/usr/bin/env bash

mode="debug" #"release"
nworkers=16

# Run the program; its output goes directly to the terminal
if [ "$mode" = "release" ]; then
    echo "Running in release mode with $nworkers workers..."
    python -m sims.fossil --workers "$nworkers"
else
    echo "Running in debug mode with $nworkers workers..."
    /usr/bin/time -v -o ../output/time_stats.txt \
        python -u -m sims.fossil --no-plots --workers "$nworkers"

    # Parse the time stats file for Max RSS
    max_kb=$(awk '/Maximum resident set size/ {print $6}' ../output/time_stats.txt)
    max_gb=$(awk -v kb="$max_kb" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')

    echo "Maximum resident set size: ${max_kb} kB (~${max_gb} GiB) with $nworkers workers"

    echo "Check profiling.txt for detailed profiling information."
fi

