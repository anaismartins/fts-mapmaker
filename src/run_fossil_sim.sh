#!/usr/bin/env bash

run_name="fossil_sim_v9"
mode="debug" #"release"

owls=(owl{39..46}.uio.no)
nworkers=(128 128 128 128 128 128 128 128)


for i in "${!owls[@]}"; do
  v="${owls[$i]}"
  if [[ "$HOSTNAME" == *"${v%%.*}"* || "$HOSTNAME" == "$v" ]]; then
    nworker="${nworkers[$i]}"
    break
  fi
done

export OMP_NUM_THREADS="$nworker"
export MKL_NUM_THREADS="$nworker"
export OPENBLAS_NUM_THREADS="$nworker"


# Run the program; its output goes directly to the terminal
if [ "$mode" = "release" ]; then
    echo "Running in release mode with $nworker workers..."
    python -m sims.fossil --workers "$nworker" --plots "paper_only"
else
    echo "Running in debug mode with $nworker workers."
    /usr/bin/time -v -o ../output/time_stats.txt \
        python -u -m sims.fossil --workers "$nworker" --run-name "$run_name" --plots "debug"

    if [ $? -ne 0 ]; then
      echo "Error: The simulation failed. Check the output above for details."
      exit 1
    fi

    # Parse the time stats file for Max RSS
    max_kb=$(awk '/Maximum resident set size/ {print $6}' ../output/time_stats.txt)
    max_gb=$(awk -v kb="$max_kb" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')

    echo "Maximum resident set size: ${max_kb} kB (~${max_gb} GiB) with $nworker workers"

    echo "Check profiling.txt for detailed profiling information."
fi