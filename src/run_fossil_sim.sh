#!/usr/bin/env bash

run_name="fossil_sim_v38"
mode="debug" #"release"

owl1=owl{36..37}.uio.no
owl2=owl{39..46}.uio.no
owls=(owl1 owl2)
len=${#owls[@]}

nworkers=(64 128)

for (( i=0; i<${len}; i++ )); do
  c=${!owls[$i]}
  v=$(eval "echo $c")
  if [[ "$v" == *"$HOSTNAME"* ]]; then
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
    python -m sims.fossil --nworkers "$nworker" --plots "paper_only"
else
    echo "Running in debug mode with $nworker workers."
    /usr/bin/time -v -o ../output/time_stats.txt \
        python -u -m sims.fossil --nworkers "$nworker" --run-name "$run_name" --plots "debug" \
        --sim-type "fossil"

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