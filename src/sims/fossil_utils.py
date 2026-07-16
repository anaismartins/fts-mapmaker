from memory_profiler import profile

def resolve_worker_count(n_batches):
    if args.workers is not None:
        return max(1, min(args.workers, n_batches))
    return max(1, min(available_cpu_count(), n_batches))

@profile
def create_pointings():
    # instrument parameters
    survey_len = 4 # years
    survey_time = survey_len * 365.25 * 24 * 3600 # seconds
    obs_eff = 0.7

    # run for full survey  using parallelization
    n_batches = int(survey_len * 365.25 * obs_eff) # one day batches
    n_workers = resolve_worker_count(n_batches)
    print(f"\n{'='*60}")
    print(f"Starting parallel processing of {n_batches} batches")
    print(f"Using {n_workers} workers (CPU cores available: {available_cpu_count()})")
    print(f"{'='*60}\n")

    t_start = time()
    with Pool(n_workers) as pool:
        results = pool.map(calculate_batch, range(n_batches))
    t_end = time()

    print(f"\n{'='*60}")
    print(f"Parallel processing complete!")
    print(f"Total time: {t_end - t_start:.2f} seconds")
    print(f"Average time per batch: {(t_end - t_start)/n_batches:.2f} seconds")
    print(f"{'='*60}\n")

    # Combine results
    print("Combining results from all batches...")
    # extract pix from results
    pix_list, lon_list, lat_list = zip(*results)
    pix_ecl = np.concatenate(pix_list)
    ecl_lon = np.concatenate(lon_list)
    ecl_lat = np.concatenate(lat_list)

    # save all pointings
    np.savez(pointing_cache, pix=pix_ecl, lon=ecl_lon, lat=ecl_lat)
    print(f"Saved pointings to {pointing_cache} --------------------------------------------------")