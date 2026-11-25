"""
Configuration of a simulation version and of everything run on it.

Every setting lives in exactly one place:

1. Simulation parameters: *what* is simulated (resolution, noise, pointing, ...).
   One YAML file per version: configs/<sim_name>.yaml. The file name is the version name.
   When the simulation runs, the full parameter set (defaults filled in) is recorded next to the
   data in output/data/<sim_type>/<sim_name>/config.yaml, with the git commit, host and command.
   Once a version is simulated, its config file must not change: to change something, copy the
   file to a new name (a new version).
2. Runtime options: *how* a script runs (workers, plots, ...). Command line only. They do not
   change the results, so they are not part of a version.
3. Instrument constants that are the same for every version (IFG length, ...): globals.py.

Usage, from src/:
    python -m sims.fossil fossil_sim_v40_noiseless_nside128 --nworkers 64 --plots debug
    python binned_mapmaker.py fossil_sim_v40_noiseless_nside128
    python config.py fossil_sim_v40_noiseless_nside128     # print the parameters of a version

In code, `from config import cfg` gives the parameters and runtime options of the current run as
attributes: cfg.sim_name, cfg.sim_type, cfg.nside, cfg.noise, ..., cfg.plots, cfg.nworkers, ...,
plus cfg.npix, cfg.ifg_size, cfg.spec_size and cfg.log (the profiling log of this run).
"""

import argparse
import datetime
import os
import socket
import subprocess
import sys
from pathlib import Path

import yaml

import globals as g

CONFIG_DIR = g.ROOT_DIR / "configs"
RECORD_NAME = "config.yaml"      # parameter record of a simulation, in its data folder
RUN_INFO_NAME = "run_info.yaml"  # record of a mapmaker / compare run, in its output folder

# Every simulation parameter and its default. A config file can only set keys listed here, so a
# typo raises an error instead of being silently ignored.
DEFAULTS = {
    "fossil": {
        "nside": 128,           # resolution of the output maps (and of the pixelised pointing)
        "dust_nside": None,     # resolution of the sky model the IFGs are drawn from;
                                # null = native resolution of the Planck map (2048)
        "beam_deg": 1.6,        # FWHM the Planck dust map is smoothed with, in degrees
        "noise": True,          # add white noise to the IFGs
        "pointing": "default",  # pointing set in output/data/fossil/pointing/<pointing>/, shared by
                                # all versions that use it (it is expensive to generate)
    },
    "firas": {
        "nside": 32,
        "beam_deg": 7.0,
        "noise": True,
        "pointing": "firas",    # scanning strategy: "firas" (real FIRAS pointing) or "fossil"
                                # (middle sample of the FOSSIL "default" pointing)
    },
}


def load_params(sim: str) -> tuple:
    """
    Return (sim_name, params) of a version, given by name or by the path of its config file.

    If the version has already been simulated, its config file must still match the recorded
    parameters, so that a name always refers to one set of parameters.
    """
    if sim.endswith(".yaml"):
        sim_name, config_file = Path(sim).stem, Path(sim)
    else:
        sim_name, config_file = sim, CONFIG_DIR / f"{sim}.yaml"

    records = list(g.OUTPUT_DIR.glob(f"data/*/{sim_name}/{RECORD_NAME}"))
    recorded = yaml.safe_load(open(records[0]))["parameters"] if records else None
    if not config_file.exists():
        if recorded is None:
            raise FileNotFoundError(f"No version '{sim_name}': create {config_file} first.")
        return sim_name, recorded

    raw = yaml.safe_load(open(config_file)) or {}
    sim_type = raw.get("sim_type")
    if sim_type not in DEFAULTS:
        raise ValueError(f"{config_file}: sim_type must be one of {list(DEFAULTS)}.")
    unknown = set(raw) - set(DEFAULTS[sim_type]) - {"sim_type"}
    if unknown:
        raise ValueError(f"{config_file}: unknown parameters {sorted(unknown)}. "
                         f"Known: {sorted(DEFAULTS[sim_type])}.")
    params = {"sim_type": sim_type, **DEFAULTS[sim_type], **raw}

    if recorded is not None and recorded != params:
        diff = [f"  {k}: simulated with {recorded.get(k)!r}, config file says {params.get(k)!r}"
                for k in params if recorded.get(k) != params.get(k)]
        raise ValueError(f"{config_file} was edited after '{sim_name}' was simulated:\n"
                         + "\n".join(diff) + "\nCopy it to a new version name instead.")
    return sim_name, params


def save_record(directory: Path, filename: str = RUN_INFO_NAME):
    """
    Write the parameters, runtime options and provenance of the current run to directory/filename.

    If the code has uncommitted changes, the diff is saved next to it as <filename stem>.patch,
    so the run can always be reproduced.
    """
    git = ["git", "-C", str(g.ROOT_DIR)]
    commit = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    diff = subprocess.run(git + ["diff", "HEAD"], capture_output=True, text=True).stdout
    record = {
        "sim_name": cfg.sim_name,
        "parameters": params,
        "runtime": runtime,
        "provenance": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": commit.stdout.strip(),
            "uncommitted_changes": bool(diff),
            "host": socket.gethostname(),
            "command": " ".join(sys.argv),
            "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
                        if k in os.environ},
        },
    }
    with open(directory / filename, "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)
    patch = directory / f"{Path(filename).stem}.patch"
    if diff:
        patch.write_text(diff)
    elif patch.exists():
        patch.unlink()


parser = argparse.ArgumentParser(description="All simulation parameters come from "
                                 "configs/<sim>.yaml; the options below only control how the "
                                 "script runs.")
parser.add_argument("sim", help="Simulation version: the name of a file in configs/ (e.g. "
                    "fossil_sim_v40_noiseless_nside128) or the path to a config file.")
parser.add_argument("--plots", choices=["none", "debug", "paper_only"], default="none",
                    help="Which diagnostic plots to make. Default: none.")
parser.add_argument("--nworkers", type=int, default=os.cpu_count(),
                    help="Number of worker processes. Default: all CPUs.")
parser.add_argument("--chunk-size", type=int, default=200_000, help="Number of recorded IFGs "
                    "processed at a time in the FIRAS simulation. Lower it if the process runs "
                    "out of memory, e.g. with the FOSSIL pointing.")
parser.add_argument("--cg-dummy", action="store_true",
                    help="compare.py: use an empty map instead of the CG map.")
parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")

runtime = vars(parser.parse_args())
sim_name, params = load_params(runtime.pop("sim"))

cfg = argparse.Namespace(sim_name=sim_name, **params, **runtime)
cfg.npix = 12 * cfg.nside**2
cfg.ifg_size = g.IFG_SIZE[cfg.sim_type]
cfg.spec_size = g.SPEC_SIZE[cfg.sim_type]
cfg.log = g.OUTPUT_DIR / "profiling" / cfg.sim_type / sim_name / f"{Path(sys.argv[0]).stem}.txt"
cfg.log.parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"# {sim_name}")
    print(yaml.safe_dump(params, sort_keys=False), end="")
