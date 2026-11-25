"""
Single place that defines where versioned outputs live.

Every simulation version is identified by (sim_type, sim_name), e.g. ("fossil",
"fossil_sim_v40_noiseless_nside128"), and everything derived from it follows the layout

    ../output/data/<sim_type>/<sim_name>/<kind>.npy       ifgs, noise, ecl_lon, ecl_lat, pix_nsideN
    ../output/sims/<sim_type>/<sim_name>/<category>/      debug plots, dust_maps, ifgs, pix_hits
    ../output/hit_maps/<sim_type>/<sim_name>/
    ../output/<binned|cg|legacy>/<sim_type>/<sim_name>/   mapmaker outputs
    ../output/compare/<sim_type>/<sim_name>/

By default sim_type and sim_name are taken from the command line.
"""

from pathlib import Path

from argparser import args

OUTPUT = Path("../output")


def _resolve(sim_type, sim_name):
    sim_type = sim_type or args.sim_type
    sim_name = sim_name or args.sim_name
    if sim_type is None or sim_name is None:
        raise ValueError("Both --sim-type and --sim-name must be given.")
    return sim_type, sim_name


def _dir(root, *parts, sim_type=None, sim_name=None):
    sim_type, sim_name = _resolve(sim_type, sim_name)
    path = OUTPUT / root / sim_type / sim_name
    path = path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(sim_type=None, sim_name=None):
    return _dir("data", sim_type=sim_type, sim_name=sim_name)


def data_file(kind, sim_type=None, sim_name=None):
    """Path of ../output/data/<sim_type>/<sim_name>/<kind>.npy, e.g. kind='ifgs'."""
    return data_dir(sim_type, sim_name) / f"{kind}.npy"


def sim_dir(category, sim_type=None, sim_name=None):
    return _dir("sims", category, sim_type=sim_type, sim_name=sim_name)


def hit_map_dir(sim_type=None, sim_name=None):
    return _dir("hit_maps", sim_type=sim_type, sim_name=sim_name)


def mapmaker_dir(mapmaker, *parts, sim_type=None, sim_name=None):
    """Output folder of 'binned', 'cg' or 'legacy', optionally with subfolders (e.g. 'maps')."""
    return _dir(mapmaker, *parts, sim_type=sim_type, sim_name=sim_name)


def compare_dir(*parts, sim_type=None, sim_name=None):
    return _dir("compare", *parts, sim_type=sim_type, sim_name=sim_name)
