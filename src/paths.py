"""
Single place that defines where outputs live.

A simulation version is identified by its name (see config.py), and every output derived from it
sits in a folder named after it:

    output/data/<sim_type>/<sim_name>/              ifgs.npy, noise.npy, config.yaml (parameters)
    output/data/<sim_type>/pointing/<pointing>/     ecl_lon, ecl_lat and pixel-index caches; shared
                                                    by all versions with the same pointing
    output/sims/<sim_type>/<sim_name>/<category>/   dust_maps (truth maps), ifgs, pix_hits
    output/hit_maps/<sim_type>/<sim_name>/
    output/<binned|cg|legacy>/<sim_type>/<sim_name>/   mapmaker outputs, with run_info.yaml
    output/compare/<sim_type>/<sim_name>/
    output/profiling/<sim_type>/<sim_name>/<script>.txt

By default sim_type, sim_name and pointing are those of the current run (config.cfg).

Pixel-index caches (pointing/<pointing>/pix_nside<N>.npy) always have the same layout as
ecl_lon/ecl_lat: one row per recorded IFG.
"""

import globals as g
from config import cfg

OUTPUT = g.OUTPUT_DIR


def _dir(root, *parts, sim_type=None, sim_name=None):
    path = OUTPUT / root / (sim_type or cfg.sim_type) / (sim_name or cfg.sim_name)
    path = path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(sim_type=None, sim_name=None):
    return _dir("data", sim_type=sim_type, sim_name=sim_name)


def data_file(kind, sim_type=None, sim_name=None):
    """Path of output/data/<sim_type>/<sim_name>/<kind>.npy, e.g. kind='ifgs'."""
    return data_dir(sim_type, sim_name) / f"{kind}.npy"


def pointing_file(kind: str, sim_type: str = None, pointing: str = None):
    """Path of output/data/<sim_type>/pointing/<pointing>/<kind>.npy, e.g. kind='ecl_lon'."""
    path = OUTPUT / "data" / (sim_type or cfg.sim_type) / "pointing" / (pointing or cfg.pointing)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{kind}.npy"


def sim_dir(category, sim_type=None, sim_name=None):
    return _dir("sims", category, sim_type=sim_type, sim_name=sim_name)


def hit_map_dir(sim_type=None, sim_name=None):
    return _dir("hit_maps", sim_type=sim_type, sim_name=sim_name)


def mapmaker_dir(mapmaker, *parts, sim_type=None, sim_name=None):
    """Output folder of 'binned', 'cg' or 'legacy', optionally with subfolders (e.g. 'maps')."""
    return _dir(mapmaker, *parts, sim_type=sim_type, sim_name=sim_name)


def compare_dir(*parts, sim_type=None, sim_name=None):
    return _dir("compare", *parts, sim_type=sim_type, sim_name=sim_name)
