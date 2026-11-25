"""
Instrument constants that are the same for every simulation version.

Anything that differs between versions (resolution, noise, pointing, beam, ...) is a simulation
parameter and lives in configs/<sim_name>.yaml instead, see config.py.
"""

from pathlib import Path

IFG_SIZE = {"fossil": 256, "firas": 512}
SPEC_SIZE = {"fossil": 129, "firas": 257}

# samples per recorded IFG
NPIXPERIFG = {"fossil": 256, "firas": 512}

# number of on-board IFGs coadded into one telemetered FIRAS IFG
N_IFGS = 16

F_NYQ = 112  # Hz (LLSS from FIRAS)
FIRAS_CHANNELS = {"rh": 0, "rl": 1, "lh": 2, "ll": 3}

# output formats for maps
PNG = True
FITS = True

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
