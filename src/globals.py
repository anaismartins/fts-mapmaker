from pathlib import Path

SIM_TYPE = "fossil"  # "fossil" or "firas"

IFG_SIZE = {}
SPEC_SIZE = {}

IFG_SIZE["fossil"] = 256
SPEC_SIZE["fossil"] = 129

IFG_SIZE["firas"] = 512
SPEC_SIZE["firas"] = 257

PNG = True
FITS = True

BEAM = {}
BEAM["fossil"] = 1.6 # degrees
BEAM["firas"] = 7 # degrees

NSIDE = {}
NSIDE["fossil"] = 128
NSIDE["firas"] = 32

NPIX = {}
NPIX["fossil"] = 12 * NSIDE["fossil"]**2
NPIX["firas"] = 12 * NSIDE["firas"]**2

NPIXPERIFG = {}
NPIXPERIFG["fossil"] = 256
NPIXPERIFG["firas"] = 512

N_IFGS = 16

NOISE = "only white"  # "only white" or "1f"

F_NYQ = 112  # Hz (LLSS from FIRAS)
FIRAS_CHANNELS = {"rh": 0, "rl": 1, "lh": 2, "ll": 3}

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
DUST_MAP_DIR = OUTPUT_DIR / "sims" / SIM_TYPE / "dust_maps"
IFG_DIR = OUTPUT_DIR / "sims" / SIM_TYPE / "ifgs"
PIX_HIT_DIR = OUTPUT_DIR / "sims" / SIM_TYPE / "pix_hits"
DATA_DIR = OUTPUT_DIR / "data" / SIM_TYPE