SIM_TYPE = "firas"  # "fossil" or "firas"

IFG_SIZE = {}
SPEC_SIZE = {}

IFG_SIZE["fossil"] = 256
SPEC_SIZE["fossil"] = 129

IFG_SIZE["firas"] = 512
SPEC_SIZE["firas"] = 257

PNG = True
FITS = True

BEAM = {}
BEAM["fossil"] = 2 # degrees
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

RUN_NAME = "run1"
