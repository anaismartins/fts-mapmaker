IFG_SIZE = 512
SPEC_SIZE = 257

PNG = True
FITS = True

BEAM = {}
BEAM["fossil"] = 2.0 # degrees
BEAM["firas"] = 7.0 # degrees

NSIDE = {}
NSIDE["fossil"] = 128
NSIDE["firas"] = 32

SIM_TYPE = "fossil"  # "fossil" or "firas"

NPIX = 12 * NSIDE[SIM_TYPE]**2
NPIXPERIFG = {}
NPIXPERIFG["fossil"] = 256
NPIXPERIFG["firas"] = 512
N_IFGS = 16

NOISE = "only white"  # "only white" or "1f"

F_NYQ = 112  # Hz (LLSS from FIRAS)
