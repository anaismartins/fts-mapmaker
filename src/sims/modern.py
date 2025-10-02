"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then telemetered, i.e. we assume that on-board = telemetered IFG.
"""

import numpy as np

import sims.utils as sims

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

pix_ecl = np.load("../input/firas_scanning_strategy.npy").astype(int)

print(f"Shape of pix_ecl: {pix_ecl.shape} and of spec: {spec.shape}")
