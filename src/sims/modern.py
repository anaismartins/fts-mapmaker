"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then telemetered, i.e. we assume that on-board = telemetered IFG.
"""

import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import sims.utils as sims
from sims.scanning_strategy import generate_scanning_strategy

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

user = os.environ["USER"]
data_path = f"/mn/stornext/u3/{user}/d5/firas-reanalysis/Commander/commander3/todscripts/firas/data/sky_v4.4.h5"
sky_data = h5py.File(
    data_path,
    "r",
)

mtm_speed = sky_data["df_data"]["mtm_speed"][:]
mtm_length = sky_data["df_data"]["mtm_length"][:]
ss_filter = (mtm_speed == 0) & (mtm_length == 0)
ecl_lat = sky_data["df_data"]["ecl_lat"][ss_filter]
ecl_lon = sky_data["df_data"]["ecl_lon"][ss_filter]
scan = sky_data["df_data"]["scan"][ss_filter]

npixperifg = 512
pix_ecl = generate_scanning_strategy(ecl_lat, scan, npixperifg)
print(f"Shape of pix_ecl: {pix_ecl.shape} and of spec: {spec.shape}")

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

# now we frankenstein the IFGs together
ifg_scanning = np.zeros((len(pix_ecl), g.IFG_SIZE))
for i in range(npixperifg):
    for pixi, pix in enumerate(pix_ecl[:, i]):
        ifg_scanning[pixi, i] = ifg[pix, i]

print(f"Shape of ifg_scanning: {ifg_scanning.shape}")

n = random.randint(0, ifg_scanning.shape[0])
plt.plot(ifg_scanning[n])
plt.title(f"IFG {n}")
plt.ylabel("Interferogram")
plt.savefig(f"../output/sim_ifgs_modern/{n}.png")
plt.close()

# add white noise
noise, sigma = sims.white_noise(ifg_scanning.shape[0])
ifg_scanning = ifg_scanning + noise

np.savez("../output/ifgs_modern.npz", ifg=ifg_scanning, pix=pix_ecl, sigma=sigma)
print("Saved IFGs")
