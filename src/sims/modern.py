"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then telemetered, i.e. we assume that on-board = telemetered IFG.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import sims.scanning_strategy as scanning_strategy
import sims.utils as sims

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

npixperifg = 2
pix_ecl = scanning_strategy.generate_scanning_strategy(ecl_lat, scan, npixperifg)
print(f"Shape of pix_ecl: {pix_ecl.shape} and of spec: {spec.shape}")

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

# now we frankenstein the IFGs together
ifg_scanning = np.zeros((len(pix_ecl[0]), g.IFG_SIZE))
for i in range(npixperifg):
    for pixi, pix in enumerate(pix_ecl[i]):
        ifg_scanning[pixi][
            i * g.IFG_SIZE // npixperifg : (i + 1) * g.IFG_SIZE // npixperifg
        ] = ifg[pix][i * g.IFG_SIZE // npixperifg : (i + 1) * g.IFG_SIZE // npixperifg]

print(f"Shape of ifg_scanning: {ifg_scanning.shape}")

plt.plot(ifg_scanning[::100, :].T)
plt.title("Some example IFGs")
plt.ylabel("Interferogram")
plt.savefig("../output/simulated_ifgs_modern.png")
plt.close()

# add white noise
ifg_scanning = ifg_scanning + sims.white_noise(ifg_scanning.shape[0])
np.savez("../output/ifgs_modern.npz", ifg=ifg_scanning)
print("Saved IFGs")
