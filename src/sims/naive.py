import os
import sys
import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import sims.utils as sims
from globals import IFG_SIZE, SPEC_SIZE

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import globals as g

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

print("Calculating and plotting IFGs")

# time ifg making
time_start = time.time()

ifg = np.fft.irfft(spec, axis=1)

# add phase to ifg
ifg = np.roll(ifg, 360, axis=1)
# turn ifg into real signal
ifg = ifg.real

# introduce scanning strategy
# pix_gal = np.load("../input/firas_scanning_strategy.npy").astype(int)
# pix_ecl = np.load("../input/firas_scanning_strategy.npy").astype(int)
print(f"Shape of pix_ecl: {pix_ecl.shape}")

ifg_scanning = np.zeros((len(pix_ecl), IFG_SIZE))
spec_scanning = np.zeros((len(pix_ecl), SPEC_SIZE))
hit_scanning = np.zeros((len(pix_ecl)), dtype=int)
print("Calculating IFGs for scanning strategy")
for pixi, pix in enumerate(pix_ecl):
    ifg_scanning[pixi] = ifg[pix]
    spec_scanning[pixi] = spec[pix]
    hit_scanning[pixi] = 1

# add noise to ifg
noise = sims.white_noise(ifg_scanning.shape[0] // 3)
# adds same white noise to each of 3 corresponding to one original IFG
ifg_scanning = ifg_scanning + np.repeat(noise, 3, axis=0)[: ifg_scanning.shape[0], :]

# add white noise to spec to see how it looks in the difference/ratio
spec_scanning = spec_scanning + sims.white_noise(spec_scanning.shape[0], ifg=False)

# bin spec_scanning into spec maps
spec_map = np.zeros((g.NPIX, g.SPEC_SIZE))
dd = np.zeros(g.NPIX)
for pixi, pix in enumerate(pix_ecl):
    spec_map[pix] += spec_scanning[pixi]
    dd[pix] += 1

mask = dd == 0
spec_map[~mask] = spec_map[~mask] / dd[~mask][:, np.newaxis]
spec_map[mask] = np.nan

for i, freq in enumerate(frequencies):
    if g.PNG:
        hp.mollview(
            spec_map[:, i],
            title=f"{int(freq):04d} GHz",
            unit="MJy/sr",
            min=0,
            max=50,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(f"../output/sim_maps/{int(freq):04d}.png")
        plt.close()
    if g.FITS:
        hp.write_map(
            f"../output/sim_maps/{int(freq):04d}.fits",
            spec_map[:, i],
            overwrite=True,
            dtype=np.float64,
        )

# plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 1")
# plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 2")
# plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 3")
# plt.xlabel("Spacing")
# plt.ylabel("Signal (mJy)")
# plt.title("Interferogram")
# plt.legend()
# plt.show()
# plt.savefig("../output/interferogram.png")
# plt.close()

print("Saving IFGs to file")

# save ifg products in a npz file
np.savez("../output/ifgs_naive.npz", ifg=ifg_scanning)

time_end = time.time()
print(f"Time elapsed for IFGs: {(time_end - time_start)/60} minutes")
