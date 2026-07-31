from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import spectra
import utils
from argparser import args

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for binned mapmaker for FOSSIL\n")
    f.write("=" * 50 + "\n")

t00 = _time()
t0 = _time()

ifgs = np.load(f"../output/data/{g.SIM_TYPE}/ifgs.npy")
t0 = utils.log_step("load_ifgs", t0, args.run_name)
pix = np.load(f"../output/data/{g.SIM_TYPE}/pointing.npy")
t0 = utils.log_step("load_pointing", t0, args.run_name)

if g.SIM_TYPE == "firas":
    ifgs = ifgs / g.N_IFGS

# use only the middle pixel
if g.SIM_TYPE == "fossil":
    pix = pix[:, g.NPIXPERIFG[g.SIM_TYPE] // 2]
elif g.SIM_TYPE == "firas":
    pix = pix[:, g.NPIXPERIFG[g.SIM_TYPE] // 2, g.N_IFGS // 2]
else:
    raise ValueError("g.SIM_TYPE must be 'fossil' or 'firas'")
t0 = utils.log_step("select_middle_pixel", t0, args.run_name)

# plot hit map of the scanning strategy
npix = g.NPIX[g.SIM_TYPE]
hit_map = np.bincount(pix, minlength=npix).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
t0 = utils.log_step("create_hit_map", t0, args.run_name)
if g.PNG:
    hp.mollview(hit_map, title="Scanning strategy hit map",
                unit="Number of hits over the full mission", min=0, max=hit_map.max(), xsize=2000,
                coord=["E", "G"])
    plt.savefig("../output/hit_maps/binned.png")
    plt.close()

    print("Saved hit map of the scanning strategy to ../output/hit_maps/binned.png.")
t0 = utils.log_step("plot_hit_map", t0, args.run_name)


pix = pix.astype(np.int64, copy=False)

m_ifg = np.zeros((npix, g.IFG_SIZE[g.SIM_TYPE]), dtype=float)

# Vectorized accumulation is much faster than looping in Python.
np.add.at(m_ifg, pix, ifgs)

mask = hit_map == 0
np.divide(m_ifg, hit_map[:, np.newaxis], out=m_ifg, where=~mask[:, np.newaxis])
t0 = utils.log_step("divide_by_hit_map", t0, args.run_name)
m_ifg[mask] = np.nan
t0 = utils.log_step("set empty to nan", t0, args.run_name)

m = np.fft.rfft(m_ifg, axis=1)
t0 = utils.log_step("rfft", t0, args.run_name)
m_abs = m.real

frequencies = spectra.generate_frequencies(nfreq=g.SPEC_SIZE[g.SIM_TYPE], simtype=g.SIM_TYPE)
t0 = utils.log_step("bb_addition", t0, args.run_name)

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(f"../output/binned/{g.SIM_TYPE}/{int(frequencies[nui]):04d}.fits",
                     m_abs[:, nui], overwrite=True, dtype=np.float64)
    if g.PNG:
        hp.mollview(m_abs[:, nui], title=f"{int(frequencies[nui]):04d} GHz", unit="MJy/sr",
            # min=0, max=50,
            norm='hist',
            xsize=2000, coord=["E", "G"])
        plt.savefig(f"../output/binned/{g.SIM_TYPE}/{int(frequencies[nui]):04d}.png")
        plt.close()
        plt.clf()

if g.PNG:
    print(f"Saved maps to ../output/binned/{g.SIM_TYPE}/.")
    t0 = utils.log_step("save_maps", t0, args.run_name)

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for binned mapmaker: {(_time() - t00)/60:.2f} min\n")