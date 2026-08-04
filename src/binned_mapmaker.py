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

ifgs = np.load(f"../output/data/{args.sim_type}/ifgs.npy", mmap_mode="r")
t0 = utils.log_step("load_ifgs", t0, args.run_name)
ecl_lon = np.load(f"../output/data/{args.sim_type}/ecl_lon.npy", mmap_mode="r")
ecl_lat = np.load(f"../output/data/{args.sim_type}/ecl_lat.npy", mmap_mode="r")
t0 = utils.log_step("load_pointing", t0, args.run_name)

if args.sim_type == "firas":
    ifgs = ifgs / g.N_IFGS

pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True)
t0 = utils.log_step("ang2pix", t0, args.run_name)
# use only the middle pixel
if args.sim_type == "fossil":
    pix = pix[:, g.NPIXPERIFG[args.sim_type] // 2]
elif args.sim_type == "firas":
    pix = pix[:, g.NPIXPERIFG[args.sim_type] // 2, g.N_IFGS // 2]
else:
    raise ValueError("args.sim_type must be 'fossil' or 'firas'")
t0 = utils.log_step("select_middle_pixel", t0, args.run_name)

# plot hit map of the scanning strategy
npix = g.NPIX[args.sim_type]
hit_map = np.bincount(pix, minlength=npix).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
t0 = utils.log_step("create_hit_map", t0, args.run_name)
if g.PNG:
    hp.mollview(hit_map, title="Scanning strategy hit map",
                unit="Number of hits over the full mission", min=0, max=hit_map.max(), xsize=2000,
                coord=["E", "G"])
    plt.savefig(f"../output/hit_maps/binned_{args.sim_type}.png")
    plt.close()

    print(f"Saved hit map of the scanning strategy to ../output/hit_maps/binned_{args.sim_type}.png.")
t0 = utils.log_step("plot_hit_map", t0, args.run_name)


pix = pix.astype(np.int64, copy=False)

m_ifg = np.zeros((npix, g.IFG_SIZE[args.sim_type]), dtype=float)

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

frequencies = spectra.generate_frequencies(nfreq=g.SPEC_SIZE[args.sim_type], simtype=args.sim_type)
t0 = utils.log_step("bb_addition", t0, args.run_name)

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(f"../output/binned/{args.sim_type}/{int(frequencies[nui]):04d}.fits",
                     m_abs[:, nui], overwrite=True, dtype=np.float64)
    if g.PNG:
        hp.mollview(m_abs[:, nui], title=f"{int(frequencies[nui]):04d} GHz", unit="MJy/sr",
            min=0, max=50,
            # norm='hist',
            xsize=2000, coord=["E", "G"])
        plt.savefig(f"../output/binned/{args.sim_type}/{int(frequencies[nui]):04d}.png")
        plt.close()
        plt.clf()

if g.PNG:
    print(f"Saved maps to ../output/binned/{args.sim_type}/.")
    t0 = utils.log_step("save_maps", t0, args.run_name)

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for binned mapmaker: {(_time() - t00)/60:.2f} min\n")