from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import spectra
import utils
from argparser import args

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for legacy mapmaker\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'load ifgs':<40} | ")

t00 = _time()
t0 = _time()

ifgs = np.load(f"../output/data/{args.sim_type}/ifgs.npy", mmap_mode="r")
t0 = utils.log_step("load_pointing", t0, args.run_name)
ecl_lon = np.load(f"../output/data/{args.sim_type}/ecl_lon.npy", mmap_mode="r")
ecl_lat = np.load(f"../output/data/{args.sim_type}/ecl_lat.npy", mmap_mode="r")

if args.sim_type == "firas":
    ifgs = ifgs / g.N_IFGS

t0 = utils.log_step("ang2pix", t0, args.run_name)

# use only the middle pixel
t0 = utils.log_step("select_middle_pixel", t0, args.run_name)
mid_pix = g.NPIXPERIFG[args.sim_type] // 2
if args.sim_type == "fossil":
    ecl_lon_mid = ecl_lon[:, mid_pix]
    ecl_lat_mid = ecl_lat[:, mid_pix]
    pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon_mid, ecl_lat_mid, lonlat=True)
elif args.sim_type == "firas":
    mid_ifg = g.N_IFGS // 2
    ecl_lon_mid = ecl_lon[:, mid_pix, mid_ifg]
    ecl_lat_mid = ecl_lat[:, mid_pix, mid_ifg]
    pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon_mid, ecl_lat_mid, lonlat=True)
else:
    raise ValueError("args.sim_type must be 'fossil' or 'firas'")

# plot hit map of the scanning strategy
t0 = utils.log_step("create_hit_map", t0, args.run_name)
npix = g.NPIX[args.sim_type]
hit_map = np.bincount(pix, minlength=npix).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
if g.PNG:
    t0 = utils.log_step("plot_hit_map", t0, args.run_name)
    hp.mollview(hit_map, title="Scanning strategy hit map",
                unit="Number of hits over the full mission", min=0, max=hit_map.max(), xsize=2000,
                coord=["E", "G"])
    t0 = utils.log_step("save_hit_map", t0, args.run_name)
    plt.savefig(f"../output/hit_maps/legacy_{args.sim_type}.png")
    plt.savefig(f"../output/hit_maps/legacy_{args.sim_type}.pdf")
    plt.close()

    print(f"Saved hit map of the scanning strategy to ../output/hit_maps/legacy_{args.sim_type}.png.")

pix = pix.astype(np.int64, copy=False)

m_ifg = np.zeros((npix, g.IFG_SIZE[args.sim_type]), dtype=float)

np.add.at(m_ifg, pix, ifgs)

mask = hit_map == 0
t0 = utils.log_step("divide_by_hit_map", t0, args.run_name)
np.divide(m_ifg, hit_map[:, np.newaxis], out=m_ifg, where=~mask[:, np.newaxis])

t0 = utils.log_step("set empty to nan", t0, args.run_name)
m_ifg[mask] = np.nan

t0 = utils.log_step("rfft", t0, args.run_name)
m = np.fft.rfft(m_ifg, axis=1).real

frequencies = spectra.generate_frequencies(nfreq=g.SPEC_SIZE[args.sim_type], simtype=args.sim_type)

# save m as maps
t0 = utils.log_step("save_maps", t0, args.run_name)
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(f"../output/legacy/{args.sim_type}/{int(frequencies[nui]):04d}.fits",
                     m[:, nui], overwrite=True, dtype=np.float64)
    if g.PNG:
        hp.mollview(m[:, nui], title=f"{int(frequencies[nui]):04d} GHz", unit="MJy/sr",
            min=0, max=50, xsize=2000, coord=["E", "G"])
        plt.savefig(f"../output/legacy/{args.sim_type}/{int(frequencies[nui]):04d}.png")
        plt.close()
        plt.clf()

print(f"Saved maps to ../output/legacy/{args.sim_type}/.")
    
with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write(f"{(_time() - t0):0.2f}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total time for legacy mapmaker: {(_time() - t00)/60:.2f} min\n")