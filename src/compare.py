"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
from argparser import args

if args.sim_type == "fossil":
    ref_freq = 540
elif args.sim_type == "firas":
    ref_freq = 544
else:
    raise ValueError("args.sim_type must be 'fossil' or 'firas'")

# plot simulated 545 map
simulated_map = hp.read_map(f"../output/dust_maps/{args.sim_type}/{ref_freq:04d}.fits")
hp.mollview(
    simulated_map,
    title=f"Simulated {ref_freq} GHz map",
    unit="MJy/sr",
    min=0,
    max=50,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/simulated_map.png")
plt.close()


# compare naive mapmaker with maps made from the sed
binned_mapmaker = hp.read_map(f"../output/binned_mapmaker/{args.sim_type}/{ref_freq:04d}.fits")
hp.mollview(
    binned_mapmaker,
    title=f"Binned mapmaker ({args.sim_type})",
    unit="MJy/sr",
    min=0,
    max=50,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig(f"../output/compare/binned_mapmaker_{args.sim_type}.png")
plt.close()
difference_map = simulated_map - binned_mapmaker
hp.mollview(
    difference_map,
    title=f"Simulated - Binned mapmaker ({args.sim_type})",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
    norm="hist",
)
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{args.sim_type}/binned.png")
plt.close()
ratio_map = simulated_map / binned_mapmaker
print("Ratio between simulated map and binned mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(ratio_map, title=f"Ratio map simulated / binned ({args.sim_type})", unit="MJy/sr",
            min=0.5, max=1.5, cmap="RdBu_r", coord=["E", "G"], # norm="hist"
            )
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{args.sim_type}/binned.png")
plt.close()

white_noise_map = hp.read_map(f"../output/white_noise_mapmaker/{args.sim_type}/maps/"
                              f"{ref_freq:04d}.fits")
hp.mollview(white_noise_map, title=f"White noise mapmaker ({args.sim_type})", unit="MJy/sr", min=0,
            max=50, coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/white_noise_mapmaker_{args.sim_type}.png")
plt.close()
difference_map = simulated_map - white_noise_map
hp.mollview(difference_map, title=f"Simulated - White noise mapmaker ({args.sim_type})", unit="MJy/sr",
            min=-1, max=1, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{args.sim_type}/white_noise.png")
plt.close()
ratio_map = simulated_map / white_noise_map
print("Ratio between simulated map and white noise mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(ratio_map, title=f"Simulated / white noise mapmaker ({args.sim_type})", unit="MJy/sr",
            min=0.5, max=1.5, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{args.sim_type}/white_noise.png")
plt.close()

cg_mapmaker = hp.read_map(f"../output/cg_mapmaker/{args.sim_type}/0544.fits")
cg_mapmaker[cg_mapmaker == 0] = np.nan
hp.mollview(cg_mapmaker, title=f"CG mapmaker ({args.sim_type})", unit="MJy/sr", min=0, max=50,
            coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/cg_mapmaker_{args.sim_type}.png")
plt.close()

difference_map = simulated_map - cg_mapmaker
hp.mollview(difference_map, title=f"Simulated - CG mapmaker ({args.sim_type})", unit="MJy/sr",
            min=-1, max=1, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{args.sim_type}/cg.png")
plt.close()
ratio_map = simulated_map / cg_mapmaker
print("Ratio between simulated map and cg mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(ratio_map, title=f"Simulated / CG mapmaker ({args.sim_type})", unit="MJy/sr", min=0.5,
            max=1.5, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{args.sim_type}/cg.png")
plt.close()

# compare white noise mapmaker to cg mapmaker
difference_map = white_noise_map - cg_mapmaker
hp.mollview(difference_map, title=f"White noise - CG mapmaker ({args.sim_type})", unit="MJy/sr",
            min=-1, max=1, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{args.sim_type}/white_noise_cg.png")
plt.close()
ratio_map = white_noise_map / cg_mapmaker
print("Ratio between white noise mapmaker and cg mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(ratio_map, title=f"White noise / CG mapmaker ({args.sim_type})", unit="MJy/sr", min=0.5,
            max=1.5, cmap="RdBu_r", coord=["E", "G"])
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{args.sim_type}/white_noise_cg.png")
plt.close()