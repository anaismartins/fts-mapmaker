"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import setup_matplotlib
from argparser import args

if args.sim_type == "fossil":
    ref_freq = 540
    max = 0.01
elif args.sim_type == "firas":
    ref_freq = 544
    max = 1
else:
    raise ValueError("args.sim_type must be 'fossil' or 'firas'")

# plot tiled difference and ratio maps for each mapmaking method
dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{ref_freq:04d}.fits")
# downgrade dust map to the same resolution as the other maps
dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

binned_map = hp.read_map(f"../output/binned/{args.sim_type}/{ref_freq:04d}.fits")
white_noise_map = hp.read_map(f"../output/white_noise/{args.sim_type}/maps/{ref_freq:04d}.fits")
if args.cg_dummy:
    cg_map = np.zeros_like(dust_map)
else:
    cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{ref_freq:04d}.fits")

difference_binned = binned_map - dust_map
difference_white_noise = white_noise_map - dust_map
difference_cg = cg_map - dust_map

hp.mollview(difference_binned/dust_map, title="Binned", min=-max, max=max, cbar=False,
            coord=["E", "G"], cmap="RdBu_r")
plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_binned.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_binned.png")
plt.close()

hp.mollview(difference_white_noise/dust_map, title="White Noise", min=-max, max=max, cbar=True,
            coord=["E", "G"], cmap="RdBu_r")
plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_white_noise.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_white_noise.png")
plt.close()

hp.mollview(difference_cg/dust_map, title="CG", min=-max, max=max, cbar=False, coord=["E", "G"],
            cmap="RdBu_r")
plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_cg.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_cg.png")
plt.close()

