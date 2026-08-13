"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import setup_matplotlib
import spectra
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
noise_weighted_map = hp.read_map(f"../output/noise_weighted/{args.sim_type}/maps/{ref_freq:04d}.fits")
if args.cg_dummy:
    cg_map = np.zeros_like(dust_map)
else:
    cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{ref_freq:04d}.fits")

difference_binned = binned_map - dust_map
difference_noise_weighted = noise_weighted_map - dust_map
difference_cg = cg_map - dust_map

hp.mollview(difference_binned/dust_map, title="Binned", min=-max, max=max, cbar=False,
            coord=["E", "G"], cmap="RdBu_r")
# plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_binned.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_binned.png")
plt.close()

hp.mollview(difference_noise_weighted/dust_map, title="Noise Weighted", min=-max, max=max, cbar=True,
            coord=["E", "G"], cmap="RdBu_r")
# plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted.png")
plt.close()

hp.mollview(difference_cg/dust_map, title="CG", min=-max, max=max, cbar=False, coord=["E", "G"],
            cmap="RdBu_r")
# plt.tight_layout()
plt.savefig(f"../output/compare/{args.sim_type}_cg.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_cg.png")
plt.close()

# calculate the chi2 -- we want to sum over frequencies so we need to load all the maps
hit_map = hp.read_map(f"../output//hit_maps/scanning_strategy_{args.sim_type}_sim.fits")
frequencies = spectra.generate_frequencies(simtype=args.sim_type, nfreq=g.IFG_SIZE[args.sim_type])

sq_weight_binned = np.zeros((g.NPIX[args.sim_type]), dtype=float)
sq_weight_noise_weighted = np.zeros((g.NPIX[args.sim_type]), dtype=float)
sq_weight_cg = np.zeros((g.NPIX[args.sim_type]), dtype=float)
sgn_binned = np.zeros((g.NPIX[args.sim_type]), dtype=float)
sgn_noise_weighted = np.zeros((g.NPIX[args.sim_type]), dtype=float)
sgn_cg = np.zeros((g.NPIX[args.sim_type]), dtype=float)

for nu_i in range(g.IFG_SIZE[args.sim_type]):
    dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{int(frequencies[nu_i]):04d}.fits")
    dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

    binned_map = hp.read_map(f"../output/binned/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")
    noise_weighted_map = hp.read_map(f"../output/noise_weighted/{args.sim_type}/maps/{int(frequencies[nu_i]):04d}.fits")
    if args.cg_dummy:
        cg_map = np.zeros_like(dust_map)
    else:
        cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")

    sq_weight_binned += ((dust_map - binned_map)**2)/(hit_map**2)
    sq_weight_noise_weighted += ((dust_map - noise_weighted_map)**2)/(hit_map**2)
    sq_weight_cg += ((dust_map - cg_map)**2)/(hit_map**2)

    sgn_binned += dust_map - binned_map
    sgn_noise_weighted += dust_map - noise_weighted_map
    sgn_cg += dust_map - cg_map

sgn_binned = np.sgn(sgn_binned)
sgn_noise_weighted = np.sgn(sgn_noise_weighted)
sgn_cg = np.sgn(sgn_cg)

chi2_binned = sgn_binned * sq_weight_binned
chi2_noise_weighted = sgn_noise_weighted * sq_weight_noise_weighted
chi2_cg = sgn_cg * sq_weight_cg

hp.mollview(chi2_binned, title="Binned", min=-max, max=max, cbar=False,
            coord=["E", "G"], cmap="RdBu_r")
plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.png")
plt.close()

hp.mollview(chi2_noise_weighted, title="Noise Weighted", min=-max, max=max, cbar=True,
            coord=["E", "G"], cmap="RdBu_r")
plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted_chi2.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted_chi2.png")
plt.close() 

hp.mollview(chi2_cg, title="CG", min=-max, max=max, cbar=False, coord=["E", "G"],
            cmap="RdBu_r")
plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.pdf")
plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.png")
plt.close()