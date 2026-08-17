"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import multiprocessing
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import setup_matplotlib
import spectra
import utils
from argparser import args


def sum_chi2(nu_i):
    dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{int(frequencies[nu_i]):04d}.fits")
    dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

    binned_map = hp.read_map(f"../output/binned/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")
    noise_weighted_map = hp.read_map(f"../output/noise_weighted/{args.sim_type}/maps/{int(frequencies[nu_i]):04d}.fits")
    if args.cg_dummy:
        cg_map = np.zeros_like(dust_map)
    else:
        cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")

    sq_weight_binned = ((dust_map - binned_map)**2)/(hit_map**2)
    sq_weight_noise_weighted = ((dust_map - noise_weighted_map)**2)/(hit_map**2)
    sq_weight_cg = ((dust_map - cg_map)**2)/(hit_map**2)

    sgn_binned = dust_map - binned_map
    sgn_noise_weighted = dust_map - noise_weighted_map
    sgn_cg = dust_map - cg_map

    return (sq_weight_binned, sq_weight_noise_weighted, sq_weight_cg, sgn_binned,
            sgn_noise_weighted, sgn_cg)

if __name__ == "__main__":

    with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
            f.write("Profiling output for comparison script\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'starting':<40} | ")

    t00 = _time()
    t0 = _time()

    if args.sim_type == "fossil":
        ref_freq = 540
        max = 0.01
    elif args.sim_type == "firas":
        ref_freq = 544
        max = 1
    else:
        raise ValueError("args.sim_type must be 'fossil' or 'firas'")

    t0 = utils.log_step("load dust map", t0, args.run_name)
    # plot tiled difference and ratio maps for each mapmaking method
    dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{ref_freq:04d}.fits")
    # downgrade dust map to the same resolution as the other maps
    dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

    # get contour of the galaxy by making a snr cut
    snr_cut = 25

    binned_map = hp.read_map(f"../output/binned/{args.sim_type}/{ref_freq:04d}.fits")
    noise_weighted_map = hp.read_map(f"../output/noise_weighted/{args.sim_type}/maps/{ref_freq:04d}.fits")
    if args.cg_dummy:
        cg_map = np.zeros_like(dust_map)
    else:
        cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{ref_freq:04d}.fits")

    difference_binned = binned_map - dust_map
    difference_noise_weighted = noise_weighted_map - dust_map
    difference_cg = cg_map - dust_map

    hit_map = hp.read_map(f"../output/hit_maps/scanning_strategy_{args.sim_type}_sim.fits")

    mask = (~np.isfinite(hit_map) | (hit_map <= 0) | (hit_map == hp.UNSEEN))

    difference_binned[mask] = hp.UNSEEN
    difference_noise_weighted[mask] = hp.UNSEEN
    difference_cg[mask] = hp.UNSEEN

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
    t0 = utils.log_step("generate frequencies", t0, args.run_name)
    frequencies = spectra.generate_frequencies(simtype=args.sim_type, nfreq=g.SPEC_SIZE[args.sim_type])
    print(f"Generated {len(frequencies)} frequencies from {frequencies[0]} to {frequencies[-1]} GHz.")

    # Source - https://stackoverflow.com/a/9786225
    # Posted by Sven Marnach, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-15, License - CC BY-SA 4.0

    t0 = utils.log_step("sum_chi2 function", t0, args.run_name)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    (sq_weight_binned, sq_weight_noise_weighted, sq_weight_cg, sgn_binned, sgn_noise_weighted,
     sgn_cg) = zip(*pool.map(sum_chi2, range(len(frequencies))))

    t0 = utils.log_step("sum_chi2", t0, args.run_name)
    sq_weight_binned = np.sum(sq_weight_binned, axis=0)
    sq_weight_noise_weighted = np.sum(sq_weight_noise_weighted, axis=0)
    sq_weight_cg = np.sum(sq_weight_cg, axis=0)

    sgn_binned = np.sum(sgn_binned, axis=0)
    sgn_noise_weighted = np.sum(sgn_noise_weighted, axis=0)
    sgn_cg = np.sum(sgn_cg, axis=0)

    chi2_binned = sgn_binned * np.log10(sq_weight_binned)
    chi2_binned[mask] = hp.UNSEEN
    chi2_noise_weighted = sgn_noise_weighted * np.log10(sq_weight_noise_weighted)
    chi2_cg = sgn_cg * np.log10(sq_weight_cg)

    if args.sim_type == "fossil":
        max_chi2 = -5
    elif args.sim_type == "firas":
        max_chi2 = 150

    hp.mollview(chi2_binned, title="Binned", min=-max_chi2, max=max_chi2, cbar=False,
                coord="E", cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.png")
    plt.close()

    hp.mollview(chi2_noise_weighted, title="Noise Weighted", min=-max_chi2, max=max_chi2, cbar=True,
                coord="E", cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_noise_weighted_chi2.png")
    plt.close() 

    hp.mollview(chi2_cg, title="CG", min=-max_chi2, max=max_chi2, cbar=False, coord="E",
                cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.png")
    plt.close()

    with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
        f.write(f"{(_time() - t0):.2f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total time for comparison: {(_time() - t00)/60:.2f} min\n")