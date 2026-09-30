"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import csv
import multiprocessing
from functools import partial
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import setup_matplotlib
import spectra
import utils
from argparser import args


def overlay_contour(mask, coord=None, colors="yellow", linewidths=1.0):
    """Draw the outline of a binary HEALPix mask on the current healpy projection axes.

    matplotlib's contour needs a 2D array in the projection plane, so the mask is first
    projected with the projector of the axes made by the preceding mollview call. Pass the
    same coord as that mollview call so the rotation matches.
    """
    ax = plt.gca()
    nside = hp.npix2nside(len(mask))
    img = ax.proj.projmap(mask, partial(hp.vec2pix, nside), coord=coord)
    # pixels outside the projection disc come back as -inf; NaN is ignored by contour
    img = np.where(np.isfinite(img), img, np.nan)
    return ax.contour(img, levels=[0.5], extent=ax.proj.get_extent(), origin="lower",
                      colors=colors, linewidths=linewidths)


def sum_chi2(nu_i):
    dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{int(frequencies[nu_i]):04d}.fits")
    dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

    legacy_map = hp.read_map(f"../output/legacy/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")
    binned_map = hp.read_map(f"../output/binned/{args.sim_type}/maps/{int(frequencies[nu_i]):04d}.fits")
    if args.cg_dummy:
        cg_map = np.zeros_like(dust_map)
    else:
        cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{int(frequencies[nu_i]):04d}.fits")

    sq_weight_legacy = ((dust_map - legacy_map)**2)/(hit_map**2)
    sq_weight_binned = ((dust_map - binned_map)**2)/(hit_map**2)
    sq_weight_cg = ((dust_map - cg_map)**2)/(hit_map**2)

    sgn_legacy = dust_map - legacy_map
    sgn_binned = dust_map - binned_map
    sgn_cg = dust_map - cg_map

    return (sq_weight_legacy, sq_weight_binned, sq_weight_cg, sgn_legacy,
            sgn_binned, sgn_cg)


def compute_residual_metrics(frequencies, hit_map):
    observed = np.isfinite(hit_map) & (hit_map > 0) & (hit_map != hp.UNSEEN)
    methods = ("legacy", "binned", "cg")
    rows = []

    for frequency in frequencies:
        freq = int(frequency)
        truth = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{freq:04d}.fits")
        truth = hp.ud_grade(truth, nside_out=g.NSIDE[args.sim_type])
        maps = {"legacy": hp.read_map(f"../output/legacy/{args.sim_type}/{freq:04d}.fits"), "binned": hp.read_map(f"../output/binned/{args.sim_type}/maps/{freq:04d}.fits"), "cg": (np.zeros_like(truth) if args.cg_dummy else hp.read_map(f"../output/cg/{args.sim_type}/{freq:04d}.fits"))}

        common = observed.copy()
        for data in (truth, *maps.values()):
            common &= np.isfinite(data) & (data != hp.UNSEEN)
        n_common = int(np.count_nonzero(common))
        if n_common == 0:
            raise ValueError(f"No common observed pixels for {freq} GHz")

        row = {"frequency_GHz": freq, "n_common_pixels": n_common}
        for method in methods:
            residual = maps[method][common] - truth[common]
            row[f"{method}_bias"] = float(np.mean(residual))
            row[f"{method}_rms"] = float(np.sqrt(np.mean(residual**2)))
            row[f"{method}_centered_rms"] = float(np.std(residual))
        rows.append(row)

    output_path = f"../output/compare/{args.sim_type}_residual_metrics.csv"
    fieldnames = ["frequency_GHz", "n_common_pixels"] + [f"{method}_{metric}" for method in methods for metric in ("bias", "rms", "centered_rms")]
    with open(output_path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-frequency residual metrics to {output_path}.")
    print(f"Common observed pixels per frequency: {min(row['n_common_pixels'] for row in rows)}-{max(row['n_common_pixels'] for row in rows)}.")
    for method in methods:
        biases = np.array([row[f"{method}_bias"] for row in rows])
        rms_values = np.array([row[f"{method}_rms"] for row in rows])
        print(f"{method}: median |bias|={np.median(np.abs(biases)):.6g}, median per-frequency RMS={np.median(rms_values):.6g}, equal-frequency pooled RMS={np.sqrt(np.mean(rms_values**2)):.6g} MJy/sr")

    for method in methods[1:]:
        lower_count = sum(row[f"{method}_rms"] < row["legacy_rms"] for row in rows)
        print(f"{method} has lower RMS than legacy in {lower_count}/{len(rows)} frequencies.")
    plot_residual_metrics(rows)
    return rows

def plot_residual_metrics(rows):
    plt.figure(figsize=(10, 6))
    plt.plot([row["frequency_GHz"] for row in rows], [row["legacy_centered_rms"] for row in rows],
             label="Legacy")
    plt.plot([row["frequency_GHz"] for row in rows], [row["binned_centered_rms"] for row in rows],
             label="Binned")
    plt.plot([row["frequency_GHz"] for row in rows], [row["cg_centered_rms"] for row in rows],
             label="CG", linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Centered RMS (MJy/sr)")
    plt.title("Per-frequency centered RMS of residuals vs. dust simulation")
    plt.legend()
    plt.savefig(f"../output/compare/{args.sim_type}_residual_rms.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot([row["frequency_GHz"] for row in rows], [np.abs(row["legacy_bias"]) for row in rows],
             label="Legacy")
    plt.plot([row["frequency_GHz"] for row in rows], [np.abs(row["binned_bias"]) for row in rows],
             label="Binned")
    plt.plot([row["frequency_GHz"] for row in rows], [np.abs(row["cg_bias"]) for row in rows],
             label="CG")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(r"|Bias| (MJy/sr)")
    plt.title(r"Per-frequency |bias| of residuals vs. dust simulation")
    plt.legend()
    plt.savefig(f"../output/compare/{args.sim_type}_residual_bias.png")
    plt.close()

if __name__ == "__main__":

    with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
            f.write("Profiling output for comparison script\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'starting':<40} | ")

    t00 = _time()
    t0 = _time()

    if args.sim_type == "fossil":
        ref_freq = 540
        plot_limit = 5
    elif args.sim_type == "firas":
        ref_freq = 544
        plot_limit = 2
    else:
        raise ValueError("args.sim_type must be 'fossil' or 'firas'")

    t0 = utils.log_step("load dust map", t0, args.run_name)
    # plot tiled difference and ratio maps for each mapmaking method
    dust_map = hp.read_map(f"../output/sims/{args.sim_type}/dust_maps/{ref_freq:04d}.fits")
    # downgrade dust map to the same resolution as the other maps
    dust_map = hp.ud_grade(dust_map, nside_out=g.NSIDE[args.sim_type])

    # get contour of the galaxy by making a snr cut
    snr_cut = 10
    galaxy_mask = np.zeros_like(dust_map)
    galaxy_mask[dust_map > snr_cut] = 1
    if args.plots == "debug":
        hp.mollview(dust_map, title="Galaxy Contour", min=0, max=50, cbar=False, coord=["E", "G"])
        overlay_contour(galaxy_mask, coord=["E", "G"], colors="red", linewidths=1.5)
        plt.savefig(f"../output/debug/{args.sim_type}_galaxy_contour.png")
        plt.close()
        print(f"Saved galaxy contour to ../output/debug/{args.sim_type}_galaxy_contour.png.")

    legacy_map = hp.read_map(f"../output/legacy/{args.sim_type}/{ref_freq:04d}.fits")
    binned_map = hp.read_map(f"../output/binned/{args.sim_type}/maps/{ref_freq:04d}.fits")
    if args.cg_dummy:
        cg_map = np.zeros_like(dust_map)
    else:
        cg_map = hp.read_map(f"../output/cg/{args.sim_type}/{ref_freq:04d}.fits")

    difference_legacy = legacy_map - dust_map
    difference_binned = binned_map - dust_map
    difference_cg = cg_map - dust_map

    hit_map = hp.read_map(f"../output/hit_maps/scanning_strategy_{args.sim_type}_sim.fits")

    mask = (~np.isfinite(hit_map) | (hit_map <= 0) | (hit_map == hp.UNSEEN))
    mask2 = (binned_map == hp.UNSEEN) | (binned_map == 0) | (np.isnan(binned_map))

    rel_legacy = difference_legacy / dust_map
    rel_binned = difference_binned / dust_map
    rel_cg = difference_cg / dust_map

    rel_legacy = np.sign(rel_legacy) * np.log10(np.abs(rel_legacy))
    rel_binned = np.sign(rel_binned) * np.log10(np.abs(rel_binned))
    rel_cg = np.sign(rel_cg) * np.log10(np.abs(rel_cg))

    for m in (rel_legacy, rel_binned, rel_cg):
        m[mask | mask2] = hp.UNSEEN

    hp.mollview(rel_legacy, title="Sign x Log10(Legacy)", min=-plot_limit, max=plot_limit, #cbar=False,
                coord=["E", "G"], cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_legacy.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_legacy.png")
    plt.close()
    print(f"Saved comparison plots to ../output/compare/{args.sim_type}_legacy.pdf and "
          f"../output/compare/{args.sim_type}_legacy.png.")

    hp.mollview(rel_binned, title="Sign x Log10(Binned)", min=-plot_limit, max=plot_limit, #cbar=False,
                coord=["E", "G"], cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_binned.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_binned.png")
    plt.close()
    print(f"Saved comparison plots to ../output/compare/{args.sim_type}_binned.pdf and "
          f"../output/compare/{args.sim_type}_binned.png.")

    hp.mollview(rel_cg, title="Sign x Log10(CG)", min=-plot_limit, max=plot_limit, #cbar=False,
                coord=["E", "G"], cmap="RdBu_r")
    plt.savefig(f"../output/compare/{args.sim_type}_cg.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_cg.png")
    plt.close()
    print(f"Saved comparison plots to ../output/compare/{args.sim_type}_cg.pdf and "
          f"../output/compare/{args.sim_type}_cg.png.")

    # calculate the chi2 -- we want to sum over frequencies so we need to load all the maps
    t0 = utils.log_step("generate frequencies", t0, args.run_name)
    frequencies = spectra.generate_frequencies(simtype=args.sim_type,
                                               nfreq=g.SPEC_SIZE[args.sim_type])

    t0 = utils.log_step("compute residual metrics", t0, args.run_name)
    compute_residual_metrics(frequencies, hit_map)

    # Source - https://stackoverflow.com/a/9786225
    # Posted by Sven Marnach, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-15, License - CC BY-SA 4.0

    t0 = utils.log_step("sum_chi2 function", t0, args.run_name)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    (sq_weight_legacy, sq_weight_binned, sq_weight_cg, sgn_legacy, sgn_binned,
     sgn_cg) = zip(*pool.map(sum_chi2, range(len(frequencies))))

    t0 = utils.log_step("sum_chi2", t0, args.run_name)
    sq_weight_legacy = np.sum(sq_weight_legacy, axis=0)
    sq_weight_binned = np.sum(sq_weight_binned, axis=0)
    sq_weight_cg = np.sum(sq_weight_cg, axis=0)

    sgn_legacy = np.sum(sgn_legacy, axis=0)
    sgn_binned = np.sum(sgn_binned, axis=0)
    sgn_cg = np.sum(sgn_cg, axis=0)

    chi2_legacy = sgn_legacy * np.log10(sq_weight_legacy)
    chi2_legacy[mask | mask2] = hp.UNSEEN
    chi2_binned = sgn_binned * np.log10(sq_weight_binned)
    chi2_binned[mask | mask2] = hp.UNSEEN
    chi2_cg = sgn_cg * np.log10(sq_weight_cg)
    chi2_cg[mask | mask2] = hp.UNSEEN

    if args.sim_type == "fossil":
        max_chi2 = 5
    elif args.sim_type == "firas":
        max_chi2 = 1000

    hp.mollview(chi2_legacy, title="Sign x Log10(Legacy)", min=-max_chi2, max=max_chi2,
                cbar=False,
                coord="E", cmap="RdBu_r")
    overlay_contour(galaxy_mask, coord="E")
    plt.savefig(f"../output/compare/{args.sim_type}_legacy_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_legacy_chi2.png")
    plt.close()
    print(f"Saved legacy chi2 plots to ../output/compare/{args.sim_type}_legacy_chi2.pdf and "
              f"../output/compare/{args.sim_type}_legacy_chi2.png.")

    hp.mollview(chi2_binned, title="Sign x Log10(Binned)", min=-max_chi2, max=max_chi2, cbar=False,
                coord="E", cmap="RdBu_r")
    overlay_contour(galaxy_mask, coord="E")
    plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_binned_chi2.png")
    plt.close()
    print(f"Saved binned chi2 plots to ../output/compare/{args.sim_type}_binned_chi2.pdf and "
              f"../output/compare/{args.sim_type}_binned_chi2.png.")

    hp.mollview(chi2_cg, title="Sign x Log10(CG)", min=-max_chi2, max=max_chi2, cbar=False, coord="E",
                cmap="RdBu_r")
    overlay_contour(galaxy_mask, coord="E")
    plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.pdf")
    plt.savefig(f"../output/compare/{args.sim_type}_cg_chi2.png")
    plt.close()
    print(f"Saved cg chi2 plots to ../output/compare/{args.sim_type}_cg_chi2.pdf and "
              f"../output/compare/{args.sim_type}_cg_chi2.png.")

    with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
        f.write(f"{(_time() - t0):.2f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total time for comparison: {(_time() - t00)/60:.2f} min\n")