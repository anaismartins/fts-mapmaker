"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then telemetered, i.e. we assume that on-board = telemetered IFG.
"""

import os
import random
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from time import time

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
# import spiceypy
from astropy.coordinates import get_sun
from astropy.time import Time
from erfa import ErfaWarning

import globals as g
import sims.utils as sims
from sims.scanning_strategy import generate_scanning_strategy

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

# instrument parameters
# survey_len = 4 # years
survey_len = 30 / 12 # years
survey_time = survey_len * 365.25 * 24 * 3600 # seconds

obs_eff = 0.7

# samp_rate = 32.5 # Hz
# one_pointing = 1 / samp_rate # seconds

one_ifg = 3.04 # seconds

# n_ifgs = survey_time * obs_eff // one_ifg
# print(f"Total number of IFGs taken: {n_ifgs}")

one_pointing = one_ifg / g.NPIXPERIFG # seconds
n_total_pointings = int(survey_time * obs_eff // one_pointing)

# speed = 0.3 # deg/min - planck is 1 rpm
# speed = speed / 60 # deg/s
# speed = speed / 360 # rotations per second
speed = 1 # rpm
speed = speed / 60 # rps
spin_rate = 1.00165345964511  # rpm

# spin_axis_tilt = 5 # deg
spin_axis_tilt = 7.5 # deg
spin_axis_tilt_rad = np.deg2rad(spin_axis_tilt)
tilt_cos = np.cos(spin_axis_tilt_rad)
tilt_sin = np.sin(spin_axis_tilt_rad)

los_angle = 87 # deg
# los_angle = 85 # deg
los_angle = np.deg2rad(los_angle)
los_cos = np.cos(los_angle)
los_sin = np.sin(los_angle)

start_date = Time("2041-01-01T00:00:00")
# start_date = datetime(year=2041, month=1, day=1, hour=0, minute=0, second=0)
full_sky = 365.25 / 2 * 24 * 3600 # 6 months in seconds

# overall calculations
ecl_pole_vec = np.vstack([0, 0, 1])

batch_duration = 1
coarse_step_sec = 600 # every 10 minutes

def calculate_batch(batch_idx):
    """
    Get the pointing for one batch.

    Parameters
    ----------
    batch_idx : int
        Index of the batch.
    """
    print(f"  Starting batch {batch_idx + 1}...")
    # Load the SPICE kernels via a meta file (use absolute path for multiprocessing)
    # if spiceypy.ktotal('ALL') == 0:
    #     kernel_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel_meta.txt')
    #     spiceypy.furnsh(kernel_meta)

    # set up times
    # start time is the start date offset by whiever batch idx we are at
    start_day = batch_idx * batch_duration
    start_time_offset = start_day * 24 * 3600 * u.s
    start_time = Time("2041-01-01T00:00:00") + start_time_offset

    batch_duration_sec = batch_duration * 24 * 3600 # seconds

    # time in seconds, but we want to get one pointing per ifg point
    # we need it more fine-grained than seconds
    n_pointings_batch = int(batch_duration_sec / one_pointing)
    t_seconds = np.linspace(0, batch_duration_sec, n_pointings_batch, endpoint=False)
    t_coarse = np.arange(0, batch_duration_sec + coarse_step_sec, coarse_step_sec)  

    t1 = time()
    obs_times = start_time + t_coarse * u.s

    # Convert datetime to list of datetimes by adding timedeltas
    # obs_times = [start_time + timedelta(seconds=float(t)) for t in t_coarse]
    # get sun positions

    sun_coords_coarse = get_sun(obs_times).transform_to("geocentrictrueecliptic")
    anti_sun_lon_coarse = sun_coords_coarse.lon.rad + np.pi  # rad

    # # interpolate
    cos_interp = np.interp(t_seconds, t_coarse, np.cos(anti_sun_lon_coarse))
    sin_interp = np.interp(t_seconds, t_coarse, np.sin(anti_sun_lon_coarse))
    norm = np.sqrt(cos_interp**2 + sin_interp**2)
    anti_sun_lon = np.arctan2(sin_interp / norm, cos_interp / norm)

    # generate basis for anti-sun coordinate system
    anti_sun_cos = np.cos(anti_sun_lon)
    anti_sun_sin = np.sin(anti_sun_lon)
    a_vec = np.vstack([anti_sun_cos, anti_sun_sin, np.zeros_like(anti_sun_lon)]).T
    b_vec = np.vstack([-anti_sun_sin, anti_sun_cos, np.zeros_like(anti_sun_lon)]).T
    c_vec = np.array([0.0, 0.0, 1.0])

    # generate basis spin
    tilt_angle = np.deg2rad(spin_axis_tilt)  # radians
    six_months = 182.625 * 24 * 3600  # seconds
    precession_phase = (
        2 * np.pi * (start_day * 24 * 3600 + t_seconds) / six_months
    )  # radians
    s_vec = np.cos(tilt_angle) * a_vec + np.sin(tilt_angle) * (
        np.cos(precession_phase)[:, np.newaxis] * b_vec
        + np.sin(precession_phase)[:, np.newaxis] * c_vec
    )
    u_vec = np.cross(s_vec, c_vec)
    v_vec = np.cross(s_vec, u_vec)

    # calculate line-of-sight vector in ecliptic coordinates
    spin_phase = (2 * np.pi * spin_rate / 60.0) * (
        start_day * 24 * 3600 + t_seconds
    )  # radians
    spin_cos = np.cos(spin_phase)[:, np.newaxis]
    spin_sin = np.sin(spin_phase)[:, np.newaxis]
    los_vec = los_cos * s_vec + los_sin * (spin_cos * u_vec + spin_sin * v_vec)

    # convert pointing vectors to spherical coordinates
    theta, phi = hp.vec2ang(los_vec)
    pix = hp.ang2pix(g.NSIDE["fossil"], theta, phi)

    # save map for each day
    hit_map = np.bincount(pix, minlength=hp.nside2npix(g.NSIDE["fossil"]))/g.NPIXPERIFG
    print("Saving hit map for day {:03d}...".format(batch_idx + 1))
    hp.write_map("../output/hit_maps/fossil/day_{:03d}.fits".format(batch_idx + 1), hit_map,
                 overwrite=True)

    # plot orbit for each batch
    earth_coords = [0, 0, 0]
    # sun_coords = 

    return pix

# run for full survey  using parallelization
n_batches = int(survey_len * 365.25 * obs_eff) # one day batches
n_workers = min(cpu_count(), n_batches)
print(f"\n{'='*60}")
print(f"Starting parallel processing of {n_batches} batches")
print(f"Using {n_workers} workers (CPU cores available: {cpu_count()})")
print(f"{'='*60}\n")

t_start = time()
with Pool(n_workers) as pool:
    results = pool.map(calculate_batch, range(n_batches))
t_end = time()

print(f"\n{'='*60}")
print(f"Parallel processing complete!")
print(f"Total time: {t_end - t_start:.2f} seconds")
print(f"Average time per batch: {(t_end - t_start)/n_batches:.2f} seconds")
print(f"{'='*60}\n")

# Combine results
print("Combining results from all batches...")
pix_list = zip(*results)
pix = np.concatenate(pix_list)

print(f"Total number of pointings for the whole survey: {len(pix):,}")
print("Creating hit map...")
hit_map = np.bincount(pix, minlength=hp.nside2npix(g.NSIDE["fossil"]))/g.NPIXPERIFG

print("Generating and saving plot...")
hp.mollview(hit_map, title="Hit Map for Fossil Scanning", unit="Hits",coord=["E", "G"])
plt.savefig("../output/hit_maps/scanning_strategy_fossil.png", bbox_inches="tight")
plt.close()
print("Saved hit map to ../output/hit_maps/scanning_strategy_fossil.png")

# save fits file
hp.write_map(
    "../output/hit_maps/scanning_strategy_fossil.fits",
    hit_map,
    overwrite=True,
)

exit()

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

pix_ecl = generate_scanning_strategy(ecl_lat, ecl_lon, scan, npixperifg, modern=True)
print(f"Shape of pix_ecl: {pix_ecl.shape} and of spec: {spec.shape}")

# save map of scanning strategy
hit_map = np.bincount(pix_ecl.flatten(), minlength=hp.nside2npix(g.NSIDE)) / npixperifg
mask = hit_map == 0
hit_map[mask] = np.nan
if g.PNG:
    hp.mollview(
        hit_map,
        title="Hit Map for Fossil Scanning",
        unit="Hits",
        # norm="hist",
        min=np.percentile(hit_map, 1),
        max=np.percentile(hit_map, 99),
    )  # , coord=["E", "G"])
    plt.savefig("../output/hit_maps/scanning_strategy_fossil.png", bbox_inches="tight")
    plt.close()
    print("Saved hit map to ../output/hit_maps/scanning_strategy_planck.png")

    hp.mollview(
        hit_map,
        title="Hit Map for Fossil Scanning",
        unit="Hits",
        # norm="hist",
        min=np.percentile(hit_map, 1),
        max=np.percentile(hit_map, 99),
        coord=["E", "G"],
    )
    plt.savefig(
        "../output/hit_maps/scanning_strategy_fossil_galactic.png", bbox_inches="tight"
    )
    plt.close()
    print("Saved hit map to ../output/hit_maps/scanning_strategy_fossil_galactic.png")

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(np.degrees(theta), label="Theta (deg)", alpha=0.7)
    # ax.legend()
    # ax.set_ylabel("Theta (degrees)")
    # ax2 = ax.twinx()
    # ax2.plot(np.degrees(phi), color="orange", label="Phi (deg)", alpha=0.7)
    # ax2.set_ylabel("Phi (degrees)")
    # ax2.legend()
    # ax.set_xlabel("Sample Index")
    # fig.savefig("../output/sim_pointing_fossil.png", bbox_inches="tight")

    exit()

    dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
    sed = np.nan_to_num(sed)

    spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

    pix_ecl = generate_scanning_strategy(
        ecl_lat, ecl_lon, scan, npixperifg, modern=True
    )
    print(f"Shape of pix_ecl: {pix_ecl.shape} and of spec: {spec.shape}")

    # save map of scanning strategy
    hit_map = (
        np.bincount(pix_ecl.flatten(), minlength=hp.nside2npix(g.NSIDE)) / npixperifg
    )
    mask = hit_map == 0
    hit_map[mask] = np.nan
    if g.PNG:
        hp.mollview(
            hit_map,
            title="Scanning Strategy Hit Map for a Modern Experiment",
            unit="Hits",
            min=0,
            # max=np.nanmax(hit_map),
            max=332,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(
            "../output/hit_maps/scanning_strategy_modern.png", bbox_inches="tight"
        )
        plt.close()
    if g.FITS:
        hp.write_map(
            "../output/hit_maps/scanning_strategy_modern.fits",
            hit_map,
            overwrite=True,
        )

    ifg = np.fft.irfft(spec, axis=1)
    ifg = np.roll(ifg, 360, axis=1)
    ifg = ifg.real

    # check if ifg has units
    print("ifg type:", type(ifg), "unit:", getattr(ifg, "unit", None))

    # now we frankenstein the IFGs together
    ifg_scanning = np.zeros((len(pix_ecl), g.IFG_SIZE))
    for i in range(npixperifg):
        for pix_i, pix in enumerate(pix_ecl[:, i]):
            ifg_scanning[pix_i, i] = ifg[pix, i]

    print(f"Shape of ifg_scanning: {ifg_scanning.shape}")

    n = random.randint(0, ifg_scanning.shape[0])
    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(f"../output/sim_ifgs_modern/{n}.png")
    plt.close()

    # plot pixels hit on a map
    print(f"Pixels hit: {np.unique(pix_ecl[n])}")
    npix = hp.nside2npix(g.NSIDE)
    map_pix = np.bincount(pix_ecl[n], minlength=npix)
    hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="Reds")
    hp.projplot(
        ecl_lon[n],
        ecl_lat[n],
        coord="E",
        color="green",  # "blue",
        lonlat=True,
        marker="x",
    )
    plt.savefig(f"../output/pix_hits/{n}.png")

    # add white noise
    noise, sigma = sims.white_noise(ifg_scanning.shape[0])

    ifg_scanning = ifg_scanning + noise
    print(f"Shape of noise: {noise.shape} and shape of sigma: {sigma.shape}")

    np.savez("../output/ifgs_modern.npz", ifg=ifg_scanning, pix=pix_ecl, sigma=sigma)
    print("Saved IFGs")
