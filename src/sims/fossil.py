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

samp_rate = 32.5 # Hz
one_pointing = 1 / samp_rate # seconds

# one_ifg = 3.04 # seconds

# n_ifgs = survey_time * obs_eff // one_ifg
# print(f"Total number of IFGs taken: {n_ifgs}")

# one_pointing = one_ifg / g.NPIXPERIFG # seconds
# n_total_pointings = survey_time * obs_eff // one_pointing

# speed = 0.3 # deg/min - planck is 1 rpm
# speed = speed / 60 # deg/s
# speed = speed / 360 # rotations per second
speed = 1 # rpm
speed = speed / 60 # rps

# spin_axis_tilt = 5 # deg
spin_axis_tilt = 7.5 # deg
spin_axis_tilt = np.deg2rad(spin_axis_tilt)
tilt_cos = np.cos(spin_axis_tilt)
tilt_sin = np.sin(spin_axis_tilt)

# los_angle = 87 # deg
los_angle = 85 # deg
los_angle = np.deg2rad(los_angle)
los_cos = np.cos(los_angle)
los_sin = np.sin(los_angle)

start_date = Time("2041-01-01T00:00:00")
# start_date = datetime(year=2041, month=1, day=1, hour=0, minute=0, second=0)
full_sky = 365.25 / 2 * 24 * 3600 # 6 months in seconds

# overall calculations
ecl_pole_vec = np.vstack([0, 0, 1])

batch_duration = 1

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
    start_time = Time('2041-01-01T00:00:00') + start_time_offset

    batch_duration_sec = batch_duration * 24 * 3600 # seconds
    n_samples = (batch_duration_sec * samp_rate)

    # time in seconds, but we want to get one pointing per ifg point
    # we need it more fine-grained than seconds
    t_seconds = np.linspace(0, batch_duration_sec, one_pointing)
    t_coarse = np.arange(0, batch_size, 600) # every 10 minutes

    t1 = time()
    obs_times = start_time + t_coarse * u.s
    # Convert datetime to list of datetimes by adding timedeltas
    # obs_times = [start_time + timedelta(seconds=float(t)) for t in t_coarse]
    # get sun positions
    sun_coords = get_sun(obs_times).transform_to('geocentrictrueecliptic')
    anti_sun_lon = sun_coords.lon.rad + np.pi # rad

    # # interpolate
    cos_interp = np.interp(t_seconds, t_coarse, np.cos(anti_sun_lon))
    sin_interp = np.interp(t_seconds, t_coarse, np.sin(anti_sun_lon))
    norm = np.sqrt(cos_interp**2 + sin_interp**2)
    anti_sun_lon = np.arctan2(sin_interp / norm, cos_interp / norm)

    # get L2 ephemeris
    # solsys_dict = {'SSB': 0, 'SUN': 10, 'EARTH': 399, 'L2': 392}

    # start_date = datetime(year=2041, month=1, day=1, hour=0, minute=0, second=0)
    # start_time_offset = timedelta(seconds=batch_idx * batch_size)
    # # start_time = start_date + start_time_offset
    # # end_time = start_time + timedelta(seconds=batch_size)
    
    # # start_time_UTC_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
    # start_time_UTC_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
    # # end_time_UTC_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

    # start_time_ET = spiceypy.str2et(start_time_UTC_str)
    # end_time_ET = spiceypy.str2et(end_time_UTC_str)

    # time_interval_et = np.arange(start_time_ET, end_time_ET, 600)

    # posvec_ecl, ltt = spiceypy.spkezp(targ=solsys_dict['L2'], et=time_interval_et, ref='ECLIPJ2000',
    #                                   abcorr='LT',obs=solsys_dict['EARTH'])
    # # check these results
    # posvec_ecl = np.array(posvec_ecl) # shape (N, 3)
    # print(f"    Calculated L2 positions for {len(time_interval_et)} time points.")
    # print(posvec_ecl)

    # if start_time_ET is not None:
    #     return 

    # set up coordinate basis in ecliptic polar coordinates
    anti_sun_cos = np.cos(anti_sun_lon)
    anti_sun_sin = np.sin(anti_sun_lon)

    n_points = len(anti_sun_lon)
    anti_sun_vec = np.vstack([anti_sun_cos, anti_sun_sin, np.zeros(n_points)])
    anti_sun_perp_vec = np.vstack([-anti_sun_sin, anti_sun_cos, np.zeros(n_points)])
    # print(f"Calculated basis vectors with shape: {anti_sun_vec.shape}")

    # use basis vectors to calculate precession pattern
    # generate time arrays again to not use astropy units
    t = np.arange(0, batch_size, one_pointing)
    times = batch_idx * batch_size + t

    precession_phase = 2 * np.pi * times / full_sky
    spin_phase = 2 * np.pi * speed * times

    precession_cos = np.cos(precession_phase)
    precession_sin = np.sin(precession_phase)
    spin_vec = (tilt_cos * anti_sun_vec +
                tilt_sin * (precession_cos * anti_sun_perp_vec + precession_sin * ecl_pole_vec))
    # print(f"Calculated spin_vec with shape: {spin_vec.shape}")

    # generate new basis around the spin vector
    spin_perp_vec1 = np.cross(spin_vec, ecl_pole_vec, axis=0)
    spin_perp_vec2 = np.cross(spin_vec, spin_perp_vec1, axis=0)

    # calculate line of sight
    spin_cos = np.cos(spin_phase)
    spin_sin = np.sin(spin_phase)
    los_vec = (los_cos * spin_vec +
               los_sin * (spin_cos * spin_perp_vec1 + spin_sin * spin_perp_vec2))

    theta, phi = hp.vec2ang(los_vec)
    
    print(f"  Completed batch {batch_idx + 1} ({len(theta)} pointings in {time() - t1:.2f}s)")
    return theta, phi

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
theta_list, phi_list = zip(*results)
theta = np.concatenate(theta_list)
phi = np.concatenate(phi_list)

print(f"Total number of pointings for the whole survey: {len(theta):,}")

print("\nConverting angles to pixel indices...")
pix = hp.ang2pix(g.NSIDE["fossil"], theta, phi)

print("Creating hit map...")
hit_map = np.bincount(pix, minlength=hp.nside2npix(g.NSIDE["fossil"]))/g.NPIXPERIFG

print("Generating and saving plot...")
hp.mollview(hit_map, title="Hit Map for Fossil Scanning", unit="Hits", norm="hist", max=100)#, coord=["E", "G"])
plt.savefig("../output/hit_maps/scanning_strategy_fossil.png", bbox_inches="tight")
plt.close()
print("Saved hit map to ../output/hit_maps/scanning_strategy_planck.png")

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
        title="Scanning Strategy Hit Map for a Modern Experiment",
        unit="Hits",
        min=0,
        # max=np.nanmax(hit_map),
        max=332,
        xsize=2000,
        coord=["E", "G"],
    )
    plt.savefig("../output/hit_maps/scanning_strategy_modern.png", bbox_inches="tight")
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
