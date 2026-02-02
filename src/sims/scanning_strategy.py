"""
Utils for scanning strategy, in particular simulates the scanning strategy for a Planck-like
satellite in batches.
"""

import warnings

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time
from erfa import ErfaWarning

import globals as g

# Suppress ERFA warnings about dubious year for future dates
warnings.filterwarnings('ignore', category=ErfaWarning)


def calculate_batch(batch_idx, 
                    #batch_duration=1,
                    num_ifgs=12343,
                    one_ifg=7, coarse_step_sec=600, tilt_angle=5,
                    spin_rate=1/60, los_angle=87, verbose=False):
    """
    Get the pointing for one batch.

    Parameters
    ----------
    batch_idx : int
        Index of the batch.
    batch_duration : int, optional (DEPRECATED)
        Duration of the batch in days. Default is 1 day.
    num_ifgs : int, optional
        Number of interferograms in the batch. Default is 12343 IFGs (1 day worth).
    one_ifg : float, optional
        Time for one interferogram in seconds. Default is 7 seconds.
    coarse_step_sec : int, optional
        Coarse time step in seconds for sun position calculation. Default is 600 seconds.
    tilt_angle : float, optional
        Tilt angle of the spin axis in degrees. Default is 5 degrees.
    spin_rate : float, optional
        Spin rate in revolutions per minute. Default is 1/60 rpm (1 revolution per hour).
    """
    print(f"  Starting batch {batch_idx + 1}...")

    # set up times
    # start time is the start date offset by whiever batch idx we are at
    # start_day = batch_idx * batch_duration
    start_ifgs = batch_idx * num_ifgs
    start_day = (start_ifgs * one_ifg) / (24 * 3600)  # days
    start_time_offset = start_day * 24 * 3600 * u.s
    start_time = Time("2041-01-01T00:00:00") + start_time_offset

    # batch_duration_sec = batch_duration * 24 * 3600 # seconds
    batch_duration_sec = num_ifgs * one_ifg

    # time in seconds, but we want to get one pointing per ifg point
    # we need it more fine-grained than seconds
    one_pointing = one_ifg / g.NPIXPERIFG["fossil"] # seconds
    n_pointings_batch = int(batch_duration_sec / one_pointing)
    t_seconds = np.linspace(0, batch_duration_sec, n_pointings_batch, endpoint=False)
    t_coarse = np.arange(0, batch_duration_sec + coarse_step_sec, coarse_step_sec)  

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
    # six_months = 182.625 * 24 * 3600  # seconds
    hour20 = 20 * 3600  # seconds
    tilt_angle_rad = np.deg2rad(tilt_angle)
    precession_phase = (
        2 * np.pi * (start_day * 24 * 3600 + t_seconds) / hour20
    )  # radians
    s_vec = np.cos(tilt_angle_rad) * a_vec + np.sin(tilt_angle_rad) * (
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
    los_angle_rad = np.deg2rad(los_angle)
    los_cos = np.cos(los_angle_rad)
    los_sin = np.sin(los_angle_rad)
    los_vec = los_cos * s_vec + los_sin * (spin_cos * u_vec + spin_sin * v_vec)

    # convert pointing vectors to spherical coordinates
    lon, lat = hp.vec2ang(los_vec, True)

    pix = hp.ang2pix(g.NSIDE["fossil"], lon, lat, lonlat=True)
    print(f"  Finished batch {batch_idx + 1}.")
    if verbose:
        hit_map = np.bincount(pix, minlength=hp.nside2npix(g.NSIDE["fossil"]))
        # save hit map for each batch
        hp.write_map(
            f"../output/hit_maps/fossil/day_{batch_idx + 1:04d}.fits",
            hit_map,
            overwrite=True,
        )

    # split pix, lon, lat into chunks of size num_ifgs
    pix = np.array(np.split(pix, num_ifgs))
    lon = np.array(np.split(lon, num_ifgs))
    lat = np.array(np.split(lat, num_ifgs))

    return pix, lon, lat


if __name__ == "__main__":
    # test splitting into IFGs and generating hit map
    pix, lon, lat = calculate_batch(0, verbose=True)
    print(f"Shape of pix: {pix.shape}, lon: {lon.shape}, lat: {lat.shape}")

    # plot hit map of just one IFG
    hit_map = np.bincount(pix[0], minlength=hp.nside2npix(g.NSIDE["fossil"]))
    hp.mollview(
        hit_map,
        title="Scanning Strategy Hit Map - One IFG",
        unit="Hits",
        min=0,
        max=hit_map.max(),
        xsize=2000,
        coord=["E", "G"],
    )
    plt.savefig("../output/hit_maps/one_ifg.png")
    plt.close()

    # plot ten first IFGs
    hit_map = np.bincount(pix[:10].flatten(), minlength=hp.nside2npix(g.NSIDE["fossil"]))
    hp.mollview(
        hit_map,
        title="Scanning Strategy Hit Map - First 10 IFGs",
        unit="Hits",
        min=0,
        max=hit_map.max(),
        xsize=2000,
        coord=["E", "G"],
    )
    plt.savefig("../output/hit_maps/first_10_ifgs.png")
    plt.close()