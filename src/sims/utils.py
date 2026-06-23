import os

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body
from astropy.io import fits
from astropy.time import Time

import globals as g
import utils


def smooth_map(input_map):
    dust_map_smoothed = hp.smoothing(input_map, fwhm=np.deg2rad(g.BEAM[g.SIM_TYPE])) * u.uK

    dust_map_Mjy = dust_map_smoothed.to(
        u.MJy / u.sr,
        equivalencies=u.brightness_temperature(545 * u.GHz),
    )
    return dust_map_Mjy.value

def sim_dust(simtype):

    dust_map_path = "../input/COM_CompMap_ThermalDust-commander_2048_R2.00.fits"
    dust_map = fits.open(dust_map_path)[1].data["I_ML_FULL"]

    if not os.path.exists("../output/dust_map_ecl.fits"):
        # Convert from NESTED to RING (Planck maps are NESTED, rotate_map_alms expects RING)
        dust_map_ring = hp.reorder(dust_map, n2r=True)
        # rotate to ecliptic coordinates
        rot = hp.Rotator(coord=["G", "E"])
        m_ecl = rot.rotate_map_alms(dust_map_ring)
        # save
        hp.write_map("../output/dust_map_ecl.fits", m_ecl, overwrite=True)
        print("Saved dust map in ecliptic coordinates to ../output/dust_map_ecl.fits")
    else:
        print("Loading dust map in ecliptic coordinates from ../output/dust_map_ecl.fits")
        m_ecl = hp.read_map("../output/dust_map_ecl.fits")

    dust_map_Mjy = smooth_map(m_ecl)

    nu0_dust = 545 * u.GHz  # Planck 2015
    A_d = 163 * u.uK
    T_d = 21 * u.K
    beta_d = 1.53

    frequencies = utils.generate_frequencies(simtype, nfreq=g.SPEC_SIZE[simtype])

    signal = utils.dust(frequencies * u.GHz, A_d, nu0_dust, beta_d, T_d).value
    # check for invalid value encountered in divide
    signal = np.nan_to_num(signal)

    return dust_map_Mjy, frequencies, signal


def white_noise(ntod, simtype, sigma_min=None, sigma_max=None, signal=None, ifg=True):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform distribution.

    Parameters
    ----------
    npix : int
        How many pixels we assume each interferogram hits. For FIRAS, each interferogram will hit a maximum of 3 pixels, meaning this is the minimum number we want to use to divide the scanning strategy into multiple pixels. The higher this number, the closer to simulating a continuous scanning strategy.
    ntod : int
        Number of interferograms.
    sigma_min : float
        Standard deviation minimum value for the uniform distribution.
    sigma_max : float
        Standard deviation maximum value for the uniform distribution.
    ifg : bool
        If True, generate noise for interferograms (IFG_SIZE). If False, generate noise for spectra (SPEC_SIZE).
    Returns
    -------
    noise : array
        Array of shape (npix, ntod, IFG_SIZE) with the white noise to add to each interferogram.
    """
    if sigma_min is None and sigma_max is None:
        sigma_final = 1 / (np.max(signal) * 1e6)
        print(f"Shape of signal: {signal.shape}, max value: {np.max(signal)} MJy/sr")
        sigma_each = sigma_final * np.sqrt(signal.shape[0])

        sigma_min = sigma_each / 10
        sigma_max = sigma_each * 10
    sigmarand = np.random.uniform(sigma_min, sigma_max, (ntod))
    if ifg:
        size = g.IFG_SIZE[simtype]
    else:
        # open noise file
        with open('sims/noise_fossil.txt') as f:
            lines = f.readlines()[21:]
        
        for line in lines:
            _, frequency, sensitivity = line.split()
            print(f"Frequency: {frequency}, Sensitivity: {sensitivity}")

        size = g.SPEC_SIZE[simtype]
    noise = np.random.normal(0, sigmarand[:, np.newaxis], (ntod, size))
    return noise, sigmarand

def read_ephemerides():
    start = 58
    end = 35122
    julian_date = np.zeros(end - start + 1)
    x = np.zeros(end - start + 1)
    y = np.zeros(end - start + 1)
    lon = np.zeros(end - start + 1)
    lat = np.zeros(end - start + 1)
    with open("./sims/horizons_results.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # print(f"Line {i}: {line}")
            if i >= start-1 and i <= end-1:
                data = line.split()
                julian_date[i - (start-1)] = float(data[2])
                # x = 
                lon[i - (start-1)] = float(data[11])
                lat[i - (start-1)] = float(data[12])
                # if i == 58:
                #     print(f"Julian Date: {julian_date}, Lon: {lon}, Lat: {lat}")
    
    return julian_date, lon, lat

def plot_system(jd, l2_lon, l2_lat):
    # get ephemerides of earth
    earth = get_body('earth', Time(jd, format='jd')).transform_to("heliocentrictrueecliptic")
    earth_lon = earth.lon.deg
    earth_lat = earth.lat.deg

    for t in range(0, len(jd), 24):
        plt.plot(earth_lon[t], earth_lat[t], label='Earth', color='blue', marker='o')
        plt.plot(l2_lon[t], l2_lat[t], label='L2', color='red', marker='o')
        plt.xlabel('Ecliptic Longitude (deg)')
        plt.ylabel('Ecliptic Latitude (deg)')
        plt.title('Ephemerides of Earth and L2 Point')

        plt.xlim(-np.max(np.abs(l2_lon)), np.max(np.abs(l2_lon)))
        plt.ylim(-np.max(np.abs(l2_lat)), np.max(np.abs(l2_lat)))
        plt.legend()
        plt.savefig(f"../output/sims/scanning_strategy/ephemerides_{t:05d}.png")
        plt.clf()


if __name__ == "__main__":
    # jd, l2_lon, l2_lat = read_ephemerides()
    # plot_system(jd, l2_lon, l2_lat)

    dust_map_downgraded = sim_dust("firas")
