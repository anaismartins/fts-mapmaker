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


def white_noise(ntod, simtype, ifg=True, signal=None):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform distribution.

    Parameters
    ----------
    ntod : int
        Number of interferograms.
    simtype : str
        Type of simulation, e.g. "fossil" or "firas".
    ifg : bool
        If True, generate noise for interferograms (IFG_SIZE). If False, generate noise for spectra (SPEC_SIZE).
    signal : array, optional
        The signal array to determine the noise level.
    Returns
    -------
    noise : array
        Array of shape (npix, ntod, IFG_SIZE) with the white noise to add to each interferogram.
    """
    size = g.IFG_SIZE[simtype]
    sigma = None
    
    if not ifg:
        if simtype == "fossil":
            # open noise file for FOSSIL
            with open('sims/data/noise_fossil.txt') as f:
                lines = f.readlines()[20:]
            
            noise_each = np.zeros(129) # TODO: my frequencies and the noise frequencies do not match, decide what to do
            for i, line in enumerate(lines):
                if i >= 129:
                    break
                _, _, sensitivity = line.split()

                noise_each[i] = float(sensitivity) * np.sqrt(signal.shape[0]) / 10e6 # MJy
                
            # Transform tabulated spectral sensitivities to the interferogram domain.
            # A standard deviation must be non-negative, so we enforce positivity.
            sigma = np.fft.irfft(noise_each, n=size)
            sigma = np.abs(np.real(sigma))
            print(f"Noise level: {np.max(sigma):.2g} MJy/sr")
        elif simtype == "firas":
            firas_noise = fits.open("sims/FIRAS_CALIBRATION_ERRORS_LHSS.FITS")
            print(firas_noise.info()) # TODO: check this and plot against calibration paper, figure 9
            raise NotImplementedError("FIRAS noise model is not implemented yet.")

        print(f"Noise shape: {sigma.shape}, signal shape: {signal.shape}")

    if sigma is None:
        raise ValueError("Could not derive noise sigma; check simtype/ifg configuration.")

    noise = np.random.normal(0, sigma[np.newaxis, :], (ntod, size))
    return noise, sigma

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
