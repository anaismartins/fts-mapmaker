import os

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.io import fits

import globals as g
import spectra
import utils


def smooth_map(input_map):
    dust_map_smoothed = hp.smoothing(input_map, fwhm=np.deg2rad(g.BEAM[g.SIM_TYPE])) * u.uK

    dust_map_Mjy = dust_map_smoothed.to(
        u.MJy / u.sr,
        equivalencies=u.brightness_temperature(545 * u.GHz),
    )
    return dust_map_Mjy.value

def sim_dust(simtype, t0, run_name):
    dust_map_path = "../input/COM_CompMap_ThermalDust-commander_2048_R2.00.fits"
    dust_map = fits.open(dust_map_path)[1].data["I_ML_FULL"]
    t0 = utils.log_step("load_dust_map", t0, run_name)

    if not os.path.exists(f"../output/data/{simtype}/dust_map_ecl.fits"):
        # Convert from NESTED to RING (Planck maps are NESTED, rotate_map_alms expects RING)
        dust_map_ring = hp.reorder(dust_map, n2r=True)
        # rotate to ecliptic coordinates
        rot = hp.Rotator(coord=["G", "E"])
        m_ecl = rot.rotate_map_alms(dust_map_ring)
        # save
        hp.write_map(f"../output/data/{simtype}/dust_map_ecl.fits", m_ecl, overwrite=True)
        print("Saved dust map in ecliptic coordinates.")
    else:
        print("Loading dust map in ecliptic coordinates.")
        m_ecl = hp.read_map(f"../output/data/{simtype}/dust_map_ecl.fits")

    dust_map_Mjy = smooth_map(m_ecl)

    nu0_dust = 545 * u.GHz  # Planck 2015
    A_d = 163 * u.uK
    T_d = 21 * u.K
    beta_d = 1.53

    frequencies = spectra.generate_frequencies(simtype, nfreq=g.SPEC_SIZE[simtype])

    signal = spectra.dust(frequencies * u.GHz, A_d, nu0_dust, beta_d, T_d).value
    # check for invalid value encountered in divide
    signal = np.nan_to_num(signal)

    return dust_map_Mjy, frequencies, signal