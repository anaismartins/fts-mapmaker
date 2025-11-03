import astropy.constants as const
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g

channels = {"rh": 0, "rl": 1, "lh": 2, "ll": 3}


def planck(freq, temp):
    """
    Planck function returning in units of MJy/sr.
    Input frequency in GHz and temperature in K.
    """
    h = 6.62607015e-34 * 1e9  # J GHz-1
    c = 299792458e-9  # m GHz
    k = 1.380649e-23  # J K-1

    if temp.shape != ():
        freq = freq[np.newaxis, :]
        temp = temp[:, np.newaxis]

    b = 2 * h * freq**3 / c**2 / (np.exp(h * freq / (k * temp)) - 1) * 1e20  # MJy sr-1

    return b


def dust(nu, A_d, nu_0, beta_d, T_d):
    """
    Returns the dust SED in units of MJy/sr.
    Input frequencies in GHz and temperature in K.

    Parameters
    ----------
    nu : float
        Frequency in GHz.
    nu0 : float
        Reference frequency in GHz.
    beta : float
        Dust spectral index.
    T : float
        Dust temperature in K.

    Returns
    -------
    float
        Dust SED in MJy/sr.
    """
    gamma = const.h / (const.k_B * T_d)
    s_d = (
        A_d
        * (nu / nu_0) ** (beta_d + 1)
        * (np.exp(gamma * nu_0) - 1)
        / (np.exp(gamma * nu) - 1)
    ).to(u.MJy / u.sr, equivalencies=u.brightness_temperature(nu_0))

    return s_d


def generate_frequencies(channel, mode, nfreq=None):
    """
    Generates an array with the frequencies in GHz for the given channel and mode.

    Parameters
    ----------
    channel : str
        The channel to generate frequencies for. Can be "lh", "ll", "rh", or "rl".
    mode : str or int
        The mode to generate frequencies for. Can be "ss" or "lf" for str or 0 or 3 for int.

    Returns
    -------
    f_ghz : np.ndarray
        An array with the frequencies in GHz.
    """

    # check if channel is str or int
    if isinstance(channel, int):
        channel_str = list(channels.keys())[list(channels.values()).index(channel)]
    elif isinstance(channel, str):
        channel_str = channel
    else:
        raise ValueError("Channel must be either int or str")

    nu0 = {"ss": 68.020812, "lf": 23.807283}
    dnu = {"ss": 13.604162, "lf": 3.4010405}
    nf = {
        "lh_ss": 210,
        "ll_lf": 182,
        "ll_ss": 43,
        "rh_ss": 210,
        "rl_lf": 182,
        "rl_ss": 43,
    }

    if not (mode == "lf" and (channel_str == "lh" or channel_str == "rh")):
        if nfreq == None:
            nfreq = nf[f"{channel_str}_{mode}"]
            f_ghz = np.linspace(
                nu0[mode],
                nu0[mode] + dnu[mode] * (nfreq - 1),
                nfreq,
            )
        else:
            f_ghz = np.linspace(0, dnu[mode] * (nfreq - 1), nfreq)
    else:
        raise ValueError("Invalid channel and mode combination")

    return f_ghz


def save_maps(freq, m):
    if g.FITS:
        hp.write_map(
            f"./../output/cg_mapmaker/{g.SIM_TYPE}/{int(freq):04d}.fits",
            m,
            overwrite=True,
            dtype=np.float64,
        )
    if g.PNG:
        hp.mollview(
            m,
            title=f"{int(freq):04d} GHz",
            unit="MJy/sr",
            min=0,
            max=50,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(f"../output/cg_mapmaker/{g.SIM_TYPE}/{int(freq):04d}.png")
        plt.close()
