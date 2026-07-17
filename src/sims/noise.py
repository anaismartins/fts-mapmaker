import numpy as np
from astropy.io import fits

import globals as g


def white_noise(ntod, simtype, ifg=True, signal=None):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform
    distribution.

    Parameters
    ----------
    ntod : int
        Number of interferograms.
    simtype : str
        Type of simulation, e.g. "fossil" or "firas".
    ifg : bool
        If True, generate noise for interferograms (IFG_SIZE). If False, generate noise for spectra
        (SPEC_SIZE).
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
            print(f"Maximum noise level per IFG: {np.max(sigma):.2g} MJy/sr.")
        elif simtype == "firas":
            firas_noise = fits.open("sims/FIRAS_CALIBRATION_ERRORS_LHSS.FITS")
            print(firas_noise.info()) # TODO: check this and plot against calibration paper, figure 9
            raise NotImplementedError("FIRAS noise model is not implemented yet.")

    if sigma is None:
        raise ValueError("Could not derive noise sigma; check simtype/ifg configuration.")

    noise = np.random.normal(0, sigma[np.newaxis, :], (ntod, size))
    return noise, sigma