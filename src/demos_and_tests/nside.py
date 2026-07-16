import globals as g
import numpy as np
import healpy as hp

beam = g.BEAM["fossil"]
print(f"The beam size of FOSSIL is {beam} degrees.")

nsides = [64, 128]
for nside in nsides:
    resol = np.rad2deg(hp.nside2resol(nside))
    print(f"NSIDE {nside} has pixels with {resol:.2f} degree resolution.")

    print(f"At NSIDE {nside}, there are {beam/resol:.2f} beams per pixel.")
