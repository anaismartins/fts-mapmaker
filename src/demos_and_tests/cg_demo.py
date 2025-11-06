"""
This script aims to make sure we are correctly doing the CG by comparing it to a small example where we explicitely solve Ax=b.
"""

import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import sims.utils as sims
import utils

n_ifgs = 5
nside = 16
npix = hp.nside2npix(nside)

# first let's build all the matrices and solve it straight
# let's build d
if os.path.exists("../output/demo/spec.npz"):
    spec_data = np.load("../output/demo/spec.npz")
    spec = spec_data["spec"]
else:
    dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
    sed = np.nan_to_num(sed)

    spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]
    np.savez("../output/demo/spec.npz", spec=spec)

if os.path.exists("../output/demo/ifg_scanning.npz"):
    data = np.load("../output/demo/ifg_scanning.npz")
    ifg_scanning = data["ifg_scanning"]
    pix_ecl = data["pix_ecl"]
    sigma = data["sigma"]
else:
    ifg = np.fft.irfft(spec, axis=1)
    ifg = np.roll(ifg, 360, axis=1)
    ifg = ifg.real

    pix = np.random.randint(0, npix, size=(n_ifgs))
    pix_ecl = pix[:, np.newaxis] + np.random.randint(-1, 1, size=(n_ifgs, g.IFG_SIZE))
    print(f"Shape of pix_ecl: {pix_ecl.shape} and of ifg: {ifg.shape}")

    ifg_scanning = np.zeros((n_ifgs, g.IFG_SIZE))
    for i in range(g.IFG_SIZE):
        for pix_i, pix in enumerate(pix_ecl[:, i]):
            ifg_scanning[pix_i, i] = ifg[pix, i]

    noise, sigma = sims.white_noise(n_ifgs)
    ifg_scanning = ifg_scanning + noise

    d = ifg_scanning.flatten()

    np.savez(
        "../output/demo/ifg_scanning.npz",
        ifg_scanning=ifg_scanning,
        pix_ecl=pix_ecl,
        sigma=sigma,
    )
# end of else

if os.path.exists("../output/demo/d.npz"):
    data = np.load("../output/demo/d.npz")
    d = data["d"]
else:
    d = ifg_scanning.flatten()
    np.savez("../output/demo/d.npz", d=d)

if os.path.exists("../output/demo/N_inv.npz"):
    data = np.load("../output/demo/N_inv.npz")
    N_inv = data["N_inv"]
else:
    N = np.diag((sigma[:, np.newaxis] * np.ones(g.IFG_SIZE)).flatten() ** 2)
    N_inv = np.linalg.inv(N)
    print(f"Shape of N_inv: {N_inv.shape}")
    np.savez("../output/demo/N_inv.npz", N_inv=N_inv)

if os.path.exists("../output/demo/P.npz"):
    data = np.load("../output/demo/P.npz")
    P = data["P"]
    P_T = data["P_T"]
else:
    P = np.zeros((n_ifgs * g.IFG_SIZE, npix * g.IFG_SIZE))
    for pix_i in range(n_ifgs):
        for x_i in range(g.IFG_SIZE):
            P[pix_i * g.IFG_SIZE + x_i, pix_ecl[pix_i, x_i]] = 1

    P_T = P.T

    np.savez("../output/demo/P.npz", P=P, P_T=P.T)

if os.path.exists("../output/demo/A.npz"):
    data = np.load("../output/demo/A.npz")
    A_inv = data["A_inv"]
else:
    A = P_T @ N_inv @ P
    A_inv = np.linalg.pinv(A)
    print(f"Shape of A: {A.shape}")
    np.savez("../output/demo/A.npz", A=A, A_inv=A_inv)

if os.path.exists("../output/demo/b.npz"):
    data = np.load("../output/demo/b.npz")
    b = data["b"]
else:
    b = P_T @ N_inv @ d
    print(f"Shape of b: {b.shape}")
    np.savez("../output/demo/b.npz", b=b)

# print shapes of everything
print(f"Shape of d: {d.shape}")
print(f"Shape of N_inv: {N_inv.shape}")
print(f"Shape of P: {P.shape}")
print(f"Shape of P_T: {P_T.shape}")
print(f"Shape of A_inv: {A_inv.shape}")
print(f"Shape of b: {b.shape}")

m = A_inv @ b
m = m.reshape((npix, g.IFG_SIZE))
m = np.fft.rfft(m, axis=1)

frequencies = utils.generate_frequencies("ll", "ss", 257)
for freq_i, freq in enumerate(frequencies):
    hp.mollview(
        m[:, freq_i].real,
        title=f"Map at {freq/1e9:.2f} GHz solved directly",
        unit="MJy/sr",
    )
    plt.savefig(f"../output/demo/map_direct_{freq/1e9:.2f}_GHz.png")
    plt.close()
