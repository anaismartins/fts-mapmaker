"""
This script aims to make sure we are correctly doing the CG by comparing it to a small example where we explicitely solve Ax=b.
"""

import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import cg_mapmaker as cg
import globals as g
import sims.utils as sims
import utils

n_ifgs = 3  # Reduced from 5 to make the problem smaller
nside = 2  # Reduced from 8 to make A fit in memory (was 1.12 TiB, now ~50 MB)
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
    print(f"Shape of A: {A.shape}")

    # Check for NaN or Inf values
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        print("WARNING: A contains NaN or Inf values!")

    # Use pseudo-inverse with error handling
    try:
        print("Computing pseudo-inverse of A...")
        A_inv = np.linalg.pinv(A, rcond=1e-10)
        print("Pseudo-inverse computed successfully")
    except np.linalg.LinAlgError as e:
        print(f"Error computing pseudo-inverse: {e}")
        print("Trying with higher rcond...")
        A_inv = np.linalg.pinv(A, rcond=1e-5)

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

if os.path.exists("../output/demo/m.npz"):
    data = np.load("../output/demo/m.npz")
    m = data["m"]
else:
    m = A_inv @ b
    m = m.reshape((npix, g.IFG_SIZE))
    m = np.fft.rfft(m, axis=1)
    np.savez("../output/demo/m.npz", m=m)

frequencies = utils.generate_frequencies("ll", "ss", 257)
for freq_i, freq in enumerate(frequencies):
    hp.mollview(
        m[:, freq_i].real,
        title=f"Map at {int(freq):04} GHz solved directly",
        unit="MJy/sr",
    )
    plt.savefig(f"../output/demo/map_direct/{int(freq):04}_GHz.png")
    plt.close()

print("Direct solution done.")

# now we want to run our cg to see if it gives the same solutions
# Flatten pix_ecl and expand sigma to match the expected format
pix_ecl_flat = pix_ecl.flatten()
sigma_expanded = (sigma[:, np.newaxis] * np.ones(g.IFG_SIZE)).flatten()

print(f"Shape of pix_ecl_flat: {pix_ecl_flat.shape}")
print(f"Shape of sigma_expanded: {sigma_expanded.shape}")

rms_map = np.zeros((npix, g.IFG_SIZE))
for pix_i in range(n_ifgs):
    for x_i in range(g.IFG_SIZE):
        pixel_idx = pix_ecl[pix_i, x_i]
        rms_map[pixel_idx, x_i] += (
            1 / sigma_expanded[pix_i * g.IFG_SIZE + x_i] ** 2
        )
rms_map = rms_map.flatten()

print(f"Min/max of rms_map: {np.min(rms_map)}, {np.max(rms_map)}")
print(f"Number of non-zero elements in rms_map: {np.count_nonzero(rms_map)}")

# Make b complex to match the CG solver expectations
b_complex = b.astype(np.complex128)

x = cg.preconditioned_conjugate_gradient(
    b_complex,
    pix_ecl_flat,
    sigma_expanded,
    rms_map,
    npix=npix,
    save_path="../output/demo/iteration_maps/",
)
print("CG solution done.")
