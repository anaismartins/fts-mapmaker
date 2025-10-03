import os

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g


def generate_scanning_strategy(ecl_lat, ecl_lon, scan, npixperifg):
    # times each mode takes for a full telemetered interferogram (in seconds)
    times = {"ss": 55.36, "ls": 44.92, "sf": 39.36, "lf": 31.76}

    speed = 3.5  # degrees per minute
    speed = speed / 60  # degrees per second

    print(f"ecl_lat: {ecl_lat}")
    print(f"Shape of ecl_lat: {ecl_lat.shape}")

    if npixperifg > 1:
        ecl_lats = np.zeros((len(ecl_lat), npixperifg))
        if npixperifg % 2 == 0:
            for i in range(npixperifg // 2):
                ecl_lats[:, npixperifg // 2 + i] = (
                    ecl_lat + speed * times["ss"] * scan * (1 + i * 2) / npixperifg
                )
                ecl_lats[:, npixperifg // 2 - (i + 1)] = (
                    ecl_lat - speed * times["ss"] * scan * (1 + i * 2) / npixperifg
                )
        else:
            ecl_lats[:, npixperifg // 2] = ecl_lat
            for i in range(1, npixperifg // 2 + 1):
                ecl_lats[:, npixperifg // 2 + i] = (
                    ecl_lat + speed * times["ss"] * scan * (i * 2) / npixperifg
                )
                ecl_lats[:, npixperifg // 2 - i] = (
                    ecl_lat - speed * times["ss"] * scan * (i * 2) / npixperifg
                )
    elif npixperifg == 1:
        ecl_lats = ecl_lat
    else:
        raise ValueError("npixperifg must be at least 1")

    # print(f"ecl_lats: {ecl_lats}")
    print(
        f"Maximum latitude: {np.max(ecl_lats)} and minimum latitude: {np.min(ecl_lats)}"
    )

    for i in range(npixperifg):
        if i == npixperifg // 2:
            continue
        # adjust latitudes to be in the range [-90, 90]
        ecl_lats[:, i][ecl_lats[:, i] < -90] = (
            -ecl_lats[:, i][ecl_lats[:, i] < -90] - 180
        )
        ecl_lats[:, i][ecl_lats[:, i] > 90] = 180 - ecl_lats[:, i][ecl_lats[:, i] > 90]

    pix_ecl = np.zeros((len(ecl_lat), npixperifg), dtype=int)
    if npixperifg > 1:
        for i in range(npixperifg):
            pix_ecl[:, i] = hp.ang2pix(g.NSIDE, ecl_lon, ecl_lats[:, i], lonlat=True)
    else:
        pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon, ecl_lats, lonlat=True)
    # pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon[:, np.newaxis], ecl_lats, lonlat=True)
    print(f"Shape of pix_ecl: {pix_ecl.shape}")

    # P = np.zeros((len(start_pix_ecl) * npixperifg), dtype=int)
    # for i in range(npixperifg):
    #     P[i * len(start_pix_ecl) : (i + 1) * len(start_pix_ecl)] =
    # P[0 : len(start_pix_ecl)] = start_pix_ecl
    # P[len(start_pix_ecl) : len(start_pix_ecl) * 2] = middle_pix_ecl
    # P[len(start_pix_ecl) * 2 :] = end_pix_ecl
    print(f"pix_ecl inside function: {pix_ecl}")
    return pix_ecl


if __name__ == "__main__":
    user = os.environ["USER"]
    data_path = f"/mn/stornext/u3/{user}/d5/firas-reanalysis/Commander/commander3/todscripts/firas/data/sky_v4.4.h5"

    sky_data = h5py.File(
        data_path,
        "r",
    )

    mtm_speed = sky_data["df_data"]["mtm_speed"][:]
    mtm_length = sky_data["df_data"]["mtm_length"][:]
    ss_filter = (mtm_speed == 0) & (mtm_length == 0)

    ecl_lat = sky_data["df_data"]["ecl_lat"][ss_filter]
    ecl_lon = sky_data["df_data"]["ecl_lon"][ss_filter]
    scan = sky_data["df_data"]["scan"][ss_filter]  # up is 1, down is -1
    # pix_ecl = sky_data["df_data"]["pix_ecl"][:].astype(int)

    pix_ecl_original = hp.ang2pix(g.NSIDE, ecl_lon, ecl_lat, lonlat=True)
    print(f"pix_ecl from beginning: {pix_ecl_original}")

    # plot original hit map
    npix = hp.nside2npix(g.NSIDE)
    original_hit_map = np.bincount(pix_ecl_original, minlength=npix)
    if g.PNG:
        hp.mollview(
            original_hit_map,
            title="Original Hit Map",
            unit="Hits",
            min=0,
            max=original_hit_map.max(),
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig("../output/hit_maps/original.png")
        plt.close()
    if g.FITS:
        hp.write_map(
            "../output/hit_maps/original.fits", original_hit_map, overwrite=True
        )

    npixperifg = 512
    pix_ecl = generate_scanning_strategy(ecl_lat, ecl_lon, scan, npixperifg)
    print(f"Shape of pix_ecl: {pix_ecl.shape}")

    npix = hp.nside2npix(g.NSIDE)

    # remake hit map
    hit_map = np.bincount(pix_ecl[:, npixperifg // 2], minlength=npix)
    # hit_map = np.bincount(pix_ecl, minlength=npix)
    hp.mollview(
        hit_map,
        title="Hit Map",
        unit="Hits",
        min=0,
        max=hit_map.max(),
        xsize=2000,
        coord=["E", "G"],
    )
    if g.PNG:
        plt.savefig("../output/hit_maps/remade.png")
        plt.close()
    if g.FITS:
        hp.write_map("../output/hit_maps/remade.fits", hit_map, overwrite=True)

    P = pix_ecl.flatten()

    # save pointing matrix
    # np.save("../input/firas_scanning_strategy.npy", P)
    # print("Pointing matrix saved to ../input/firas_scanning_strategy.npy")

    # plot hit map of the scanning strategy
    npix = hp.nside2npix(g.NSIDE)
    hit_map_ss = np.bincount(P, minlength=npix) / npixperifg

    if g.PNG:
        hp.mollview(
            hit_map_ss,
            title="Scanning Strategy Hit Map",
            unit="Hits",
            min=0,
            max=hit_map_ss.max(),
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig("../output/hit_maps/scanning_strategy.png")
        plt.close()

    # compare hit maps
    difference_map = original_hit_map - hit_map_ss
    if g.PNG:
        hp.mollview(
            difference_map,
            title="Original hit map - Scanning strategy hit map",
            unit="Hits",
            min=-1,
            max=1,
            cmap="RdBu_r",
            coord=["E", "G"],
        )
        plt.savefig("../output/hit_maps/difference.png")
        plt.close()
    ratio_map = original_hit_map / hit_map_ss
    print("Ratio between original hit map and scanning strategy hit map: ", ratio_map)
    # plot ratio map
    if g.PNG:
        hp.mollview(
            ratio_map,
            title="Original hit map / Scanning strategy hit map",
            unit="Hits",
            min=0.5,
            max=1.5,
            cmap="RdBu_r",
            coord=["E", "G"],
        )
        plt.savefig("../output/hit_maps/ratio.png")
        plt.close()
