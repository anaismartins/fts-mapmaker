import os

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g

user = os.environ["USER"]
data_path = f"/mn/stornext/u3/{user}/d5/firas-reanalysis/Commander/commander3/todscripts/firas/data/sky_v4.4.h5"

sky_data = h5py.File(
    data_path,
    "r",
)

ecl_lat = sky_data["df_data"]["ecl_lat"][:]
ecl_lon = sky_data["df_data"]["ecl_lon"][:]
mtm_speed = sky_data["df_data"]["mtm_speed"][:]
mtm_length = sky_data["df_data"]["mtm_length"][:]
scan = sky_data["df_data"]["scan"][:]  # up is 1, down is -1

# times each mode takes for a full telemetered interferogram (in seconds)
times = {"ss": 55.36, "ls": 44.92, "sf": 39.36, "lf": 31.76}

# only using short slow for the simulations
short_slow_filter = (mtm_speed == 0) & (mtm_length == 0)
ecl_lat_ss = ecl_lat[short_slow_filter]
ecl_lon_ss = ecl_lon[short_slow_filter]
scan_ss = scan[short_slow_filter]

speed = 3.5  # degrees per minute
speed = speed / 60  # degrees per second

ecl_lat_start = ecl_lat_ss - speed * times["ss"] * scan_ss / 2
ecl_lat_end = ecl_lat_ss + speed * times["ss"] * scan_ss / 2

# adjust latitudes to be in the range [-90, 90]
ecl_lat_start[ecl_lat_start < -90] = -ecl_lat_start[ecl_lat_start < -90] - 180
ecl_lat_start[ecl_lat_start > 90] = 180 - ecl_lat_start[ecl_lat_start > 90]
ecl_lat_end[ecl_lat_end < -90] = -ecl_lat_end[ecl_lat_end < -90] - 180
ecl_lat_end[ecl_lat_end > 90] = 180 - ecl_lat_end[ecl_lat_end > 90]

start_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_start, lonlat=True)
end_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_end, lonlat=True)
middle_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_ss, lonlat=True)

npix = hp.nside2npix(g.NSIDE)

# remake hit map
hit_map = np.zeros(npix, dtype=float)
for i in range(middle_pix_ecl.shape[0]):
    hit_map[middle_pix_ecl[i]] += 1
hp.mollview(
    hit_map,
    title="Hit Map",
    unit="Hits",
    min=0,
    max=hit_map.max(),
    xsize=2000,
    coord=["E", "G"],
)
plt.savefig("../output/hit_map.png")
hp.write_map("../output/hit_map.fits", hit_map, overwrite=True)
plt.close()

P = np.zeros((len(start_pix_ecl) * 3), dtype=int)
P[0 : len(start_pix_ecl)] = start_pix_ecl
P[len(start_pix_ecl) : len(start_pix_ecl) * 2] = middle_pix_ecl
P[len(start_pix_ecl) * 2 :] = end_pix_ecl

# save pointing matrix
np.save("../input/firas_scanning_strategy.npy", P)
print("Pointing matrix saved to ../input/firas_scanning_strategy.npy")

# plot hit map of the scanning strategy
npix = hp.nside2npix(g.NSIDE)
hit_map_ss = np.bincount(P, minlength=npix)

hp.mollview(
    hit_map_ss,
    title="Scanning Strategy Hit Map",
    unit="Hits",
    min=0,
    max=hit_map_ss.max(),
    xsize=2000,
    coord=["E", "G"],
)
plt.savefig("../output/scanning_strategy_hit_map.png")
plt.close()

# compare hit maps
difference_map = hit_map - hit_map_ss
hp.mollview(
    difference_map,
    title="Hit map - Scanning strategy hit map",
    unit="Hits",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
plt.show()
ratio_map = hit_map / hit_map_ss
print("Ratio between hit map and scanning strategy hit map: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title="Ratio map hit map / scanning strategy",
    unit="Hits",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
plt.show()
