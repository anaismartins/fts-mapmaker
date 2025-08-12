import h5py
import os
import healpy as hp
import globals as g
import numpy as np
import matplotlib.pyplot as plt

user = os.environ["USER"]
data_path = f"/mn/stornext/u3/{user}/d5/firas-reanalysis/Commander/commander3/todscripts/firas/data/sky_v4.4.h5"

sky_data = h5py.File(
    data_path,
    "r",
)

ecl_lat = sky_data['df_data']['ecl_lat'][:]
ecl_lon = sky_data['df_data']['ecl_lon'][:]
mtm_speed = sky_data['df_data']['mtm_speed'][:]
mtm_length = sky_data['df_data']['mtm_length'][:]
scan = sky_data['df_data']['scan'][:] # up is 1, down is -1
pix_gal = (sky_data['df_data']['pix_gal'][:]).astype(int)
pix_ecl = (sky_data['df_data']['pix_ecl'][:]).astype(int)

# hit map
npix = hp.nside2npix(g.NSIDE)
hit_map = np.zeros(npix, dtype=int)
for i in range(pix_gal.shape[0]):
    hit_map[pix_gal[i]] += 1
hp.mollview(hit_map, title="Hit Map", unit="Hits", min=0, max=hit_map.max(), xsize=2000)
plt.savefig("../output/hit_map.png")

hit_map = np.zeros(npix, dtype=float)
for i in range(pix_ecl.shape[0]):
    hit_map[pix_ecl[i]] += 1
hp.mollview(hit_map, title="Hit Map Ecliptic", unit="Hits", min=0, max=hit_map.max(), xsize=2000, coord=["E", "G"])
plt.savefig("../output/hit_map_ecliptic.png")
plt.close()

# remake pix_ecl
pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon, ecl_lat, lonlat=True).astype(int)

# remake hit map
hit_map = np.zeros(npix, dtype=float)
for i in range(pix_ecl.shape[0]):
    hit_map[pix_ecl[i]] += 1
hp.mollview(hit_map, title="Hit Map Ecliptic Remade", unit="Hits", min=0, max=hit_map.max(), xsize=2000, coord=["E", "G"])
plt.savefig("../output/hit_map_ecliptic_remade.png")
plt.close()


print(f"shape of ecl_lat: {ecl_lat.shape}")

# times each mode takes for a full telemetered interferogram (in seconds)
times = {"ss": 55.36, "ls": 44.92, "sf": 39.36, "lf": 31.76}

# only using short slow for the simulations
short_slow_filter = ((mtm_speed == 0) & (mtm_length == 0))
ecl_lat_ss = ecl_lat[short_slow_filter]
ecl_lon_ss = ecl_lon[short_slow_filter]
scan_ss = scan[short_slow_filter]

# check for naNs
if ~np.isfinite(ecl_lat_ss).any() or ~np.isfinite(ecl_lon_ss).any():
    print("NaNs found in ecl_lat_ss or ecl_lon_ss. Exiting.")
    exit()

print("Ecliptic latitude (SS) max", ecl_lat_ss.max(), "min", ecl_lat_ss.min())
print("Ecliptic longitude (SS) max", ecl_lon_ss.max(), "min", ecl_lon_ss.min())

# set boundaries for longitude
# ecl_lon[ecl_lon < -180] = -180
# ecl_lon[ecl_lon > 180] = 180

speed = 3.5 # degrees per minute
speed = speed / 60  # degrees per second

# middle_position = (ecl_lat_ss, ecl_lon_ss)
# start_position = (ecl_lat_ss - speed * times["ss"] * scan_ss / 2, ecl_lon_ss)
ecl_lat_start = ecl_lat_ss - speed * times["ss"] * scan_ss / 2
ecl_lat_end = ecl_lat_ss + speed * times["ss"] * scan_ss / 2
# end_position = (ecl_lat_ss + speed * times["ss"] * scan_ss / 2, ecl_lon_ss)
# print(middle_position[0].min(), middle_position[1])
# print(start_position)
# set boundaries for latitude
# start_position[0][start_position[0] < -90] = - start_position[0][start_position[0] < -90] - 180
# start_position[0][start_position[0] > 90] = 180 - start_position[0][start_position[0] > 90]
# end_position[0][end_position[0] < -90] = - end_position[0][end_position[0] < -90] - 180
# end_position[0][end_position[0] > 90] = 180 - end_position[0][end_position[0] > 90]

ecl_lat_start[ecl_lat_start < -90] = -ecl_lat_start[ecl_lat_start < -90] - 180
ecl_lat_start[ecl_lat_start > 90] = 180 - ecl_lat_start[ecl_lat_start > 90]
ecl_lat_end[ecl_lat_end < -90] = -ecl_lat_end[ecl_lat_end < -90] - 180
ecl_lat_end[ecl_lat_end > 90] = 180 - ecl_lat_end[ecl_lat_end > 90]

# start_pix_ecl = hp.ang2pix(nside = g.NSIDE, theta=start_position[1], phi=start_position[0], lonlat=True)
# end_pix_ecl = hp.ang2pix(nside = g.NSIDE, theta=end_position[1], phi=end_position[0], lonlat=True)
# middle_pix_ecl = hp.ang2pix(nside = g.NSIDE, theta=middle_position[1], phi=middle_position[0], lonlat=True)
print(f"max ecl_lon_ss: {ecl_lon_ss.max()}, min: {ecl_lon_ss.min()}")

lat = np.linspace(ecl_lat_start.min(), ecl_lat_start.max(), 10000)
print(ecl_lat_start.min(), ecl_lat_start.max())
lon = np.ones(10000) * 180.00042091829943
print(speed * times["ss"] * scan_ss / 2)

pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon[short_slow_filter], ecl_lat[short_slow_filter], lonlat=True).astype(int)

#hp.ang2pix(32, lon, lat, lonlat = True)
#exit()

start_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_start, lonlat=True) #.astype(float)
end_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_end, lonlat=True) #.astype(float)
middle_pix_ecl = hp.ang2pix(g.NSIDE, ecl_lon_ss, ecl_lat_ss, lonlat=True) #.astype(float)

P = np.zeros((len(start_pix_ecl), 3), dtype=int)
P[:, 0] = start_pix_ecl
P[:, 1] = middle_pix_ecl
P[:, 2] = end_pix_ecl

for i in range(len(P)):
    if P[i,1] < 5000 or P[i,1] > 7000:
        print(f"Scan {i+1}: {P[i, 0]}, {P[i, 1]}, {P[i, 2]}")

# plot hit map of the scanning strategy
npix = hp.nside2npix(g.NSIDE)
hit_map = np.zeros(npix, dtype=float)
for i in range(P.shape[1]):
    hit_map += np.bincount(P[:, i], minlength=npix)/3

hp.mollview(hit_map, title="Scanning Strategy Hit Map", unit="Hits", min=0, max=hit_map.max(), xsize=2000, coord=["E", "G"])
plt.savefig("../output/scanning_strategy_hit_map.png")
plt.close()

# save pointing matrix
np.save("../input/firas_scanning_strategy.npy", P)