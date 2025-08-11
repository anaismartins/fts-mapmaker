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

# times each mode takes for a full telemetered interferogram (in seconds)
times = {"ss": 55.36, "ls": 44.92, "sf": 39.36, "lf": 31.76}

# only using short slow for the simulations
short_slow_filter = ((mtm_speed == 0) & (mtm_length == 0))
ecl_lat_ss = ecl_lat[short_slow_filter]
ecl_lon_ss = ecl_lon[short_slow_filter]
scan_ss = scan[short_slow_filter]

print("Ecliptic latitude (SS) max", ecl_lat_ss.max(), "min", ecl_lat_ss.min())
print("Ecliptic longitude (SS) max", ecl_lon_ss.max(), "min", ecl_lon_ss.min())

# set boundaries for longitude
ecl_lon[ecl_lon < -180] = -180
ecl_lon[ecl_lon > 180] = 180

speed = 3.5 # degrees per minute
speed = speed / 60  # degrees per second

middle_position = (ecl_lat_ss, ecl_lon_ss)
start_position = (ecl_lat_ss - speed * times["ss"] * scan_ss / 2, ecl_lon_ss)
end_position = (ecl_lat_ss + speed * times["ss"] * scan_ss / 2, ecl_lon_ss)

# set boundaries for latitude
start_position[0][start_position[0] < -90] = - start_position[0][start_position[0] < -90] - 180
start_position[0][start_position[0] > 90] = 180 - start_position[0][start_position[0] > 90]
end_position[0][end_position[0] < -90] = - end_position[0][end_position[0] < -90] - 180
end_position[0][end_position[0] > 90] = 180 - end_position[0][end_position[0] > 90]

start_pix_ecl = hp.pixelfunc.ang2pix(nside = g.NSIDE, theta=np.deg2rad(start_position[1]), phi=np.deg2rad(start_position[0]), lonlat=True)
end_pix_ecl = hp.pixelfunc.ang2pix(nside = g.NSIDE, theta=np.deg2rad(end_position[1]), phi=np.deg2rad(end_position[0]), lonlat=True)
middle_pix_ecl = hp.pixelfunc.ang2pix(nside = g.NSIDE, theta=np.deg2rad(middle_position[1]), phi=np.deg2rad(middle_position[0]), lonlat=True)

P = np.zeros((len(start_pix_ecl), 3))
P[:, 0] = start_pix_ecl
P[:, 1] = middle_pix_ecl
P[:, 2] = end_pix_ecl

# save pointing matrix
np.save("../input/firas_scanning_strategy.npy", P)