"""
Testing things relating to scanning strategy, like is it possible to have one scan spanning more than 3 pixels?
"""

import healpy as hp
import os
import h5py
import numpy as np

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
scan = sky_data['df_data']['scan'][:]

short_slow_filter = ((mtm_speed == 0) & (mtm_length == 0))
ecl_lat_ss = ecl_lat[short_slow_filter]
ecl_lon_ss = ecl_lon[short_slow_filter]
scan_ss = scan[short_slow_filter]

times = {"ss": 55.36, "ls": 44.92, "sf": 39.36, "lf": 31.76}
speed = 3.5 # degrees per minute
speed = speed / 60  # degrees per second

position5 = (ecl_lat_ss - speed * times["ss"] * scan_ss / 10, ecl_lon_ss)
position4 = (ecl_lat_ss - speed * times["ss"] * scan_ss / 10 * 2, ecl_lon_ss)
position3 = (ecl_lat_ss - speed * times["ss"] * scan_ss / 10 * 3, ecl_lon_ss)
position2 = (ecl_lat_ss - speed * times["ss"] * scan_ss / 10 * 4, ecl_lon_ss)
position1 = (ecl_lat_ss - speed * times["ss"] * scan_ss / 10 * 5, ecl_lon_ss)
position6 = (ecl_lat_ss + speed * times["ss"] * scan_ss / 10, ecl_lon_ss)
position7 = (ecl_lat_ss + speed * times["ss"] * scan_ss / 10 * 2, ecl_lon_ss)
position8 = (ecl_lat_ss + speed * times["ss"] * scan_ss / 10 * 3, ecl_lon_ss)
position9 = (ecl_lat_ss + speed * times["ss"] * scan_ss / 10 * 4, ecl_lon_ss)
position10 = (ecl_lat_ss + speed * times["ss"] * scan_ss / 10 * 5, ecl_lon_ss)

pix1 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position1[1]), phi=np.deg2rad(position1[0]), lonlat=True)
pix2 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position2[1]), phi=np.deg2rad(position2[0]), lonlat=True)
pix3 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position3[1]), phi=np.deg2rad(position3[0]), lonlat=True)
pix4 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position4[1]), phi=np.deg2rad(position4[0]), lonlat=True)
pix5 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position5[1]), phi=np.deg2rad(position5[0]), lonlat=True)
pix6 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position6[1]), phi=np.deg2rad(position6[0]), lonlat=True)
pix7 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position7[1]), phi=np.deg2rad(position7[0]), lonlat=True)
pix8 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position8[1]), phi=np.deg2rad(position8[0]), lonlat=True)
pix9 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position9[1]), phi=np.deg2rad(position9[0]), lonlat=True)
pix10 = hp.pixelfunc.ang2pix(nside = 32, theta=np.deg2rad(position10[1]), phi=np.deg2rad(position10[0]), lonlat=True)

nunique = []

for i in range(len(pix1)):
    unique = np.unique([pix1[i], pix2[i], pix3[i], pix4[i], pix5[i], pix6[i], pix7[i], pix8[i], pix9[i], pix10[i]])
    nunique.append(len(unique))
    print(f"Scan {i+1}: {len(unique)} unique pixels")

print(f"Maximum of unique pixels is {max(nunique)}")

# build full pointing matrix - not working?
# pointing_matrix = np.zeros((len(start_pix_ecl), hp.nside2npix(g.NSIDE)))
# indices = np.stack([start_pix_ecl, middle_pix_ecl, end_pix_ecl], axis=1)
# np.add.at(pointing_matrix, (np.arange(len(indices))[:, None], indices), 1 / 3)

# plt.imshow(pointing_matrix, aspect='auto', cmap='viridis', norm='log')
# plt.colorbar(label='Pointing weight')
# plt.xlabel('Pixel index')
# plt.ylabel('Scan index')
# plt.title('FIRAS Scanning Strategy Pointing Matrix')
# # plt.show()
# plt.savefig("../output/firas_scanning_strategy.png")
# plt.close()