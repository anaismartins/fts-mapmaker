"""
Script to compare the dust simulation with the original FIRAS maps.
"""

import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import globals as g
import utils

path = "/mn/stornext/d16/cmbco/ola/firas/healpix_maps/"
filename = "FIRAS_map_0544GHz_lowf.fits"

# plot original firas map
# firas_map = hp.read_map(path + filename)
# hp.mollview(firas_map, title="FIRAS map", unit="MJy/sr", min=0, max=200)
# hp.graticule()
# plt.savefig("../output/compare/original_map.png")

# plot simulated dust map
# dust_map = fits.getdata("/mn/stornext/u3/aimartin/d5/firas-reanalysis/Commander/commander3/todscripts/firas/src/mapmaker/../output/dust_map_downgraded.fits")
# hp.mollview(dust_map, title="Dust map downgraded", unit="MJy/sr", min=0, max=200)
# hp.graticule()
# plt.savefig("../output/compare/dust_map.png")

# plot simulated 545 map
simulated_map = hp.read_map("../output/dust_maps/0544.fits")
hp.mollview(simulated_map, title="Simulated 545 GHz map", unit="MJy/sr", min=0, max=200)
hp.graticule()
plt.savefig("../output/compare/simulated_map.png")
plt.close()

# plot difference map
# difference_map = firas_map - simulated_map
# hp.mollview(difference_map, title="Original - Simulated", unit="MJy/sr", min=-200, max=200)
# hp.graticule()
# plt.savefig("../output/compare/difference_map.png")

# plot difference between input simulated map and output of invert mapmaker
# m_invert = hp.read_map("../output/m_invert/0546.fits")
# hp.mollview(m_invert, title="Invert map", unit="MJy/sr", min=0, max=200)
# hp.graticule()
# plt.savefig("../output/compare/invert_map.png")
# difference_map = simulated_map - m_invert
# hp.mollview(difference_map, title="Simulated - Invert", unit="MJy/sr", min=-1, max=1)
# hp.graticule()
# plt.savefig("../output/compare/difference_map_invert.png")

# check ratio between simulated map and invert map
# ratio_map = simulated_map / m_invert
# print("Ratio between simulated map and invert map: ", ratio_map)
# # plot ratio map
# hp.mollview(ratio_map, title="Ratio map", unit="MJy/sr", min=0.5, max=1.5)
# hp.graticule()
# plt.savefig("../output/compare/ratio_map.png")

# check differences in the maps in the middle and check for weird numbers
# check for nans
# print(f"Number of nans in simulated map: {np.isnan(simulated_map).sum()}")
# print(f"Number of nans in firas map: {np.isnan(firas_map).sum()}")
# print(f"Number of nans in invert map: {np.isnan(m_invert).sum()}")
# # print(f"Number of nans in cg map: {np.isnan(cg_map).sum()}")
# print(f"Number of nans in difference map: {np.isnan(difference_map).sum()}")
# print(f"Number of nans in ratio map: {np.isnan(ratio_map).sum()}")

# check numbers in the middle of the map
# print(f"Simulated map middle: {simulated_map[3072//2]}")
# print(f"Firas map middle: {firas_map[3072//2]}")
# print(f"Difference map middle: {difference_map[3072//2]}")
# print(f"Ratio map middle: {ratio_map[3072//2]}")
# # print(f"CG map middle: {cg_map[3072//2]}")
# print(f"m_invert map middle: {m_invert[3072//2]}")

# plot difference between input simulated map and output of cg solver
# cg_map = hp.read_map("../output/m_cg_per_tod/0546.fits")
# hp.mollview(cg_map, title="CG map", unit="MJy/sr", min=0, max=200)
# hp.graticule()
# plt.savefig("../output/compare/cg_map.png")
# difference_map = simulated_map - cg_map
# hp.mollview(difference_map, title="Simulated - CG", unit="MJy/sr", min=-1, max=1)
# hp.graticule()
# plt.savefig("../output/compare/difference_map_cg.png")
# # check ratio between simulated map and cg map
# ratio_map = simulated_map / cg_map
# print("Ratio between simulated map and cg map: ", ratio_map)
# # plot ratio map
# hp.mollview(ratio_map, title="Ratio map", unit="MJy/sr", min=0.5, max=1.5)
# hp.graticule()
# plt.savefig("../output/compare/ratio_map_cg.png")

white_noise_map = hp.read_map("../output/white_noise_mapmaker/0544.fits")
hp.mollview(
    white_noise_map,
    title="White noise mapmaker",
    unit="MJy/sr",
    min=0,
    max=200,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/white_noise_mapmaker.png")
plt.close()
difference_map = simulated_map - white_noise_map
hp.mollview(
    difference_map,
    title="Simulated - White noise mapmaker",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/difference_map_white_noise.png")
plt.close()
ratio_map = simulated_map / white_noise_map
print("Ratio between simulated map and white noise mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title="Ratio map",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/ratio_map_white_noise.png")
plt.close()

# compare naive mapmaker with maps made from the sed
binned_mapmaker = hp.read_map("../output/binned_mapmaker/0544.fits")
hp.mollview(
    binned_mapmaker,
    title="Binned mapmaker",
    unit="MJy/sr",
    min=0,
    max=200,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/binned_mapmaker.png")
plt.close()
difference_map = simulated_map - binned_mapmaker
hp.mollview(
    difference_map,
    title="Simulated - Binned mapmaker",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/difference_map_binned.png")
plt.close()
ratio_map = simulated_map / binned_mapmaker
print("Ratio between simulated map and binned mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title="Ratio map simulated / binned",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig("../output/compare/ratio_map_binned.png")
plt.close()

# compare binned mapmaker with white noise mapmaker
difference_map = binned_mapmaker - white_noise_map
hp.mollview(
    difference_map,
    title="Binned - White noise mapmaker",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
plt.savefig("../output/compare/difference_map_binned_white_noise.png")
plt.close()
ratio_map = binned_mapmaker / white_noise_map
print("Ratio between binned mapmaker and white noise mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title="Ratio map binned / white noise",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
plt.savefig("../output/compare/ratio_map_binned_white_noise.png")
plt.close()

# do the same but for all frequencies
frequencies = utils.generate_frequencies("ll", "ss", 257)
binned_mapmaker = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
for nui, frequency in enumerate(frequencies):
    binned_mapmaker[:, nui] = hp.read_map(
        f"./../output/binned_mapmaker/{int(frequency):04d}.fits"
    )
white_noise_map = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
for nui, frequency in enumerate(frequencies):
    white_noise_map[:, nui] = hp.read_map(
        f"./../output/white_noise_mapmaker/{int(frequency):04d}.fits"
    )
simulated_map = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
for nui, frequency in enumerate(frequencies):
    simulated_map[:, nui] = hp.read_map(
        f"./../output/dust_maps/{int(frequency):04d}.fits"
    )

# difference_map = simulated_map - binned_mapmaker
# for nui, frequency in enumerate(frequencies):
#     title = f"Simulated - Binned {int(frequency):04d} GHz"
#     output_path = f"../output/compare/difference_map_binned/{int(frequency):04d}.png"
#     hp.mollview(
#         difference_map[:, nui],
#         title=title,
#         unit="MJy/sr",
#         min=-1,
#         max=1,
#         cmap="RdBu_r",
#         coord=["E", "G"],
#     )
#     plt.savefig(output_path)
#     plt.close(plt.gcf())

# save 545 GHz difference map for paper
# paper_path = (
#     "/mn/stornext/u3/aimartin/d5/cosmoglobe-papers/FIRAS/ifg_mapmaking/Figures/"
# )
# hp.mollview(
#     difference_map[:, 40],
#     title=f"Simulated - Binned @ {int(frequencies[40]):04d} GHz",
#     unit="MJy/sr",
#     min=-1,
#     max=1,
#     cmap="RdBu_r",
#     coord=["E", "G"],
# )
# plt.savefig(paper_path + "difference_map_binned_0544GHz.png")
# plt.close()

difference_map = simulated_map - white_noise_map
for nui, frequency in enumerate(frequencies):
    title = f"Simulated - White noise @ {int(frequency):04d} GHz"
    output_path = (
        f"../output/compare/difference_map_white_noise/{int(frequency):04d}.png"
    )
    hp.mollview(
        difference_map[:, nui],
        title=title,
        unit="MJy/sr",
        min=-1,
        max=1,
        cmap="RdBu_r",
        coord=["E", "G"],
    )
    plt.savefig(output_path)
    plt.close(plt.gcf())


# compare simulated planck map before and after adding scanning strategy
simulated_map = hp.read_map("../output/dust_maps/0544.fits")
simulated_map_with_scanning = hp.read_map("../output/sim_maps/0544.fits")

difference_map = simulated_map - simulated_map_with_scanning
difference_map[simulated_map_with_scanning == 0] = np.nan
print(f"Difference between simulated map and simulated with scanning: {difference_map}")
hp.mollview(
    difference_map,
    title="Simulated - Simulated with scanning",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.show()
ratio_map = simulated_map / simulated_map_with_scanning
ratio_map[simulated_map_with_scanning == 0] = np.nan
print(f"Ratio between simulated map and simulated with scanning: {ratio_map}")
# plot ratio map
hp.mollview(
    ratio_map,
    title="Ratio map simulated / simulated with scanning",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.show()

# compare hit maps
