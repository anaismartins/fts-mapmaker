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

# plot simulated 545 map
simulated_map = hp.read_map("../output/dust_maps/0544.fits")
hp.mollview(simulated_map, title="Simulated 545 GHz map", unit="MJy/sr", min=0, max=50)
hp.graticule()
plt.savefig("../output/compare/simulated_map.png")
plt.close()

# compare naive mapmaker with maps made from the sed
binned_mapmaker = hp.read_map(f"../output/binned_mapmaker/{g.SIM_TYPE}/0544.fits")
hp.mollview(
    binned_mapmaker,
    title=f"Binned mapmaker ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=0,
    max=50,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig(f"../output/compare/binned_mapmaker_{g.SIM_TYPE}.png")
plt.close()
difference_map = simulated_map - binned_mapmaker
hp.mollview(
    difference_map,
    title=f"Simulated - Binned mapmaker ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
    norm="hist",
)
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{g.SIM_TYPE}/binned.png")
plt.close()
ratio_map = simulated_map / binned_mapmaker
print("Ratio between simulated map and binned mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title=f"Ratio map simulated / binned ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
    # norm="hist",
)
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{g.SIM_TYPE}/binned.png")
plt.close()

white_noise_map = hp.read_map(f"../output/white_noise_mapmaker/{g.SIM_TYPE}/0544.fits")
hp.mollview(
    white_noise_map,
    title=f"White noise mapmaker ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=0,
    max=50,
    coord=["E", "G"],
)
hp.graticule()
plt.savefig(f"../output/compare/white_noise_mapmaker_{g.SIM_TYPE}.png")
plt.close()
difference_map = simulated_map - white_noise_map
hp.mollview(
    difference_map,
    title=f"Simulated - White noise mapmaker ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=-1,
    max=1,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig(f"../output/compare/difference_maps/{g.SIM_TYPE}/white_noise.png")
plt.close()
ratio_map = simulated_map / white_noise_map
print("Ratio between simulated map and white noise mapmaker: ", ratio_map)
# plot ratio map
hp.mollview(
    ratio_map,
    title=f"Simulated / white noise mapmaker ({g.SIM_TYPE})",
    unit="MJy/sr",
    min=0.5,
    max=1.5,
    cmap="RdBu_r",
    coord=["E", "G"],
)
hp.graticule()
plt.savefig(f"../output/compare/ratio_maps/{g.SIM_TYPE}/white_noise.png")
plt.close()


# do the same but for all frequencies
# frequencies = utils.generate_frequencies("ll", "ss", 257)
# binned_mapmaker = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
# for nui, frequency in enumerate(frequencies):
#     binned_mapmaker[:, nui] = hp.read_map(
#         f"./../output/binned_mapmaker/{int(frequency):04d}.fits"
#     )
# white_noise_map = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
# for nui, frequency in enumerate(frequencies):
#     white_noise_map[:, nui] = hp.read_map(
#         f"./../output/white_noise_mapmaker/{int(frequency):04d}.fits"
#     )
# simulated_map = np.zeros((hp.nside2npix(g.NSIDE), len(frequencies)))
# for nui, frequency in enumerate(frequencies):
#     simulated_map[:, nui] = hp.read_map(
#         f"./../output/dust_maps/{int(frequency):04d}.fits"
#     )


# difference_map = simulated_map - white_noise_map
# for nui, frequency in enumerate(frequencies):
#     title = f"Simulated - White noise @ {int(frequency):04d} GHz"
#     output_path = (
#         f"../output/compare/difference_map_white_noise/{int(frequency):04d}.png"
#     )
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
