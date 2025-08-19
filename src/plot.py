import logging
import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from sim import sim_dust
from src.globals import FITS, PNG

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import globals as g


def plot_ifgs(ifg):
    # clean previous maps
    for file in os.listdir("./test_output/ifgs"):
        os.remove(f"./test_output/ifgs/{file}")

    for i in range(0, ifg.shape[0], 1000):
        # print(f"Plotting ifg {i}: {ifg[i]}")
        plt.plot(ifg[i])
        # plt.ylim(-300, 300)
        plt.savefig(f"./test_output/ifgs/{i}.png")
        plt.close()

def plot_dust_maps(dust_map_downgraded_mjy, frequencies, signal):
    # clean previous maps
    for file in os.listdir("../output/dust_maps"):
        os.remove(f"../output/dust_maps/{file}")
    # plot map for each frequency
    dust_map = dust_map_downgraded_mjy[:, np.newaxis] * signal[np.newaxis, :]
    for i, frequency in enumerate(frequencies):
        logging.info(f"Plotting dust map for frequency {i}")
        # dust_map = dust_map_downgraded_mjy * signal[i]
        if PNG:
            hp.mollview(dust_map[:, i], title=f"{int(frequency):04d} GHz", unit="MJy/sr", min=0, max=200)
            try:
                plt.savefig(f"../output/dust_maps/{int(frequency):04d}.fits")
            except Exception as e:
                logging.error(f"Failed to save plot for frequency {int(frequency):04d}: {e}")
            plt.clf()
        if FITS:
            output_file = f"../output/dust_maps/{int(frequency):04d}.fits"
            if os.path.exists(output_file):
                logging.warning(f"Overwriting existing file: {output_file}")
            hp.write_map(output_file, dust_map[:, i], overwrite=True)


def plot_m_invert(frequencies):
    # clean previous maps
    print("Cleaning previous maps")
    for file in os.listdir("./test_output/m_invert"):
        os.remove(f"./test_output/m_invert/{file}")

    m = np.load("./test_output/m_invert.npz")['m']
    # remove monopole

    if PNG:
        for i in range(m.shape[1]):
            # print(f"Plotting m for frequency {i}")
            hp.mollview(m[:, i].real, title=f"{int(frequencies.value[i]):04d} GHz", min=0, max=200, xsize=2000)
            plt.savefig(f"./test_output/m_invert/{int(frequencies.value[i]):04d}.png")
            plt.close()
            plt.clf()
    if FITS:
        for i in range(m.shape[1]):
            # print(f"Plotting m for frequency {i}")
            hp.write_map(f"./test_output/m_invert/{int(frequencies.value[i]):04d}.fits", m[:, i].real, overwrite=True)
            plt.close()
            plt.clf()

def plot_m_cg_per_tod(frequencies):
    # clean previous maps
    print("Cleaning previous maps")
    for file in os.listdir("./test_output/m_cg_per_tod"):
        os.remove(f"./test_output/m_cg_per_tod/{file}")

    m = np.load("./test_output/cg_per_tod.npz")['m']

    if PNG:
        for i in range(m.shape[1]):
            # print(f"Plotting m for frequency {i}")
            hp.mollview(m[:, i].real, title=f"{int(frequencies.value[i]):04d} GHz", min=0, max=200, xsize=2000)
            plt.savefig(f"./test_output/m_cg_per_tod/{int(frequencies.value[i]):04d}.png")
            plt.close()
            plt.clf()
    if FITS:
        for i in range(m.shape[1]):
            # print(f"Plotting m for frequency {i}")
            hp.write_map(f"./test_output/m_cg_per_tod/{int(frequencies.value[i]):04d}.fits", m[:, i].real, overwrite=True)
            plt.close()
            plt.clf()

def plot_simulated_hit_map():
    pix = np.load("test_output/ifgs.npz")["pix"]
    hit_map = np.bincount(pix, minlength=hp.nside2npix(g.NSIDE))
    hp.mollview(hit_map, title="Hit map", unit="Hits", min=0, max=1000)
    hp.graticule()
    plt.savefig("test_output/hit_map.png")

if __name__ == "__main__":
    # open ifgs
    # ifg = np.load("test_output/ifgs.npz")['ifg']
    # plot_ifgs(ifg)

    dust_map_downgraded_mjy, frequencies, signal = sim_dust()

    plot_dust_maps(dust_map_downgraded_mjy, frequencies, signal)
    # # plot_m_invert(frequencies)
    # plot_m_cg_per_tod(frequencies)

    # plot_simulated_hit_map()