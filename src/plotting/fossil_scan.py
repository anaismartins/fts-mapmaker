import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g

path = "/mn/stornext/u3/aimartin/d5/fts-mapmaker/output/hit_maps/fossil/"

survey_len = 4 # years
obs_eff = 0.7

days = int(survey_len * 365.25 * obs_eff)
cum_hit_map = np.zeros(hp.nside2npix(g.NSIDE["fossil"]))

for day in range(1, days+1):
    hit_map = hp.read_map(f"{path}day_{day:04d}.fits")
    hp.mollview(hit_map, title=f"FOSSIL Hit Map - Day {day:04d}", unit="Hits per pixel")
    plt.savefig(f"{path}day_{day:04d}.png")
    plt.close()

    cum_hit_map += hit_map
    hp.mollview(cum_hit_map / max(cum_hit_map), title=f"FOSSIL Cumulative Hit Map - Day {day:04d}",
                unit="Normalized Hits", coord="E", min=0, max=1)
    plt.savefig(f"{path}cumulative_day_{day:04d}_ecliptic.png")
    plt.close()

    hp.mollview(cum_hit_map / max(cum_hit_map), title=f"FOSSIL Cumulative Hit Map - Day {day:04d}",
                unit="Normalized Hits", coord=["E", "G"], min=0, max=1)
    plt.savefig(f"{path}cumulative_day_{day:04d}_galactic.png")
    plt.close()