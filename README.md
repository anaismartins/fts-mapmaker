# Fourier-Transform Spectrometer Mapmaker

## Simulating data

In order to run the mapmaker, we first need to have available data. If you already have your data, you can skip this step. If you do not, we will generate it starting from the dust map obtained from the component separation pipeline `Commander` from Planck data (i.e. we assume the sky is only made up of dust). You can get this file from the [Planck Legacy Archive](http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_ThermalDust-commander_2048_R2.00.fits).

Once this is done, we want to get the maps for each observed frequency, which we can do by multiplying our dust map by the dust SED at the relevant frequencies. Then, we need to Fourier transform the map data into interferograms, which are the data format output by an FTS. We can generate these by running, for example, a FOSSIL-like experiment:

```
cd src
python -m sims.fossil fossil_sim_v40_noiseless_nside128 --nworkers 16 --plots debug
python binned_mapmaker.py fossil_sim_v40_noiseless_nside128
python cg_mapmaker.py fossil_sim_v40_noiseless_nside128
python compare.py fossil_sim_v40_noiseless_nside128
```

## Simulation versions

Each simulation version has one config file, `configs/<sim_name>.yaml`, which holds all its
parameters (resolution, noise, pointing, ...). Parameters not in the file take the defaults in
`src/config.py`. The command line only controls how a script runs (`--nworkers`, `--plots`, ...),
never what is simulated.

To make a new version, copy an existing config file to a new name and edit it. When a version is
simulated, its full parameter set is recorded in `output/data/<sim_type>/<sim_name>/config.yaml`,
with the git commit, host and command. The mapmakers and `compare.py` read the parameters from
there, and each of them writes a `run_info.yaml` into its output folder. Editing the config file of
a version that has already been simulated gives an error. `python config.py <sim_name>` prints the
parameters of a version.

## Project layout

The repository is easiest to maintain when data, generated products, and source code stay separated:

```
configs/          One parameter file per simulation version
input/            Raw inputs that should rarely change
output/           Generated products, caches, and plots
src/sims/         Simulation and data-generation scripts
src/plotting/     Plot-only scripts and visual diagnostics
src/*_mapmaker.py Mapmaker entry points
```

Every output of a version is in a folder named after it, e.g.
`output/<binned|cg|legacy|compare>/<sim_type>/<sim_name>/`. The pointing is expensive to generate,
so it is shared between versions in `output/data/<sim_type>/pointing/<pointing>/`. The full layout
is described in `src/paths.py`.