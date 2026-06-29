# Fourier-Transform Spectrometer Mapmaker

## Simulating data

In order to run the mapmaker, we first need to have available data. If you already have your data, you can skip this step. If you do not, we will generate it starting from the dust map obtained from the component separation pipeline `Commander` from Planck data (i.e. we assume the sky is only made up of dust). You can get this file from the [Planck Legacy Archive](http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_ThermalDust-commander_2048_R2.00.fits).

Once this is done, we want to get the maps for each observed frequency, which we can do by multiplying our dust map by the dust SED at the relevant frequencies. Then, we need to Fourier transform the map data into interferograms, which are the data format output by an FTS. We can generate these by running, for example, a FOSSIL-like experiment:

```
python -m sims.fossil
```

You can reduce runtime by skipping diagnostics and control worker count explicitly, for example:

```
cd src
python -m sims.fossil --no-plots --workers 16
```

## Project layout

The repository is easiest to maintain when data, generated products, and source code stay separated:

```
input/            Raw inputs that should rarely change
output/           Generated products, caches, and plots
src/sims/         Simulation and data-generation scripts
src/plotting/     Plot-only scripts and visual diagnostics
src/*_mapmaker.py Mapmaker entry points
```

Within `output/`, the current script convention is to keep cached pointings and derived products grouped by simulation type, for example `output/data/`, `output/sims/ifgs_fossil/`, and `output/dust_maps/fossil/`. That keeps heavy generated files out of the source tree while still making the pipeline outputs easy to find.

## Running the mapmaker

There are a few available mapmakers. The most recent one is the `white_noise_mapmaker` which you can run by using:

```
python white_noise_mapmaker.py
```

## Compare mapmakers

The `compare.py` is a basis script for comparing the results of the different mapmakers between themselves and with the originally simulated data (without noise).
