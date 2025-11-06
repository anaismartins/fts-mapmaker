# Fourier-Transform Spectrometer Mapmaker

## Simulating data

In order to run the mapmaker, we first need to have available data. If you already have your data, you can skip this step. If you do not, we will generate it starting from the dust map obtained from the component separation pipeline `Commander` from Planck data (i.e. we assume the sky is only made up of dust). You can get this file from the [Planck Legacy Archive](http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_ThermalDust-commander_2048_R2.00.fits).

If you are running on a FIRAS-like experiment, then your beam is much larger than that of Planck, meaning we first need to smooth the dust map down to a reasonable NSIDE for a FIRAS-like experiment, by running:

```
cd src
python -m sims.downgrade_planck_map
```

Once this is done, we want to get the maps for each observed frequency, which we can do by multiplying our dust map by the dust SED at the relevant frequencies. Then, we need to Fourier transform the map data into interferograms, which are the data format output by an FTS. We can generate these by running, for example, a "modern experiment":

```
python -m sims.modern
```

This script includes the generation of data with the same scanning strategy as FIRAS and white noise.

## Running the mapmaker

There are a few available mapmakers. The most recent one is the `white_noise_mapmaker` which you can run by using:

```
python white_noise_mapmaker.py
```

## Compare mapmakers

The `compare.py` is a basis script for comparing the results of the different mapmakers between themselves and with the originally simulated data (without noise).
