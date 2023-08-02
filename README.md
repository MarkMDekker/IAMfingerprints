# IAMfingerprints
Suite for energy model diagnostics

## Introduction
The code in this repository reads in scenario output of eight energy models (most of which are integrated assessment models) from the [ECEMF](https://www.ecemf.eu/) project. These scenarios are tailored to be diagnostic and reveal model behavior.

## Data
Information on the scenarios can be found publicly on Zenodo, both the [dataset](https://zenodo.org/record/7634845) and the [protocol](https://doi.org/10.5281/zenodo.6782373). In our code, we read in the scenario data automatically from the IIASA database, using the `pyam` package. No credentials are needed for the public version of this database. To obtain the up-to-date ECEMF internal database, the user can adapt the `config.ini` file.

## Usage
In [`config.yaml`](Configuration/config.yaml), you can set general settings for the calculations. The file [`Main.ipynb`](Calculations/Main.ipynb) first initializes class `class_indicatorcalculation.py` that downloads the scenario data and reformats this into a netcdf file called [`XRdata.nc`](Data/Handling/XRdata.nc), accessible and saved into the Data directory. Subsequently, in [`Main.ipynb`](Calculations/Main.ipynb), the class `class_indicatorcalculation.py` computes the diagnostic indicators from this netcdf file, producing another netcdf file [`XRindicators.nc`](Data/Output/XRindicators.nc), which includes all indicators by model and scenario. The plotting scripts can be found in the Plotting directory, and they read the aforementioned netcdf files, storing the figures in the Figure directory.

## References
Whenever you use this code, please make sure to refer to our paper: https://www.researchsquare.com/article/rs-2638588/v1.
