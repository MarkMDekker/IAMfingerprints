# IAMfingerprints
Suite for energy model diagnostics. Latest release can be found on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8220167.svg)](https://doi.org/10.5281/zenodo.8220167)

## Analyze own fingerprint
For the purpose of comparing any single scenario (e.g., from your own model) to any given reference scenario dataset (e.g., from a project), we created some code in the folder Interactive Fingerprinting. A step-by-step manual and recordings of a workshop where this is used will be shared soon. There is also a [`Compute_own_fingerprint.ipynb`] notebook in the Calculations folder that can be used to analyze scenarios similar to in the paper.

## Setup
You can set up the code by using `conda create --name <env> --file requirements.txt` in your command prompt, where `<env>` is the name of your conda environment you want to create for this. (The requirements file may contain a number of packages that are not used here.)

## Introduction
The code in this repository reads in scenario output of eight energy models (most of which are integrated assessment models) from the [ECEMF](https://www.ecemf.eu/) project. These scenarios are tailored to be diagnostic and reveal model behavior. The analysis yields a set of diagnostic indicators and model fingerprint diagrams in which model behavior can be distinguished.

## Data
Information on the scenarios can be found publicly on Zenodo, both the [dataset](https://zenodo.org/record/7634845) and the [protocol](https://doi.org/10.5281/zenodo.6782373). In our code, we read in the scenario data automatically from the IIASA database, using the `pyam` package. No credentials are needed for the public version of this database. To obtain the up-to-date ECEMF internal database, the user can adapt the `config.ini` file.

## Reproduce paper results
In [`config.yaml`](Configuration/config.yaml), you can set general settings for the calculations. The file [`Main.ipynb`](Calculations/Main.ipynb) first initializes class `class_indicatorcalculation.py` that downloads the scenario data and reformats this into a netcdf file called [`XRdata.nc`](Data/Handling/XRdata.nc), accessible and saved into the Data directory. Subsequently, in [`Main.ipynb`](Calculations/Main.ipynb), the class `class_indicatorcalculation.py` computes the diagnostic indicators from this netcdf file, producing another netcdf file [`XRindicators.nc`](Data/Output/XRindicators.nc), which includes all indicators by model and scenario. The plotting scripts can be found in the Plotting directory, and they read the aforementioned netcdf files, storing the figures in the Figures directory.

## Create your own model fingerprint
For modellers outside of ECEMF, it may be interesting to use this code and scenario ensemble to evaluate their own fingerprint. We provide a means to do so. Follow these steps:
 <ol>
  <li>Add your scenario data to the folder Data/Own_model folder. Take note of the format of the example file that we provided there (IMAGE_example.xlsx). Make sure that you have a unique model name, other than that is already inside the ECEMF database. Also, it is currently required that you use one of the diagnostic scenario names - running a completely different scenario would be comparing apples and pears.</li>
  <li>Navigate to the Compute_own_fingerprint notebook in the Calculations folder. At the top, you can insert some general parameters such as the year and region of your scope. The defaults are Europe in 2050. Also the name of the file that you added to the Data/Own_model folder needs to be inserted here.</li>
  <li>Run all cells in the notebook, and at the bottom you will find your model fingerprint.</li>
</ol>

## References
The paper in Nature Energy can be found here: https://www.nature.com/articles/s41560-023-01399-1

## Acknowledgments

This work was financially supported by the European Union’s Horizon 2020 research and innovation programme under the grant agreement No [101022622](https://cordis.europa.eu/project/id/101022622) (European Climate and Energy Modelling Forum [ECEMF](https://ecemf.eu/)).
