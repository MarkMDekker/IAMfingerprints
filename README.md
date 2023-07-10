# IAMfingerprints
Suite for calculating model diagnostics and computing model fingerprint diagrams

# Code description
In [config.yaml](Configuration/config.yaml), you can set general settings for the calculations. For example, if a model or scenario should be added or removed from the analysis, this can be done there. This repository is currently still dependent on data is that is downloaded from the ECEMF internal database via the `pyam` API, which I in turn sort by model in separate CSV files. The `ecemf_reader` class in the [Classes.py](Calculating/Classes.py) script in turn changes those files into a netcdf file that is used in the `calculator` class (also in [Classes.py](Calculating/Classes.py)).

# User guide
Currently, if a user would like to add their model, it would be via the ECEMF database. The step to work on is the data management after that, which should be following more directly from the ECEMF database to the netcdf that is used in the `calculator` class (currently, as mentioned, it's done via CSV I created in a different script). Then, a user would really only upload their data, add their model name in `config.yaml`, and it would be included. Also, the user-friendliness of the plotting should be improved.

# Licences, citations and stuff (need some help with this)
Whenever you use this code, please make sure to refer to our paper: https://www.researchsquare.com/article/rs-2638588/v1 (preprint, currently under review).
