# =========================================================== #
# PREAMBULE
# Packages that we need
# =========================================================== #

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import pyam
import warnings
import importlib
warnings.filterwarnings("ignore")

import reading
import calculations
import plotting

importlib.reload(reading)
importlib.reload(calculations)
importlib.reload(plotting)

from reading import class_reading
from calculations import class_calculation
from plotting import class_plotting

# =========================================================== #
# READ DATA
# =========================================================== #

reader = class_reading()
reader.read_data_local()
reader.regional_aggregation()

# =========================================================== #
# COMPUTE INDICATORS
# =========================================================== #

calculator = class_calculation(reader.xr_data)
calculator.calculate_responsiveness_indicators()
calculator.calculate_mitigationstrategy_indicators()
calculator.calculate_energysupply_indicators()
calculator.calculate_energydemand_indicators()
calculator.calculate_costandeffort_indicators()
calculator.convert_to_indicator_xr()

# =========================================================== #
# PLOT RESULTS
# =========================================================== #

# plotter = class_plotting()