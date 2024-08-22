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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
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
# reader.read_data_online()
reader.read_data_local()

# =========================================================== #
# COMPUTE INDICATORS
# =========================================================== #

calculator = class_calculation(xr.open_dataset('Data/xr_variables.nc'))
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
# plotter.plot_variables()
# plotter.plot_indicators()