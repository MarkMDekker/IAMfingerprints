# =========================================================== #
# PREAMBULE
# Packages that we need
# =========================================================== #

import xarray as xr
from reading import class_reading
from calculations import class_calculation
from plotting import class_plotting

# This is generally not recommended, but for the workshop we are removing all warnings that arise from empty averages
import warnings
warnings.filterwarnings("ignore") 

# =========================================================== #
# READ DATA
# =========================================================== #

reader = class_reading()
# reader.create_reference_from_local() # Don't use this line if you already have been provided the reference data
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

plotter = class_plotting()
plotter.plot_variables()
plotter.plot_variables_norm()
plotter.plot_indicators()
