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

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class class_plotting:
    ''' Class object that does the plotting'''

    def __init__(self):
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)