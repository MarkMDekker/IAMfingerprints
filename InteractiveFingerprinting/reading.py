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

class class_reading:
    ''' Class object that reads in the data from the scenario databases '''

    def __init__(self):
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)

    def read_data_online(self):
        # Reference: ELEVATE NDC
        pyam.iiasa.set_config(self.settings['database']['username'],
                              self.settings['database']['password'])
        #pyam.iiasa.Connection(self.settings['database']['ecemf']['name'])
        self.df_ecemf = pyam.read_iiasa(self.settings['database']['ecemf']['name'],
                                          model=self.settings['models'],
                                          scenario=self.settings['scenarios'],
                                          variable=self.settings['required_variables'])
        self.xr_ecemf = self.df_ecemf.data.set_index(['model',
                                                          'scenario',
                                                          'region',
                                                          'variable',
                                                          'year']).to_xarray().drop_vars(['unit'])

        # Other national models: COMMITTED
        # XYZ

        self.xr_data = xr.merge([self.xr_ecemf])
        
        self.xr_data = self.xr_data.rename({'variable': 'Variable',
                                            'region': 'Region',
                                            'model': 'Model',
                                            'scenario': 'Scenario',
                                            'year': 'Time',
                                            'value': 'Value'})
        self.xr_data = self.xr_data.reindex(Time = np.arange(2005, 2101))
        self.xr_data = self.xr_data.interpolate_na(dim="Time", method="linear")

    def read_data_local(self):
        xr_datas = []
        for datastring in ['elevate-internal_snapshot_20240723 global and R10.csv',
                           'elevate-internal_snapshot_20240723 national.csv']:
            # Read data from the csv file
            df = pd.read_csv("K:/Code/IAMfingerprints/InteractiveFingerprinting/Data/"+datastring, quotechar='"', delimiter=',', encoding='utf-8')

            # Remove last row
            df2 = df.iloc[:-1]

            # Remove Unit column
            df2 = df2.drop('Unit', axis=1)

            # Transform the columns with years into a single column called Time
            df2 = pd.melt(df2, id_vars=['Model', 'Scenario', 'Region', 'Variable'], var_name='Time', value_name='Value')

            # Transform into xarray object with coordinates Model, Scenario, Region and Variable
            xr_data = df2.set_index(['Model', 'Scenario', 'Region', 'Variable', 'Time']).to_xarray()

            # Change the type of the Time coordinate to integer
            xr_data = xr_data.assign_coords(Time=xr_data.Time.astype(int))

            # Replace all values that are '' to nan
            xr_data = xr_data.where(xr_data != '', np.nan)

            # Change the type of the Value variable to float
            xr_data = xr_data.assign(Value=xr_data.Value.astype(float))

            # Some reindixing of time
            xr_data = xr_data.reindex(Time = np.arange(2005, 2101))
            xr_datas.append(xr_data.interpolate_na(dim="Time", method="linear"))
        xr_data = xr.concat(xr_datas, dim='Region')
        available_var = [x for x in self.settings['required_variables'] if x in xr_data['Variable'].values]
        self.xr_data_raw = xr_data
        self.xr_data_raw_sel = xr_data.sel(Scenario=self.settings['scenarios'],
                                            Model=self.settings['models'],
                                            Variable=available_var)

    def regional_aggregation(self):
        # Map certain regions towards aggregated regions
        xr_datas = []
        for i in list(self.settings['regional_mapping'].keys()):
            xr_datas.append(self.xr_data_raw_sel.sel(Region=self.settings['regional_mapping'][i]).sum(dim='Region').expand_dims({'Region': [i]}))
        xr_data_new = xr.concat(xr_datas, dim='Region')

        # entries of zero fill with nan
        xr_data_new = xr_data_new.where(xr_data_new != 0, np.nan)
        xr_data_new = xr_data_new.transpose('Model', 'Scenario', 'Region', 'Variable', 'Time')
        self.xr_data = xr_data_new
        self.xr_data.to_netcdf("Data/xr_variables.nc")