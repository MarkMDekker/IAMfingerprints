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
        print("STARTING READING")
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)

    def read_data_online(self):
        print('- Reading reference data from the scenario database')
        with open("database_credentials.yaml", "r") as stream:
            self.settings_db = yaml.load(stream, Loader=yaml.Loader)
        # Reference: ELEVATE NDC scenarios from global models
        self.df_reference = pyam.read_iiasa(self.settings['database']['elevate']['name'],
                                          model=self.settings['models'],
                                          scenario=self.settings['scenarios'],
                                          variable=self.settings['required_variables'],
                                          creds="database_credentials.yaml")
        self.xr_reference = self.df_reference.data.set_index(['model',
                                                                'scenario',
                                                                'region',
                                                                'variable',
                                                                'year']).to_xarray().drop_vars(['unit'])
        self.xr_data = xr.merge([self.xr_reference])
        self.xr_data = self.xr_data.rename({'variable': 'Variable',
                                            'region': 'Region',
                                            'model': 'Model',
                                            'scenario': 'Scenario',
                                            'year': 'Time',
                                            'value': 'Value'})
        self.xr_data = self.xr_data.reindex(Time = np.arange(2005, 2101))
        self.xr_data = self.xr_data.interpolate_na(dim="Time", method="linear")
        available_var = [x for x in self.settings['required_variables'] if x in self.xr_data['Variable'].values]
        self.xr_data_raw = self.xr_data
        self.xr_data_raw_sel = self.xr_data.sel(Scenario=self.settings['scenarios'],
                                                Model=self.settings['models'],
                                                Variable=available_var)
        xr_datas = []
        for i in list(self.settings['regional_mapping'].keys()):
            regs = np.intersect1d(self.xr_data_raw_sel.Region, self.settings['regional_mapping'][i])
            xr_datas.append(self.xr_data_raw_sel.sel(Region=regs).sum(dim='Region').expand_dims({'Region': [i]}))
        xr_data_new = xr.concat(xr_datas, dim='Region')

        # entries of zero fill with nan
        xr_data_new = xr_data_new.where(xr_data_new != 0, np.nan)
        xr_data_new = xr_data_new.transpose('Model', 'Scenario', 'Region', 'Variable', 'Time')
        self.xr_data = xr_data_new
        self.xr_data.to_netcdf("Data/xr_variables_reference.nc")

    def read_data_local(self):
        print('- Reading reference and your own scenario from local data and saving to xr_variables.nc')
        self.xr_data_ref = xr.open_dataset('Data/xr_variables_reference.nc')
        filename = 'MyScenario_rahel.csv'

        # Read data from the csv file
        try:
            df = pd.read_csv("Data/"+filename,
                                quotechar='"',
                                delimiter=',',
                                encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv("Data/"+filename,
                                quotechar='"',
                                delimiter=',',
                                encoding='latin')

        if len(df.keys()) == 1:
            try:
                df = pd.read_csv("Data/"+filename,
                                    quotechar='"',
                                    delimiter=';',
                                    encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv("Data/"+filename,
                                    quotechar='"',
                                    delimiter=';',
                                    encoding='latin')

        # Remove Unit column
        df2 = df.drop('Unit', axis=1)

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

        # Set scenario name
        if np.array(xr_data.Model)[0] == 'BLUES 2.0':
            xr_data = xr_data.sel(Scenario=['ELV-SSP2-NDC-D0-N'])
            xr_data = xr_data.assign_coords(Scenario=['ELV-SSP2-NDC-D0'])
        else:
            try: # if there is only one scenario
                xr_data = xr_data.assign_coords(Scenario=['ELV-SSP2-NDC-D0'])
            except ValueError: # if there are more, use the first
                xr_data = xr_data.sel(Scenario=[np.array(xr_data.Scenario)[0]])
                xr_data = xr_data.assign_coords(Scenario=['ELV-SSP2-NDC-D0'])

        # Some reindexing of time
        xr_data = xr_data.reindex(Time = np.arange(2005, 2101)).interpolate_na(dim="Time", method="linear")
        xr_data = xr_data.sel(Scenario=[x for x in self.settings['scenarios'] if x in xr_data['Scenario'].values],
                                Variable=[x for x in self.settings['required_variables'] if x in xr_data['Variable'].values])

        # Map regions from xr_data onto the ten regions
        xr_datas = []
        for reg_i, reg in enumerate(self.settings['regions']):
            try:
                xr_datas.append(xr_data.sel(Region=reg).expand_dims({'Region': [reg]}))
            except KeyError:
                xr_datas.append(xr_data.sel(Region=np.intersect1d(xr_data.Region, self.settings['regional_mapping'][reg])[0]).expand_dims({'Region': [reg]}))
        xr_data = xr.merge(xr_datas)

        # Concatenate the reference data with the new data
        try:
            self.xr_data_tot = xr.merge([self.xr_data_ref, xr_data])
        except: # if model name already in reference:
            xr_data = xr_data.assign_coords({'Model': [list(df.Model)[0] + " (myscenario)"]})
            self.xr_data_tot = xr.merge([self.xr_data_ref, xr_data])
        self.xr_data_tot.to_netcdf("Data/xr_variables.nc")