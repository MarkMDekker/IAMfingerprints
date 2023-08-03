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

class DataHandling:
    ''' Class object that reads in the data from the ECEMF database
    and computes the diagnostic indicators '''

    def __init__(self):
        with open("../Configuration/config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.list_of_models = [self.settings['models'][m]['full_name'] for m in self.settings['models'].keys()]
        self.list_of_colors = [self.settings['models'][m]['color'] for m in self.settings['models'].keys()]

    def read_raw_data(self):
        if self.settings['database']['username'] != 'none':
            pyam.iiasa.set_config(self.settings['database']['username'], self.settings['database']['password'])
        pyam.iiasa.Connection(self.settings['database']['name'])
        df = pyam.read_iiasa('ecemf_internal', model=self.list_of_models, scenario=self.settings['scenarios'])

        def filters(df_i, model):
            if model == 'WITCH 5.0': # Remove ResidualFossil (old / erroneous scenario)
                df_i = df_i[df_i.scenario != 'DIAG-C400-lin-ResidualFossil']
                df_i = df_i.reset_index(drop=True)
            if model == 'Euro-Calliope 2.0' or model == 'PRIMES 2022': # Obtain NPI from the base scenario
                df_i2 = df_i[df_i.scenario == 'DIAG-NPI']
                df_i2.scenario = 'DIAG-Base'
                df_i = df_i.append(df_i2)
            if model == 'REMIND 2.1': # Remove Other electricity category
                df_i = df_i[df_i.variable != 'Secondary Energy|Electricity|Other']
                df_i = df_i.reset_index(drop=True)
            if model == 'PROMETHEUS 1.2' or model == 'PRIMES 2022': # Remove erroneous zero-policy costs
                df_i = df_i[(df_i.variable != 'Policy Cost|Area under MAC Curve') & (df_i.variable != 'Policy Cost|Consumption Loss')]
                df_i = df_i.reset_index(drop=True)
            return df_i

        for model_i, model in enumerate(self.list_of_models):
            a = 0
            while a != -1:
                region = self.settings['region_order'][a]
                df_mod = df.filter(model=model, region = region).data
                if len(df_mod) == 0:
                    a+=1
                else:
                    a = -1
                    df_afterfilters = filters(df_mod, model)
                    if model_i == 0:
                        dataframe_eu = df_afterfilters
                    else:
                        dataframe_eu = pd.concat([dataframe_eu, df_afterfilters])
                if a == 4:
                    a = -1
        
        for model_i, model in enumerate(self.list_of_models):
            df_mod = df.filter(model=model, region = 'World').data
            df_afterfilters = filters(df_mod, model)
            if model_i == 0:
                dataframe_world = df_afterfilters
            else:
                dataframe_world = pd.concat([dataframe_world, df_afterfilters])
        
        self.pd_mod = pd.concat([dataframe_eu, dataframe_world])
        self.pd_mod = self.pd_mod.reset_index(drop=True)
        regs = np.array(self.pd_mod.region)
        regs[regs != 'World'] = 'Europe'
        self.pd_mod.region = regs

    def add_average_gdps(self):
        # Remove GDP data from respective models (nulls/nans)
        for m_i, m in enumerate(self.settings['models_requiring_gdpav']):
            self.pd_mod = self.pd_mod.drop(self.pd_mod[(self.pd_mod.variable == 'GDP|PPP') & (self.pd_mod.model == m)].index)
            self.pd_mod = self.pd_mod.reset_index(drop=True)

        # Compute GDP averages per model
        av_gdp = np.array(self.pd_mod[self.pd_mod.variable == 'GDP|PPP'])[:, 5:]
        ar_models = np.array(self.pd_mod.model)
        ar_variables = np.array(self.pd_mod.variable)
        ar_regions = np.array(self.pd_mod.region)
        ar_scenarios = np.array(self.pd_mod.scenario)
        ar_times = self.pd_mod.year
        time = np.unique(self.pd_mod.year)

        gdp_over_time_eu_allmods = []
        gdp_over_time_w_allmods = []
        for m_i, m in enumerate(self.list_of_models):
            pd_eu = self.pd_mod[(ar_variables == 'GDP|PPP') & (ar_models == m) & (ar_regions == 'Europe')]
            pd_w = self.pd_mod[(ar_variables == 'GDP|PPP') & (ar_models == m) & (ar_regions == 'World')]
            gdp_over_time_eu = []
            gdp_over_time_w = []
            for t_i, t in enumerate(time):
                gdp_over_time_eu.append(np.mean(pd_eu[ar_times == t].value))
                gdp_over_time_w.append(np.mean(pd_w[ar_times == t].value))
            gdp_over_time_eu_allmods.append(gdp_over_time_eu)
            gdp_over_time_w_allmods.append(gdp_over_time_w)

        gdp_eu = np.nanmean(np.array(gdp_over_time_eu_allmods), axis=0)
        gdp_w = np.nanmean(np.array(gdp_over_time_w_allmods), axis=0)

        # Add to dataframe
        rows = []
        for s_i, s in enumerate(self.settings['scenarios']):
            for m_i, m in enumerate(self.settings['models_requiring_gdpav']):
                for t_i, t in enumerate(time):
                    rows.append([m, s, 'Europe', "GDP|PPP", "billion EUR_2020/yr", t, gdp_eu[t_i]])
                    rows.append([m, s, 'World', "GDP|PPP", "billion EUR_2020/yr", t, gdp_w[t_i]])
        df_rows = pd.DataFrame(rows, columns = ["model", "scenario", "region", "variable", "unit", 'year', 'value'])
        self.pd_mod = pd.concat([self.pd_mod, df_rows])
        self.pd_mod = self.pd_mod.reset_index(drop=True)
    
    def convert_to_xr(self):
        dummy_ = self.pd_mod.drop(['unit'], axis=1)
        dummy_ = dummy_.rename(columns={'year': 'Time',
                                'model': 'Model',
                                'scenario': 'Scenario',
                                'variable': 'Variable',
                                'region': 'Region',
                                'value': 'Value'})
        dummy_ = dummy_.set_index(["Model", "Scenario", "Variable", "Region", "Time"])
        self.data_xr = xr.Dataset.from_dataframe(dummy_)
        self.data_xr = self.data_xr.reindex(Time = np.arange(1995, 2101))
        self.data_xr = self.data_xr.interpolate_na(dim="Time", method="linear")
        self.data_xr.to_netcdf(self.settings['paths']['data']['handling'] + "XRdata.nc")
