# ======================================== #
# Class for the effort sharing work of ECEMF WP5
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT ON READING THE ECEMF CSV FILES 
# =========================================================== #

class ecemf_reader:
    def __init__(self):
        with open("../Configuration/config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.list_of_models = [self.settings['models'][m]['full_name'] for m in self.settings['models'].keys()]
        self.list_of_colors = [self.settings['models'][m]['color'] for m in self.settings['models'].keys()]
    
    def read_csv_files(self):
        self.pd_mod = pd.read_csv(self.settings['paths']['data_europe']+self.list_of_models[0]+'.csv')
        for m_i, m in enumerate(self.list_of_models[1:]):
            self.pd_mod = pd.concat([self.pd_mod, pd.read_csv(self.settings['paths']['data_europe']+m+'.csv')])
        self.pd_mod = self.pd_mod.reset_index(drop=True)
        self.pd_mod['Region'] = 'Europe'
        for m_i, m in enumerate(self.list_of_models):
            self.pd_mod = pd.concat([self.pd_mod, pd.read_csv(self.settings['paths']['data_world']+m+'.csv')])
        self.pd_mod = self.pd_mod.reset_index(drop=True)
    
    def add_average_gdps(self):
        # Remove GDP data from respective models (nulls/nans)
        for m_i, m in enumerate(self.settings['models_requiring_gdpav']):
            self.pd_mod = self.pd_mod.drop(self.pd_mod[(self.pd_mod.Variable == 'GDP|PPP') & (self.pd_mod.Model == m)].index)
            self.pd_mod = self.pd_mod.reset_index(drop=True)

        # Compute GDP averages per model
        av_gdp = np.array(self.pd_mod[self.pd_mod.Variable == 'GDP|PPP'])[:, 5:]
        time = self.pd_mod.keys()[5:].astype(int)
        mods_gdp = []
        av_gdp_permodel = {}
        all_gdpsw = []
        all_gdpse = []
        for m_i, m in enumerate(self.list_of_models):
            try:
                ag = (np.array(self.pd_mod[(self.pd_mod.Variable == 'GDP|PPP') & (self.pd_mod.Model == m) & (self.pd_mod.Region == 'Europe')])[:, 5:]).mean(axis=0)
                if np.nansum(ag) > 0:
                    av_gdp_permodel[m] = ag
                    mods_gdp.append(m)
                    all_gdpse.append(ag)
            except:
                3
            try:
                ag = (np.array(self.pd_mod[(self.pd_mod.Variable == 'GDP|PPP') & (self.pd_mod.Model == m) & (self.pd_mod.Region == 'World')])[:, 5:]).mean(axis=0)
                if np.nansum(ag) > 0:
                    all_gdpsw.append(ag)
            except:
                3
        
        # Overall GDP average
        av_gdp = np.zeros(shape = (len(all_gdpse[0]), 2))
        for t_i in range(len(all_gdpse[0])):
            av_gdp[t_i, 0] = np.nanmean(np.array(all_gdpse)[:, t_i].astype(float))
        for t_i in range(len(all_gdpse[0])):
            av_gdp[t_i, 1] = np.nanmean(np.array(all_gdpsw)[:, t_i].astype(float))

        # Add to dataframe
        rows = []
        for s_i, s in enumerate(self.settings['scenarios']):
            for m_i, m in enumerate(self.settings['models_requiring_gdpav']):
                rows.append([m, s, 'Europe', "GDP|PPP", "billion EUR_2020/yr"]+list(av_gdp.T[0]))
                rows.append([m, s, 'World', "GDP|PPP", "billion EUR_2020/yr"]+list(av_gdp.T[1]))
        df_rows = pd.DataFrame(rows, columns = ["Model", "Scenario", "Region", "Variable", "Unit"]+list(time.astype(str)))
        self.pd_mod = pd.concat([self.pd_mod, df_rows])
        self.pd_mod = self.pd_mod.reset_index(drop=True)
    
    def convert_to_xr(self):
        dummy_ = self.pd_mod.drop(['Unit'], axis=1)
        dummy_ = dummy_[dummy_.Scenario != '<dummy>']
        dummy_ = dummy_.melt(id_vars=["Model", "Scenario", "Variable", "Region"], var_name="Time", value_name="Value")
        dummy_['Time'] = np.array(dummy_['Time']).astype(int)
        dummy_ = dummy_.reset_index(drop=True)
        dummy_ = dummy_.set_index(["Model", "Scenario", "Variable", "Region", "Time"])
        self.data_xr = xr.Dataset.from_dataframe(dummy_)
        self.data_xr = self.data_xr.reindex(Time = np.arange(1995, 2101))
        self.data_xr = self.data_xr.interpolate_na(dim="Time", method="linear")
        self.data_xr.to_netcdf(self.settings['paths']['data'] + "XRdata.nc")

# =========================================================== #
# CLASS OBJECT ON CALCULATING THE DIAGNOSTIC INDICATORS
# =========================================================== #

class calculator:
    def __init__(self, data_xr):
        self.data_xr = data_xr
        with open("../Configuration/config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.list_of_models = [self.settings['models'][m]['full_name'] for m in self.settings['models'].keys()]
        self.list_of_colors = [self.settings['models'][m]['color'] for m in self.settings['models'].keys()]
        self.percred = float(self.settings['params']['percred'])
    
    def calculate_responsiveness_indicators(self):
        # ============== #
        #       S1       #
        # ============== #
        self.dummy_xr = self.data_xr.assign(S1_rai = (self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                      Scenario="DIAG-NPI",
                                                                      Time=2050).Value -
                                                     self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                      Scenario=self.settings['scenarios_c400'],
                                                                      Time=2050).Value) /
                                                     self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                       Scenario="DIAG-NPI",
                                                                       Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(S1s_rai = (self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                            Scenario="DIAG-NPI",
                                                                            Time=[2017, 2018, 2019, 2020, 2021]).Value.mean(dim='Time') -
                                                        self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                            Scenario=self.settings['scenarios_c400'],
                                                                            Time=2050).Value) /
                                                        self.data_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                                            Scenario="DIAG-NPI",
                                                                            Time=[2017, 2018, 2019, 2020, 2021]).Value.mean(dim='Time'))
        
        # ============== #
        #       S2       #
        # ============== #
        initial = np.array(self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                             Scenario="DIAG-NPI",
                                             Time=np.arange(2020, 2101)).Value)
        series = np.array(self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                            Scenario=self.settings['scenarios_c400'],
                                            Time=np.arange(2020, 2101)).Value)
        years = np.zeros(shape=(len(self.settings['scenarios_c400']), len(self.list_of_models)))
        for m_i, m in enumerate(self.dummy_xr.Model):
            for s_i, s in enumerate(self.settings['scenarios_c400']):
                wh = np.where(series[m_i, s_i] < initial[m_i]*(1-self.percred))[1]
                if len(wh)>0:
                    years[s_i, m_i] = np.arange(2020, 2101)[wh[0]]-2020
                else:
                    years[s_i, m_i] = np.nan
        self.years = years
        self.dummy_xr = self.dummy_xr.assign(S2_time = xr.DataArray(data=years,
                                                                    coords=dict(Scenario=self.settings['scenarios_c400'],
                                                                                Model=self.dummy_xr.Model)))
        
        # ============== #
        #       S3       #
        # ============== #
        series = np.array(self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                    Scenario=self.settings['scenarios_c400'],
                                    Time=np.arange(2020, 2051)).Value)
        speed = np.zeros(shape=(len(self.settings['scenarios_c400']),
                                len(self.list_of_models),
                                2,
                                len(np.arange(2020, 2101))))+np.nan
        potentials = np.zeros(shape=(len(self.settings['scenarios_c400']),
                                        len(self.list_of_models), 2))
        for m_i, m in enumerate(self.dummy_xr.Model):
            for s_i, s in enumerate(self.settings['scenarios_c400']):
                for t_i, t in enumerate(np.arange(2020, 2051)):
                    if t_i != 0:
                        speed[s_i, m_i, :, t_i] = series[m_i, s_i, :, t_i-1] - series[m_i, s_i, :, t_i]
                potentials[s_i, m_i] = np.nanmax(speed[s_i, m_i], axis=1)
        self.dummy_xr = self.dummy_xr.assign(S3_speedmax = xr.DataArray(data=potentials,
                                                                        coords=dict(Scenario=self.settings['scenarios_c400'],
                                                                                    Model=self.dummy_xr.Model,
                                                                                    Region=['Europe', 'World'])))
    
        # ============== #
        #       S4       #
        # ============== #
        energy_sources = ['Coal', 'Oil', 'Gas', 'Solar', 'Wind', 'Nuclear', 'Biomass']
        Ps = [self.dummy_xr.sel(Variable = "Primary Energy|"+i).Value / self.dummy_xr.sel(Variable = "Primary Energy").Value for i in energy_sources]
        var_prim = np.zeros(shape=(len(self.dummy_xr.Model),
                                   len(Ps),
                                   2,
                                   len(np.arange(2020, 2101))))+np.nan
        for m_i, m in enumerate(self.dummy_xr.Model):
            for P_i, P in enumerate(Ps):
                var_prim[m_i, P_i] = (P.sel(Model=m,
                                            Scenario=self.settings['scenarios_c400'],
                                            Time=np.arange(2020, 2101)) /
                                      P.sel(Model=m,
                                            Scenario=self.settings['scenarios_c400'],
                                            Time=np.arange(2020, 2101)).mean(dim="Scenario")).var(dim="Scenario")
        sens_prim = np.nanmean(var_prim, axis=1)
        self.dummy_xr = self.dummy_xr.assign(S4_sensprim = xr.DataArray(data=sens_prim,
                                                                        coords=dict(Model=self.dummy_xr.Model,
                                                                                    Region=['Europe', 'World'],
                                                                                    Time=np.arange(2020, 2101))))
        
        # ============== #
        #       S5       #
        # ============== #
        E_ind = self.dummy_xr.sel(Variable = "Final Energy|Industry").Value
        E_trans = self.dummy_xr.sel(Variable = "Final Energy|Transportation").Value
        E_build = self.dummy_xr.sel(Variable = "Final Energy|Residential and Commercial").Value
        Es = [E_ind, E_trans, E_build]
        var_dem = np.zeros(shape=(len(self.dummy_xr.Model), len(Es), 2, len(np.arange(2020, 2101))))+np.nan
        for m_i, m in enumerate(self.dummy_xr.Model):
            for E_i, E in enumerate(Es):
                var_dem[m_i, E_i] = (E.sel(Model=m,
                                           Scenario=self.settings['scenarios_c400'],
                                           Time=np.arange(2020, 2101)) /
                                     E.sel(Model=m,
                                           Scenario=self.settings['scenarios_c400'],
                                           Time=np.arange(2020, 2101)).mean(dim="Scenario")).var(dim="Scenario")
        sens_dem = np.nanmean(var_dem, axis=1)
        self.dummy_xr = self.dummy_xr.assign(S5_sensdem = xr.DataArray(data=sens_dem, coords=dict(Model=self.dummy_xr.Model,
                                                                                                  Region=['Europe', 'World'],
                                                                                                  Time=np.arange(2020, 2101))))

    def calculate_mitigationstrategy_indicators(self):
        # ============== #
        #       M1       #
        # ============== #
        CI = self.dummy_xr.sel(Variable = self.settings['params']['emissionvar']).Value / self.dummy_xr.sel(Variable = "Final Energy").Value
        self.dummy_xr = self.dummy_xr.assign(M1_cir = (CI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models_touse'],
                                                              Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]) -
                                                       CI.sel(Time = 2050)) /
                                                       CI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models_touse'],
                                                              Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]))
        
        # ============== #
        #       M2       #
        # ============== #
        EI = self.dummy_xr.sel(Variable = "Final Energy").Value / self.dummy_xr.sel(Variable = "GDP|PPP").Value
        self.dummy_xr = self.dummy_xr.assign(M2_eir = (EI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models_touse'],
                                                              Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]) -
                                                       EI.sel(Time = 2050)) /
                                                       EI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models_touse'],
                                                              Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]))
        
        # ============== #
        #       M3       #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(M3_cc = self.dummy_xr.sel(Variable = "Carbon Capture", Time=2050).Value)
        
        # ============== #
        #       M4       #
        # ============== #
        nonco2 = self.dummy_xr.sel(Variable="Emissions|Kyoto Gases").Value - self.dummy_xr.sel(Variable="Emissions|CO2").Value
        co2 = self.dummy_xr.sel(Variable="Emissions|CO2").Value
        self.dummy_xr = self.dummy_xr.assign(M4_nonco2 = (nonco2.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                                     Model=self.settings['models_touse'],
                                                                     Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]) -
                                                          nonco2.sel(Time=2050)) / (
                                                          co2.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                                  Model=self.settings['models_touse'],
                                                                  Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]) - co2.sel(Time=2050)))
    
    def calculate_energysupply_indicators(self):
        # ============== #
        #    ES1-ES7     #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ES1_coal = self.dummy_xr.sel(Variable = "Primary Energy|Coal", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES2_oil = self.dummy_xr.sel(Variable = "Primary Energy|Oil", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES3_gas = self.dummy_xr.sel(Variable = "Primary Energy|Gas", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES4_solar = self.dummy_xr.sel(Variable = "Primary Energy|Solar", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES5_wind = self.dummy_xr.sel(Variable = "Primary Energy|Wind", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES6_biomass = self.dummy_xr.sel(Variable = "Primary Energy|Biomass", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
        self.dummy_xr = self.dummy_xr.assign(ES7_nuclear = self.dummy_xr.sel(Variable = "Primary Energy|Nuclear", Scenario=self.settings['scenarios_c400'], Time=2050).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios_c400'], Time=2050).Value)
    
    def calculate_energydemand_indicators(self):
        # ============== #
        #       ED1      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED1_etrans = (self.dummy_xr.sel(Variable = "Final Energy|Transportation|Electricity",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Transportation",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value)

        # ============== #
        #       ED2      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED2_eindus = (self.dummy_xr.sel(Variable = "Final Energy|Industry|Electricity",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Industry",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value)
        
        # ============== #
        #       ED3      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED3_ebuild = (self.dummy_xr.sel(Variable = "Final Energy|Residential and Commercial|Electricity",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Residential and Commercial",
                                                                             Scenario=self.settings['scenarios_c400'],
                                                                             Time=2050).Value)
        
        # ============== #
        #       ED4      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED4_emise = (self.dummy_xr.sel(Variable = "Emissions|CO2|Energy|Supply|Electricity",
                                                                            Scenario=self.settings['scenarios_c400'],
                                                                            Time=2050).Value))
        
        # ============== #
        #       ED5      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED5_hydrogen = (self.dummy_xr.sel(Variable = "Final Energy|Hydrogen",
                                                                               Scenario=self.settings['scenarios_c400'],
                                                                               Time=2050).Value) /
                                                             self.dummy_xr.sel(Variable="Final Energy",
                                                                               Scenario=self.settings['scenarios_c400'],
                                                                               Time=2050).Value)
    
    def calculate_costandeffort_indicators(self):
        # ============== #
        #       C1       #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(C1_cost = self.dummy_xr.sel(Variable="Policy Cost|Consumption Loss",
                                                               Time=np.arange(2020, 2051),
                                                               Scenario=self.settings['scenarios_c400']).Value.sum(dim=["Time"], skipna=False) /
                                                      (self.dummy_xr.sel(Variable="Price|Carbon",
                                                               Time=np.arange(2020, 2051),
                                                               Scenario=self.settings['scenarios_c400']).Value.mean(dim="Time") *
                                                      (self.dummy_xr.sel(Variable=self.settings['params']['emissionvar'],
                                                               Time=2020,
                                                               Scenario=self.settings['scenarios_c400']).Value -
                                                       self.dummy_xr.sel(Variable=self.settings['params']['emissionvar'],
                                                               Time=2050,
                                                               Scenario=self.settings['scenarios_c400']).Value)))
        
        # ============== #
        #       C2       #
        # ============== #
        pes = ["Coal", "Oil", "Gas", "Solar", "Wind", "Nuclear", "Biomass"]
        fr_2020 = []
        fr_2050 = []
        for p_i, p in enumerate(pes):
            fr_2020.append(np.array(self.dummy_xr.sel(Variable="Primary Energy|"+p,
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models_touse'],
                                                   Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]).Value /
                                 self.dummy_xr.sel(Variable="Primary Energy",
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models_touse'],
                                                   Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]).Value))
            fr_2050.append(self.dummy_xr.sel(Variable="Primary Energy|"+p,
                                             Time = 2050,
                                             Scenario=self.settings['scenarios_c400']).Value /
                           self.dummy_xr.sel(Variable="Primary Energy",
                                             Time = 2050,
                                             Scenario=self.settings['scenarios_c400']).Value)
            if p_i == 0:
                sumdif = np.abs(fr_2050[-1] - fr_2020[-1])
            else:
                sumdif += np.abs(fr_2050[-1] - fr_2020[-1])

        self.dummy_xr = self.dummy_xr.assign(C2_ti = sumdif)
        
        # ============== #
        #       C3       #
        # ============== #
        pes = ["Industry", "Transportation", "Residential and Commercial"]
        fr_2020 = []
        fr_2050 = []
        for p_i, p in enumerate(pes):
            fr_2020.append(np.array(self.dummy_xr.sel(Variable="Final Energy|"+p,
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models_touse'],
                                                   Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]).Value /
                                 self.dummy_xr.sel(Variable="Final Energy",
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models_touse'],
                                                   Scenario=self.settings['scenarios_c400']).mean(dim=["Time", "Model", "Scenario"]).Value))
            fr_2050.append(self.dummy_xr.sel(Variable="Final Energy|"+p,
                                             Time = 2050,
                                             Scenario=self.settings['scenarios_c400']).Value /
                           self.dummy_xr.sel(Variable="Final Energy",
                                             Time = 2050,
                                             Scenario=self.settings['scenarios_c400']).Value)
            if p_i == 0:
                sumdif = np.abs(fr_2050[-1] - fr_2020[-1])
            else:
                sumdif += np.abs(fr_2050[-1] - fr_2020[-1])

        self.dummy_xr = self.dummy_xr.assign(C3_dem = sumdif)

    def convert_to_indicator_xr(self):
        self.dummy_xr = self.dummy_xr.sel(Scenario=self.settings['scenarios_c400']).drop_vars(['Value', 'Variable'])
        self.pd_ind = self.dummy_xr.to_dataframe()
        self.pd_ind = self.pd_ind.reset_index()
        #self.pd_ind = self.pd_ind.drop(['Time'], axis=1)
        dummy_ = self.pd_ind.melt(id_vars=["Model", "Scenario", "Region", "Time"], var_name="Indicator", value_name="Value")
        dummy_ = dummy_.reset_index(drop=True)
        dummy_ = dummy_.set_index(["Model", "Scenario", "Region", "Indicator", "Time"])
        self.ind_xr = xr.Dataset.from_dataframe(dummy_)
    
    def export_indicators_to_netcdf(self):
        self.ind_xr.to_netcdf(self.settings['paths']['data'] + "XRindicators.nc")
