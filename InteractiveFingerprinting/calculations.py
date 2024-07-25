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

class class_calculation:
    ''' Class object that computes the diagnostic indicators '''

    def __init__(self, xr_data):
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.xr_data = xr_data.sel(Scenario=self.settings['scenarios'],
                                   Model=self.settings['models'],
                                   Region=self.settings['regions'])
        self.percred = float(self.settings['params']['percred'])
        self.curpol = self.settings['params']['curpol_scenario']

    def calculate_responsiveness_indicators(self):
        # ============== #
        #       R1       #
        # ============== #
        self.dummy_xr = self.xr_data.assign(R1_rai = (self.xr_data.sel(Variable = self.settings['params']['emissionvar'],
                                                                      Scenario=self.curpol).Value -
                                                     self.xr_data.sel(Variable = self.settings['params']['emissionvar'],
                                                                      Scenario=self.settings['scenarios']).Value) /
                                                     self.xr_data.sel(Variable = self.settings['params']['emissionvar'],
                                                                       Scenario=self.curpol).Value)
        
        # ============== #
        #       R2       #
        # ============== #
        initial = np.array(self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                                Scenario=self.curpol,
                                                Time=np.arange(2020, 2101)).Value)
        series = np.array(self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                            Scenario=self.settings['scenarios'], 
                                            Time=np.arange(2020, 2101)).Value)
        years = np.zeros(shape=(len(self.settings['scenarios']),
                                len(self.settings['models']),
                                len(self.settings['regions'])))
        for m_i, m in enumerate(self.dummy_xr.Model):
            for s_i, s in enumerate(self.settings['scenarios']):
                wh_tot = np.where(series[m_i, s_i] < initial[m_i]*(1-self.percred))
                for r_i, r in enumerate(np.array(self.settings['regions'])):
                    
                    wh = wh_tot[1][wh_tot[0]==r_i]
                    if len(wh)>0:
                        years[s_i, m_i, r_i] = np.arange(2020, 2101)[wh[0]]-2020
                    else:
                        years[s_i, m_i, r_i] = np.nan
        self.years = years
        self.dummy_xr = self.dummy_xr.assign(R2_time = xr.DataArray(data=years,
                                                                    coords=dict(Scenario=self.settings['scenarios'],
                                                                                Model=self.settings['models'],
                                                                                Region=self.settings['regions']),))
        
        # ============== #
        #       R3       #
        # ============== #
        series = self.dummy_xr.sel(Variable = self.settings['params']['emissionvar'],
                                    Scenario=self.settings['scenarios'],
                                    Time=np.arange(2020, 2051)).Value
        speed = np.zeros(shape=(len(self.settings['scenarios']),
                                len(self.settings['models']),
                                len(self.settings['regions']),
                                len(np.arange(2020, 2101))))+np.nan
        potentials = np.zeros(shape=(len(self.settings['scenarios']),
                                        len(self.settings['models']),
                                        len(self.settings['regions'])))
        for m_i, m in enumerate(self.settings['models'][:2]):
            for s_i, s in enumerate(self.settings['scenarios'][:2]):
                for t_i, t in enumerate(np.arange(2020, 2051)):
                    if t_i != 0:
                        speed[s_i, m_i, :, t_i] = series[m_i, s_i, :, t_i-1] - series[m_i, s_i, :, t_i] #np.array(series.sel(Model=m, Scenario=s, Time=t-1) - series.sel(Model=m, Scenario=s, Time=t-1))#
                potentials[s_i, m_i] = np.nanmax(speed[s_i, m_i], axis=1)
        self.dummy_xr = self.dummy_xr.assign(R3_speedmax = xr.DataArray(data=potentials,
                                                                        coords=dict(Scenario=self.settings['scenarios'],
                                                                                    Model=self.settings['models'],
                                                                                    Region=self.settings['regions'])))
    
        # ============== #
        #       R4       #
        # ============== #
        energy_sources = ['Coal', 'Oil', 'Gas', 'Solar', 'Wind', 'Nuclear', 'Biomass']
        Ps = [self.dummy_xr.sel(Variable = "Primary Energy|"+i).Value / self.dummy_xr.sel(Variable = "Primary Energy").Value for i in energy_sources]
        var_prim = np.zeros(shape=(len(self.settings['models']),
                                   len(Ps),
                                   len(self.settings['regions']),
                                   len(np.arange(2020, 2101))))+np.nan
        for m_i, m in enumerate(self.dummy_xr.Model):
            for P_i, P in enumerate(Ps):
                var_prim[m_i, P_i] = (P.sel(Model=m,
                                            Scenario=self.settings['scenarios'],
                                            Time=np.arange(2020, 2101))).var(dim='Scenario')
        sens_prim = np.nanmean(var_prim, axis=1)
        self.dummy_xr = self.dummy_xr.assign(R4_sensprim = xr.DataArray(data=sens_prim,
                                                                        coords=dict(Model=self.settings['models'],
                                                                                    Region=self.settings['regions'],
                                                                                    Time=np.arange(2020, 2101))))
        
        # ============== #
        #       R5       #
        # ============== #
        E_ind = self.dummy_xr.sel(Variable = "Final Energy|Industry").Value
        E_trans = self.dummy_xr.sel(Variable = "Final Energy|Transportation").Value
        E_build = self.dummy_xr.sel(Variable = "Final Energy|Residential and Commercial").Value
        Es = [E_ind, E_trans, E_build]
        var_dem = np.zeros(shape=(len(self.settings['models']),
                                  len(Es),
                                  len(self.settings['regions']),
                                  len(np.arange(2020, 2101))))+np.nan
        for m_i, m in enumerate(self.dummy_xr.Model):
            for E_i, E in enumerate(Es):
                var_dem[m_i, E_i] = E.sel(Model=m,
                                           Scenario=self.settings['scenarios'],
                                           Time=np.arange(2020, 2101)).var(dim="Scenario")
        sens_dem = np.nanmean(var_dem, axis=1)
        self.dummy_xr = self.dummy_xr.assign(R5_sensdem = xr.DataArray(data=sens_dem, coords=dict(Model=self.settings['models'],
                                                                                                Region=self.settings['regions'],
                                                                                                Time=np.arange(2020, 2101))))

    def calculate_mitigationstrategy_indicators(self):
        # ============== #
        #       M1       #
        # ============== #
        CI = self.dummy_xr.sel(Variable = self.settings['params']['emissionvar']).Value / self.dummy_xr.sel(Variable = "Final Energy").Value
        self.dummy_xr = self.dummy_xr.assign(M1_cir = (CI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models'],
                                                              Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]) -
                                                       CI) /
                                                       CI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models'],
                                                              Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]))
        
        # ============== #
        #       M2       #
        # ============== #
        EI = self.dummy_xr.sel(Variable = "Final Energy").Value / self.dummy_xr.sel(Variable = "GDP|PPP").Value
        self.dummy_xr = self.dummy_xr.assign(M2_eir = (EI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models'],
                                                              Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]) -
                                                       EI) /
                                                       EI.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                              Model=self.settings['models'],
                                                              Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]))
        
        # ============== #
        #       M3       #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(M3_cc = self.dummy_xr.sel(Variable = "Carbon Capture").Value)
        
        # ============== #
        #       M4       #
        # ============== #
        nonco2 = self.dummy_xr.sel(Variable="Emissions|Kyoto Gases").Value - self.dummy_xr.sel(Variable="Emissions|CO2").Value
        co2 = self.dummy_xr.sel(Variable="Emissions|CO2").Value
        self.dummy_xr = self.dummy_xr.assign(M4_nonco2 = (nonco2.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                                     Model=self.settings['models'],
                                                                     Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]) -
                                                          nonco2.sel(Time=2050)) / (
                                                          co2.sel(Time = [2017, 2018, 2019, 2020, 2021],
                                                                  Model=self.settings['models'],
                                                                  Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]) - co2.sel(Time=2050)))
    
    def calculate_energysupply_indicators(self):
        # ============== #
        #    ES1-ES7     #
        # ============== #
        
        self.dummy_xr = self.dummy_xr.assign(ES1_coal = self.dummy_xr.sel(Variable = "Primary Energy|Coal", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES2_oil = self.dummy_xr.sel(Variable = "Primary Energy|Oil", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES3_gas = self.dummy_xr.sel(Variable = "Primary Energy|Gas", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES4_solar = self.dummy_xr.sel(Variable = "Primary Energy|Solar", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES5_wind = self.dummy_xr.sel(Variable = "Primary Energy|Wind", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES6_biomass = self.dummy_xr.sel(Variable = "Primary Energy|Biomass", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
        self.dummy_xr = self.dummy_xr.assign(ES7_nuclear = self.dummy_xr.sel(Variable = "Primary Energy|Nuclear", Scenario=self.settings['scenarios']).Value / self.dummy_xr.sel(Variable = "Primary Energy", Scenario=self.settings['scenarios']).Value)
    
    def calculate_energydemand_indicators(self):
        # ============== #
        #       ED1      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED1_etrans = (self.dummy_xr.sel(Variable = "Final Energy|Transportation|Electricity",
                                                                             Scenario=self.settings['scenarios']).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Transportation",
                                                                             Scenario=self.settings['scenarios']).Value)

        # ============== #
        #       ED2      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED2_eindus = (self.dummy_xr.sel(Variable = "Final Energy|Industry|Electricity",
                                                                             Scenario=self.settings['scenarios']).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Industry",
                                                                             Scenario=self.settings['scenarios']).Value)
        
        # ============== #
        #       ED3      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED3_ebuild = (self.dummy_xr.sel(Variable = "Final Energy|Residential and Commercial|Electricity",
                                                                             Scenario=self.settings['scenarios']).Value) /
                                                           self.dummy_xr.sel(Variable="Final Energy|Residential and Commercial",
                                                                             Scenario=self.settings['scenarios']).Value)
        
        # ============== #
        #       ED4      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED4_emise = (self.dummy_xr.sel(Variable = "Emissions|CO2|Energy|Supply|Electricity",
                                                                            Scenario=self.settings['scenarios']).Value))
        
        # ============== #
        #       ED5      #
        # ============== #
        self.dummy_xr = self.dummy_xr.assign(ED5_hydrogen = (self.dummy_xr.sel(Variable = "Final Energy|Hydrogen",
                                                                               Scenario=self.settings['scenarios']).Value) /
                                                             self.dummy_xr.sel(Variable="Final Energy",
                                                                               Scenario=self.settings['scenarios']).Value)
    
    def calculate_costandeffort_indicators(self):
        # ============== #
        #       C1       #
        # ============== #

        policyvars = ["Policy Cost|Consumption Loss",
                      "Policy Cost|Area under MAC Curve",
                      "Policy Cost|Additional Total Energy System Cost"]
        available_var = [x for x in policyvars if x in self.xr_data['Variable'].values]
        policy_cost_diff = (self.xr_data.sel(Variable=available_var,
                                                Time=np.arange(2020, 2051),
                                                Scenario=self.settings['scenarios']).sum(dim=["Time"],skipna=False) - 
                            self.xr_data.sel(Variable=available_var,
                                                Time=np.arange(2020, 2051),
                                                Scenario=self.settings['params']['curpol_scenario']).sum(dim=["Time"])).max(dim=['Variable'])

        self.dummy_xr = self.dummy_xr.assign(C1_cost = policy_cost_diff.Value /
                                                      (self.xr_data.sel(Variable="Price|Carbon",
                                                               Time=np.arange(2020, 2051),
                                                               Scenario=self.settings['scenarios']).Value.mean(dim="Time") *
                                                      (self.xr_data.sel(Variable=self.settings['params']['emissionvar'],
                                                               Time=2020,
                                                               Scenario=self.settings['scenarios']).Value -
                                                       self.xr_data.sel(Variable=self.settings['params']['emissionvar'],
                                                               Time=2050,
                                                               Scenario=self.settings['scenarios']).Value)))
        
        # ============== #
        #       C2       #
        # ============== #
        pes = ["Coal", "Oil", "Gas", "Solar", "Wind", "Nuclear", "Biomass"]
        fr_2020 = []
        fr_2050 = []
        for p_i, p in enumerate(pes):
            fr_2020.append(np.array(self.dummy_xr.sel(Variable="Primary Energy|"+p,
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models'],
                                                   Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]).Value /
                                 self.dummy_xr.sel(Variable="Primary Energy",
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models'],
                                                   Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]).Value))
            fr_2050.append(self.dummy_xr.sel(Variable="Primary Energy|"+p,
                                             Time = 2050,
                                             Scenario=self.settings['scenarios']).Value /
                           self.dummy_xr.sel(Variable="Primary Energy",
                                             Time = 2050,
                                             Scenario=self.settings['scenarios']).Value)
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
                                                   Model=self.settings['models'],
                                                   Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]).Value /
                                 self.dummy_xr.sel(Variable="Final Energy",
                                                   Time = [2017, 2018, 2019, 2020, 2021],
                                                   Model=self.settings['models'],
                                                   Scenario=self.settings['scenarios']).mean(dim=["Time", "Model", "Scenario"]).Value))
            fr_2050.append(self.dummy_xr.sel(Variable="Final Energy|"+p,
                                             Time = 2050,
                                             Scenario=self.settings['scenarios']).Value /
                           self.dummy_xr.sel(Variable="Final Energy",
                                             Time = 2050,
                                             Scenario=self.settings['scenarios']).Value)
            if p_i == 0:
                sumdif = np.abs(fr_2050[-1] - fr_2020[-1])
            else:
                sumdif += np.abs(fr_2050[-1] - fr_2020[-1])

        self.dummy_xr = self.dummy_xr.assign(C3_dem = sumdif)

    def convert_to_indicator_xr(self):
        self.dummy_xr = self.dummy_xr.sel(Scenario=self.settings['scenarios']).drop_vars(['Value', 'Variable'])
        self.pd_ind = self.dummy_xr.to_dataframe()
        self.pd_ind = self.pd_ind.reset_index()
        #self.pd_ind = self.pd_ind.drop(['Time'], axis=1)
        dummy_ = self.pd_ind.melt(id_vars=["Model", "Scenario", "Region", "Time"], var_name="Indicator", value_name="Value")
        dummy_ = dummy_.reset_index(drop=True)
        dummy_ = dummy_.set_index(["Model", "Scenario", "Region", "Indicator", "Time"])
        self.xr_ind = xr.Dataset.from_dataframe(dummy_)
        self.xr_ind.to_netcdf("Data/xr_indicators.nc")