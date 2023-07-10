# Functions for the Diagnostic Indicators by Harmsen
# Read in by DiagnosticIndicators.ipynb

# ========================================================================================================================== #
# REQUIRED PACKAGES
# ========================================================================================================================== #

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import configparser
from Functions import (add_legend_item,
                       get,
                       calc_relative_abatement_index,
                       calc_carbon_and_energy_intensity,
                       calc_normalised_carbon_and_energy_intensity,
                       confidence_ellipse,
                       calc_fossil_fuel_reduction,
                       linearInterp,
                       color,
                       cp80gr5,
                       set_value_from_var_column,
                       make_rows)

# ========================================================================================================================== #
# CLASS
# ========================================================================================================================== #

class Indicators(object):
    def __init__(self, params_input):
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Read paths
        self.Path_Data = config['PATHS']['Path_Data']
        self.Path_Dataw = config['PATHS']['Path_Dataw']
        self.Path_ModelDetails = config['PATHS']['Path_ModelDetails']
        self.Path_Figures = config['PATHS']['Path_Figures']

        # Read parameters
        self.Region = config['PARAMS']['Region']
        self.startyear = int(config['PARAMS']['startyear'])
        self.dt = int(config['PARAMS']['dt'])
        self.EmissionsVar = config['PARAMS']['var_CO2_FFI']
        self.gridcolor = config['PARAMS']['gridcolor']

        # Other settings
        self.ModelCols = ['black', 'gold', 'forestgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'silver', 'green']
        self.Timeline = [2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]

    def prepare_data(self):
        Models = ['IMAGE 3.2',              # 
                    'PROMETHEUS 1.2',         # 
                    'OSeMBE v1.0.0',          # country-level (sum?)
                    'REMIND 2.1',             # 
                    'WITCH 5.1',              # 
                    'Euro-Calliope 2.0',      # Country-level (sum?)
                    'MEESA v1.1',             # 
                    'TIAM-ECN 1.2',           # 
                    'MESSAGEix-GLOBIOM 1.2'   # 
                    ]
        self.Models = Models

        # Europe
        df = pd.read_csv('X:/user/dekkerm/Projects/ECEMF/WP1/Data/EuropeanData/'+Models[0]+'.csv')
        for i in range(1, len(Models)):
            df2 = pd.read_csv('X:/user/dekkerm/Projects/ECEMF/WP1/Data/EuropeanData/'+Models[i]+'.csv')
            df = pd.concat([df, df2])
        df = df.reset_index(drop=True)
        self.DF_raw = df

        # World
        dfw = pd.read_csv('X:/user/dekkerm/Projects/ECEMF/WP1/Data/WorldData/'+Models[0]+'.csv')
        for i in range(1, len(Models)):
            df2 = pd.read_csv('X:/user/dekkerm/Projects/ECEMF/WP1/Data/WorldData/'+Models[i]+'.csv')
            dfw = pd.concat([dfw, df2])
        dfw = dfw.reset_index(drop=True)
        self.DF_raw_world = dfw
        # Choose only decadal data, connect different Europe versions, set scenarios to lowercase and add column names
        #self.DF_raw = pd.read_csv(self.Path_Data, index_col=0)
        #self.DF_raw_world = pd.read_csv(self.Path_Dataw, index_col=0)
        #self.Data = self.DF_raw.loc[:,['Model', 'Scenario', 'Region', 'Variable', 'Unit']+[str(y) for y in self.Timeline]]
        #self.Dataw = self.DF_raw_world.loc[:,['Model', 'Scenario', 'Region', 'Variable', 'Unit']+[str(y) for y in self.Timeline]]
        #if self.Region == 'European':
        #    self.Data = self.Data[self.Data['Region'].isin(['Europe', 'Europe (excl. Turkey)', 'EU27 & UK', 'EU27', 'Europe (incl. Turkey)', 'all'])]
        #self.Data['Scenario'] = self.Data['Scenario'].str.lower()
        #self.Data.insert(2, 'Name', self.Data['Model'] + ' ' + self.Data['Scenario'])
        #self.Dataw['Scenario'] = self.Dataw['Scenario'].str.lower()
        #self.Dataw.insert(2, 'Name', self.Dataw['Model'] + ' ' + self.Dataw['Scenario'])
    
    def read_modelmeta(self):

        # Read model details
        self.DF_meta = pd.read_excel(self.Path_ModelDetails)
        meta = self.DF_meta.rename(columns={
            'Model_versionname': 'Model', 
            'Model_name': 'Stripped model', 
            'Age (1 = newest)': 'Age'
        }).set_index('Model')
        meta['Newest'] = meta['Age'] == 1

        # Check if all models from `data` are in `meta`:
        in_meta = set(meta.index)
        in_data = set(self.Data['Model'].unique())
        if len(in_meta - in_data):
            print('In meta, but not in data:', in_meta - in_data)
        if len(in_data - in_meta):
            print('In data, but not in meta:', in_data - in_meta)
            
        # Only keep those models which are present in the data
        meta = meta[meta.index.isin(self.Data['Model'])]
        models = (
            meta.reset_index()
            .groupby(['Type', 'Stripped model'])
            .first()['Model']
            .reset_index()
            .rename(columns={'Model': 'Full model'})
        )

        #COLORS_PBL = ['#00AEEF', '#808D1D', '#B6036C', '#FAAD1E', '#3F1464', '#7CCFF2', '#F198C1', '#42B649', '#EE2A23', '#004019', '#F47321', '#511607', '#BA8912', '#78CBBF', '#FFF229', '#0071BB']
        # all_colors = ['#0c2c84', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4'] + ['#86469c', '#bc7fcd', '#fbcfe8', '#ed66b2'] +['#FF7F0E','#FBE426']+['rgb(248, 156, 116)', '#D62728', '#AF0033', '#E48F72'] + COLORS_PBL
        #all_colors = [COLORS_PBL[j] if type(j) == int else j for j in [11,'#bc7fcd',2,6,4,15,0,5,13,7,1, '#BCBD22', 12, 8,10,3,14]]

        models['i'] = models.index
        models['Color'] = [self.ModelCols[i] for i in models['i']]
        self.Models = models.set_index('Stripped model')
        self.DF_ind = meta
        self.ModelList = list(self.DF_ind.index)
    
    def determine_indicator_values(self):

        # Prep dataframe
        self.DF_ind2 = pd.DataFrame(columns=['Model', 'PolicyScenario', 'Year'])
        for m in self.ModelList:
            for p in ['diag-c80-gr5']:
                for y in self.Timeline:
                    dic = {}
                    dic['Model'] = m
                    dic['PolicyScenario'] = p
                    dic['Year'] = y
                    self.DF_ind2 = self.DF_ind2.append(pd.DataFrame(dic, index=[100]))
        self.DF_ind2 = self.DF_ind2.reset_index(drop=True)

        # Speed up with arrays
        I_mod = np.array(self.DF_ind2.Model)
        D_scen = np.array(self.Data.Scenario)
        D_mod = np.array(self.Data.Model)
        D_var = np.array(self.Data.Variable)

        # Add indicators
        RAI = np.zeros(len(self.DF_ind2))
        CP = np.zeros(len(self.DF_ind2))
        CInorm = np.zeros(len(self.DF_ind2))
        EInorm = np.zeros(len(self.DF_ind2))
        CIoverEI = np.zeros(len(self.DF_ind2))
        PrimEnvars = np.zeros(shape=(9, len(self.DF_ind2)))
        PrimEnvarsw = np.zeros(shape=(9, len(self.DF_ind2)))
        FFR = np.zeros(len(self.DF_ind2))
        Gap = np.zeros(len(self.DF_ind2))
        Cum = np.zeros(len(self.DF_ind2))
        IT = np.zeros(len(self.DF_ind2))
        CAV = np.zeros(len(self.DF_ind2))
        CostperGDP = np.zeros(len(self.DF_ind2))
        for m in range(len(self.ModelList)):
            model = self.ModelList[m]

            # RAI
            emissions_base = np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == self.EmissionsVar)][self.Data.keys()[6:]])[0]
            emissions_c80 = np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == self.EmissionsVar)][self.Data.keys()[6:]])[0]
            dat = (emissions_base-emissions_c80)/(1e-9+emissions_base)
            dat[emissions_base == 0] = 0
            RAI[I_mod == model] = dat
            
            # Carbon price
            carbonprice = np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Price|Carbon')][self.Data.keys()[6:]])[0]
            CP[I_mod == model] = carbonprice

            # CI norm
            finalenergy_base =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == 'Final Energy')][self.Data.keys()[6:]])[0]
            finalenergy_c80 =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Final Energy')][self.Data.keys()[6:]])[0]
            CI_base = emissions_base / (1e-9+finalenergy_base)
            CI_base[finalenergy_base == 0] = 0
            CI_c80 = emissions_c80 / (1e-9+finalenergy_c80)
            CI_c80[finalenergy_c80 == 0] = 0
            CI_norm = CI_c80/(1e-9+CI_base)
            CI_norm[CI_base == 0] = 0
            CInorm[I_mod == model] = 1-CI_norm

            # EI norm
            GDP_base =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == 'GDP|PPP')][self.Data.keys()[6:]])[0]
            GDP_c80 =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'GDP|PPP')][self.Data.keys()[6:]])[0]
            EI_base = finalenergy_base / (1e-9+GDP_base)
            EI_base[GDP_base == 0] = 0
            EI_c80 = finalenergy_c80 / (1e-9+GDP_c80)
            EI_c80[GDP_c80 == 0] = 0
            EI_norm = EI_c80/(1e-9+EI_base)
            EI_norm[EI_base == 0] = 0
            EInorm[I_mod == model] = 1-EI_norm

            # CI over EI
            CIoverEI[I_mod == model] = (1-CI_norm)/(1-CI_norm+1-EI_norm)

            # Decomposition primary energy
            vars = ['Coal', 'Gas', 'Oil', 'Nuclear', 'Biomass', 'Hydro', 'Solar', 'Wind', 'Other']
            for i in range(len(vars)):
                var_suffix = vars[i]
                var = f'Primary Energy|{var_suffix}'
                var_c80 = np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == var)][self.Data.keys()[6:]])[0]
                PrimEnvars[i, I_mod == model] = var_c80
                var_c80w = np.array(self.Dataw[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == var)][self.Data.keys()[6:]])[0]
                PrimEnvarsw[i, I_mod == model] = var_c80w

            # FFR
            fossil_c80_coal =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Primary Energy|Coal')][self.Data.keys()[6:]])[0]
            fossil_c80_oil =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Primary Energy|Oil')][self.Data.keys()[6:]])[0]
            fossil_c80_gas =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Primary Energy|Gas')][self.Data.keys()[6:]])[0]
            fossil_c80 = fossil_c80_coal+fossil_c80_oil+fossil_c80_gas
            fossil_base_coal =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == 'Primary Energy|Coal')][self.Data.keys()[6:]])[0]
            fossil_base_oil =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == 'Primary Energy|Oil')][self.Data.keys()[6:]])[0]
            fossil_base_gas =  np.array(self.Data[(D_mod == model) & (D_scen == 'diag-base') & (D_var == 'Primary Energy|Gas')][self.Data.keys()[6:]])[0]
            fossil_base = fossil_base_coal+fossil_base_oil+fossil_base_gas
            FFR_ = (fossil_base-fossil_c80)/(1e-9+fossil_base)
            FFR_[fossil_base == 0] = 0
            FFR[I_mod == model] = FFR_

            # Inertia timescale
            emissions_c0to80 = np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c0to80-gr5') & (D_var == self.EmissionsVar)][self.Data.keys()[6:]])[0]
            diff_emissions = ((emissions_c0to80 - emissions_c80)).clip(min=0)
            cumulative_excess = np.zeros(len(self.Timeline))
            for i in range(6, len(self.Timeline)):
                if i != 6:
                    x = np.copy(self.Timeline)[6:i+1]
                    y = diff_emissions[6:i+1]
                    xnew = np.arange(x[0], x[-1]+1)
                    f = interp1d(x, y)
                    cumulative_excess[i] = np.sum(f(xnew))
                else:
                    cumulative_excess[i] = diff_emissions[6]
            diff_emissions = diff_emissions * 0.001
            cumulative_excess = cumulative_excess * 0.001
            Cum[I_mod == model] = cumulative_excess
            Gap[I_mod == model] = diff_emissions
            IT_ =  cumulative_excess / (1e-9+diff_emissions[6])
            IT_[diff_emissions == 0] = 0
            IT[I_mod == model] = IT_

            # CAV
            policycost_c80 = np.abs(np.array(self.Data[(D_mod == model) & (D_scen == 'diag-c80-gr5') & (D_var == 'Policy Cost|Consumption Loss')][self.Data.keys()[6:]])[0])
            costpergdp_c80 = policycost_c80/(1e-9+GDP_c80)
            costpergdp_c80[GDP_c80 == 0] = 0
            CostperGDP[I_mod == model] = costpergdp_c80
            absreduction_c80 = (emissions_base - emissions_c80) * 0.001
            CAV_c80 = policycost_c80 / (1e-9+absreduction_c80*carbonprice)
            CAV_c80[absreduction_c80*carbonprice == 0] = 0
            CAV[I_mod == model] = CAV_c80
        self.DF_ind2['RAI'] = RAI
        self.DF_ind2['Carbon Price'] = CP
        self.DF_ind2['CInorm'] = CInorm
        self.DF_ind2['EInorm'] = EInorm
        self.DF_ind2['CIoverEI'] = CIoverEI
        for v in range(len(vars)):
            var = vars[v]
            self.DF_ind2['Primary Energy|'+var] = PrimEnvars[v]
            self.DF_ind2['Primary Energy (world)|'+var] = PrimEnvarsw[v]
        self.DF_ind2['RAI'] = RAI
        self.DF_ind2['FFR'] = FFR
        self.DF_ind2['IT'] = IT
        self.DF_ind2['Gap'] = Gap
        self.DF_ind2['CumulativeExcess'] = Cum
        self.DF_ind2['CostperGDP'] = CostperGDP
        self.DF_ind2['CAV'] = CAV
        self.DF_ind2 = self.DF_ind2.reset_index(drop=True)

        # RAI
        rais = []
        for year in self.Timeline:
            year = str(year)
            var_name='CO2 FFI'
            v1 = f'RAI c80 {year} {var_name}'
            v2 = f'Carbon price c80 {year}'
            rai = calc_relative_abatement_index(self.Data, year, pol='diag-c80-gr5', var=self.EmissionsVar)
            self.DF_ind[v1] = rai
            self.DF_ind[v2] = get(self.Data, 'diag-c80-gr5', 'Price|Carbon', year)
            rais.append(rai)
            #self.DF_ind.loc[self.DF_ind[v2] == 0, v2] = np.nan
        rais = np.array(rais)

        # CI en EI
        for year in self.Timeline:
            cib, eib = calc_normalised_carbon_and_energy_intensity(self.Data, str(year), self.DF_ind, self.EmissionsVar)
            self.DF_ind[f'Carbon intensity {year}'], self.DF_ind[f'Energy intensity {year}'] = 1-cib, 1-eib
            self.DF_ind[f'CoEI {year}'] = (1-cib) / (1-cib + 1-eib)

        # FFR and decomposition
        policy_scenario = 'diag-c80-gr5'
        year = '2050'
        variables = ['Fossil|w/o CCS', 'Fossil|w/ CCS', 'Nuclear', 'Biomass|w/o CCS', 'Biomass|w/ CCS', 'Non-Biomass Renewables']
        labels = ['Fossil<br>w/o CCS', 'Fossil<br>w. CCS', 'Nuclear', 'Biomass<br>w/o CCS', 'Biomass<br>w. CCS', 'Renewables']
        for var_suffix in variables:
            var = f'Primary Energy|{var_suffix}'
            self.DF_ind[f'{var} {year}'] = get(self.Data, policy_scenario, var, year)

        # Replace w/o CCS variables by full variable if w/ CCS data is missing:
        for fuel in ['Fossil', 'Biomass']:
            var = f'Primary Energy|{fuel}'
            self.DF_ind[f'{var} {year}'] = get(self.Data, policy_scenario, var, year)
            missing = self.DF_ind[f'{var}|w/ CCS {year}'].isna()
            self.DF_ind.loc[missing, f'{var}|w/o CCS {year}'] = self.DF_ind.loc[missing, f'{var} {year}']

        policy_scenario = 'diag-c80-gr5'
        col_FFR = f'FFR {year}'
        self.DF_ind[col_FFR] = calc_fossil_fuel_reduction(self.Data, '2050', policy_scenario)

        # Calculate inertia timescale
        diff2shock = get(self.Data, 'diag-c0to80-gr5', self.EmissionsVar) - get(self.Data, 'diag-c80-gr5', self.EmissionsVar)
        diff2shock[diff2shock > 1e6] = np.nan
        for year in self.Timeline[6:]:
            year = str(year)
            gap_emissions = diff2shock[year] * 0.001
            cumulative_excess_emissions = diff2shock.clip(lower=0).loc[:, [str(i) for i in range(2040, int(year)+1, 10)]].apply(linearInterp, axis=1) * 0.001
            self.DF_ind[f'Excess emissions c80 {year}'] = cumulative_excess_emissions
            self.DF_ind[f'Gap emissions c80 {year}'] = gap_emissions
            self.DF_ind[f'IT c80 {year}'] = (self.DF_ind[f'Excess emissions c80 {year}'] / (1e-9+self.DF_ind[f'Gap emissions c80 2040'])).clip(lower=0)

        # CAV
        for year in self.Timeline:
            cprice = cp80gr5(year)
            year = str(year)
            set_value_from_var_column(self.Data, self.DF_ind, 'Policy cost variable', f'Policy cost {year}', year, policy_scenario)
            set_value_from_var_column(self.Data, self.DF_ind, 'Emissions_for_CAV', f'Emissions CAV {year}', year, policy_scenario)
            set_value_from_var_column(self.Data, self.DF_ind, 'Emissions_for_CAV', f'Emissions CAV {year} base', year, 'diag-base')

            # Make all costs positive
            self.DF_ind[f'Policy cost {year}'] = self.DF_ind[f'Policy cost {year}'].abs()
            self.DF_ind[f'Policy cost {year}'][self.DF_ind.index == 'PROMETHEUS 1.2'] = self.DF_ind[f'Policy cost {year}'][self.DF_ind.index == 'PROMETHEUS 1.2']/1000
            
            # Cost per GDP
            GDP_column = f'GDP {year} {policy_scenario}'
            # Check if GDP column already exists
            #if GDP_column not in self.DF_ind.columns:
            #    set_value_from_var_column(self.Data, self.DF_ind, 'GDP_metric', GDP_column, year, policy_scenario)
            gdps = np.array(self.DF_ind[f'GDP {year} {policy_scenario}'])
            gdps[np.isnan(gdps)] = np.nanmedian(gdps)
            gdps[2] = gdps[2]/1000

            self.DF_ind[f'Policy cost {year} per GDP'] = self.DF_ind[f'Policy cost {year}'] / (1e-9+gdps)

            # Using this information, calculate CAV
        GHG_reduction_absolute = (self.DF_ind[f'Emissions CAV 2050 base'] - self.DF_ind[f'Emissions CAV 2050']) / 1000 # Convert Mt to Gt
        self.DF_ind[f'CAV 2050'] = self.DF_ind[f'Policy cost 2050'] / (1e-9+GHG_reduction_absolute * cp80gr5(2050))

    def fig_rai(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('(a) Carbon price vs RAI', '(b) RAI per model'), horizontal_spacing = 0.2, column_widths=[0.6, 0.4])
        fig.update_xaxes(title_text="Relative Abatement Index", row=1, col=1, range=[0, 1], showgrid=False)
        fig.update_xaxes(title_text="Relative Abatement Index in 2050", row=1, col=2, range=[0.3, 1])
        fig.update_yaxes(row=1, col=2, tickfont_family="Arial Black", tickfont_size = 14)
        fig.update_yaxes(title_text='Carbon price (US dollars/tCO2)', row=1, col=1, range=[0, 150], showgrid=False)
        #fig.update_layout(yaxis=dict(tickmode='array', ticktext=ticktext, tickvals=ticks))
        self.ModelCols = ['black', 'orange', 'darkgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'grey', 'turquoise']

        v1list = []
        v2list = []
        for year in self.Timeline[2:9]:
            fig.add_scatter(x=[0.95], y=[cp80gr5(year)*1.04], text=year, textposition='top center',
                            row=1, col=1, mode='text', showlegend=False)
            fig.add_hline(cp80gr5(year)*1.04, line=dict(color='black', width=0.4, dash='dash'))
            year = str(year)
            var_name='CO2 FFI'
            v1 = f'RAI c80 {year} {var_name}'
            v2 = f'Carbon price c80 {year}'
            v1list.append(v1)
            v2list.append(v2)
        for m in range(len(self.ModelList)):
            model = self.ModelList[m]
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1list])[0]
            if model == 'WITCH 5.0':
                model = 'WITCH 5.0'
            else:
                model = 'PROMETHEUS 1.2'
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2list])[0]
            nonans = np.where((~np.isnan(x)) | (~np.isnan(y)))[0]
            x = x[nonans]
            y = y[nonans]
            fig.add_scatter(x=x, y=y,
                        marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 12, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='lines+markers', showlegend=False)
        #fig.add_scatter(x=[0.1]*len(np.arange(2025, 2051)), y=cp80gr5(np.arange(2025, 2051)), line_color='silver')

        fig.add_scatter(x=list(self.DF_ind['RAI c80 2050 CO2 FFI']), y=[color(self.ModelCols[i], self.DF_ind.index[i]) for i in range(len(self.DF_ind.index))], text=list(self.DF_ind['RAI c80 2050 CO2 FFI'].round(2)),
                        textposition='top center', textfont=dict(color=self.ModelCols),
                        row=1, col=2, marker={'color': self.ModelCols, 'opacity': 1, 'symbol': 'star', 'size': 18, 'line': {'color': '#FFF', 'width': 1}},
                        mode='markers+text', showlegend=False)
        fig.add_vline(np.nanmean(self.DF_ind['RAI c80 2050 CO2 FFI']), line=dict(color='black', width=1, dash='dot'), row=1, col=2)

        fig.update_layout(height=600, width=1200)
        fig.show()

    def fig_ciei(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('(a) Energy Intensity vs Carbon Intensity', '(b) ERT per model'), horizontal_spacing = 0.2, column_widths=[0.6, 0.4])
        fig.update_xaxes(title_text="Carbon intensity (red. from baseline)", row=1, col=1, range=[0, 1.8], showgrid=False)
        fig.update_xaxes(title_text="CI over (EI+CI) (red. from baseline)", row=1, col=2, range=[0.3, 1])
        fig.update_yaxes(row=1, col=2, tickfont_family="Arial Black", tickfont_size = 14)
        fig.update_yaxes(title_text='Energy intensity (red. from baseline)', row=1, col=1, range=[0, 0.4], showgrid=False)
        self.ModelCols = ['black', 'orange', 'darkgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'grey', 'turquoise']
        v1list = []
        v2list = []
        v1blist = []
        v2blist = []
        for year in self.Timeline:
            year = str(year)
            v1 = f'Carbon intensity {year}'
            v2 = f'Energy intensity {year}'
            v3 = f'CoEI {year}'
            v1list.append(v1)
            v2list.append(v2)
            if year in ['2050', '2100']:
                v1blist.append(v1)
                v2blist.append(v2)
        for m in range(len(self.ModelList)):
            model = self.ModelList[m]
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1list])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2list])[0]
            nonans = np.where((~np.isnan(x)) | (~np.isnan(y)))[0]
            x = x[nonans]
            y = y[nonans]
            fig.add_scatter(x=x, y=y,
                        marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 6, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='lines+markers', showlegend=False)
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1blist])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2blist])[0]
            fig.add_scatter(x=x, y=y, text=['2050', '2100'],
                        textposition='top center', marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 15, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='markers+text', showlegend=False)
        fig.add_scatter(x=np.arange(-1, 1, 0.01), y=np.arange(-1, 1, 0.01), mode='lines', line=dict(color='black', width=1, dash='dot'), showlegend=False)

        fig.add_scatter(x=list(self.DF_ind['CoEI 2050']), y=[color(self.ModelCols[i], self.DF_ind.index[i]) for i in range(len(self.DF_ind.index))], text=list(self.DF_ind['CoEI 2050'].round(2)),
                        textposition='top center', textfont=dict(color=self.ModelCols),
                        row=1, col=2, marker={'color': self.ModelCols, 'opacity': 1, 'symbol': 'star', 'size': 18, 'line': {'color': '#FFF', 'width': 1}},
                        mode='markers+text', showlegend=False)
        fig.add_vline(np.nanmean(self.DF_ind['CoEI 2050']), line=dict(color='black', width=1, dash='dot'), row=1, col=2)
        fig.update_layout(height=600, width=1200)
        fig.show()

    def fig_ffr(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('(a) Primary Energy decomposition (2050)', '(b) FFR per model (2050)'), horizontal_spacing = 0.2, column_widths=[0.6, 0.4])
        fig.update_xaxes(title_text="", row=1, col=1, showgrid=False)
        fig.update_xaxes(title_text="Primary Energy (EJ in 2050)", row=1, col=2, range=[0.02, 1])
        fig.update_yaxes(row=1, col=2, tickfont_family="Arial Black", tickfont_size=14)
        fig.update_yaxes(title_text='Energy intensity (red. from baseline)', row=1, col=1, showgrid=False)
        self.ModelCols = ['black', 'orange', 'darkgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'grey', 'turquoise']

        variables = ['Fossil|w/o CCS', 'Fossil|w/ CCS', 'Nuclear', 'Biomass|w/o CCS', 'Biomass|w/ CCS', 'Non-Biomass Renewables']
        labels = ['Fossil<br>w/o CCS', 'Fossil<br>w. CCS', 'Nuclear', 'Biomass<br>w/o CCS', 'Biomass<br>w. CCS', 'Renewables']
        energy_colors = ['#d62728', '#d67a7a', '#ff7f0e', '#2ca02c', '#96d096', '#1f77b4']

        year = '2050'
        xpos = -0.05
        for var_suffix, label, colr, labelwidth in zip(variables, labels, energy_colors, [0.09, 0.09, 0.1, 0.11, 0.11, 0.1]):
            col = f'Primary Energy|{var_suffix} {year}'
            energy_values = self.DF_ind[col]

            var_values = self.Models.merge(
                energy_values.reset_index(),
                how='left', left_on='Full model', right_on='Model'
            )
            fig.add_bar(
                y=np.arange(9), x=np.array(var_values[col])[[np.where(var_values == list(self.DF_ind.index)[i])[0][0] for i in range(9)]],
                orientation='h', marker_color=colr,
                showlegend=False
            )

            # Add legend
            height, width = 0.04, 0.04 * 400 / 840
            ypos=-0.1
            fig.add_shape(
                type='rect',
                xref='paper', yref='paper',
                x0=xpos-width/2, x1=xpos+width/2, y0=ypos-height/2, y1=ypos+height/2,
                fillcolor=colr, line_width=0
            )
            fig.add_annotation(
                xref='paper', yref='paper', x=xpos+0.6*width, y=ypos,
                yanchor='top', yshift=12, xanchor='left', align='left',
                showarrow=False, text=label
            )
            xpos += labelwidth

        def color(color, text):
            return f"<span style='color:{str(color)}'> {str(text)} </span>"

        fig.add_scatter(x=list(self.DF_ind['FFR 2050']), y=[color(self.ModelCols[i], self.DF_ind.index[i]) for i in range(len(self.DF_ind.index))], text=list(self.DF_ind['FFR 2050'].round(2)),
                        textposition='top center', textfont=dict(color=self.ModelCols),
                        row=1, col=2, marker={'color': self.ModelCols, 'opacity': 1, 'symbol': 'star', 'size': 18, 'line': {'color': '#FFF', 'width': 1}},
                        mode='markers+text', showlegend=False)
        fig.add_vline(np.nanmean(self.DF_ind['FFR 2050']), line=dict(color='black', width=1, dash='dot'), row=1, col=2)

        fig.update_layout(height=600, width=1200)
        (fig.update_xaxes(col=2,
                    tickvals=np.arange(-0.2, 2, 0.2),
                    title='FFR',
                    title_standoff=40
                )
                .update_layout(
                    barmode='stack',
                    width=1200,
                    height=600,
                    margin={'l': 60, 'r': 30, 't': 50, 'b': 90},
                    hovermode='closest'
                )
                .update_xaxes(
                    col=1,
                    title=f'Primary Energy (EJ in {year})',
                    title_standoff=60
                )
            )
        fig.show()

    def fig_inertia(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('(a) Emissions gap vs excess emissions', '(b) Inertia Timescale per model'), horizontal_spacing = 0.2, column_widths=[0.6, 0.4])
        fig.update_xaxes(title_text="Cumulative excess emissions (GtCO2)", row=1, col=1, range=(-1, 1.5), showgrid=False, type='log')
        fig.update_yaxes(title_text='Emissions difference (GtCO2/yr)', row=1, col=1, range=(-3, 0.5), showgrid=False, type='log')
        fig.update_xaxes(title_text="Inertia Timescale (years)", row=1, col=2, range=[-3, 35])
        fig.update_yaxes(row=1, col=2, tickfont_family="Arial Black", tickfont_size = 14)
        self.ModelCols = ['black', 'orange', 'darkgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'grey', 'turquoise']
        def color(color, text):
            return f"<span style='color:{str(color)}'> {str(text)} </span>"
        def cp80gr5(t):
            return 80*1.05**(t-2040)

        v1list = []
        v2list = []
        v1blist = []
        v2blist = []
        for year in self.Timeline[6:]:
            year = str(year)
            v1 = f'Excess emissions c80 {year}'
            v2 = f'Gap emissions c80 {year}'
            v1list.append(v1)
            v2list.append(v2)
            if year in ['2050', '2100']:
                v1blist.append(v1)
                v2blist.append(v2)
        for m in range(len(self.ModelList)):
            model = self.ModelList[m]
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1list])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2list])[0]
            nonans = np.where((~np.isnan(x)) | (~np.isnan(y)))[0]
            x = x[nonans]
            y = y[nonans]
            fig.add_scatter(x=x, y=y,
                        marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 6, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='lines+markers', showlegend=False)
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1blist])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2blist])[0]
            fig.add_scatter(x=x, y=y, text=['2050', '2100'],
                        textposition='top center', marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 15, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='markers+text', showlegend=False)

        fig.add_scatter(x=list(self.DF_ind['IT c80 2100']), y=[color(self.ModelCols[i], self.DF_ind.index[i]) for i in range(len(self.DF_ind.index))], text=[str(list(self.DF_ind['IT c80 2100'].round(1))[i])+' years' for i in range(9)],
                        textposition='top center', textfont=dict(color=self.ModelCols),
                        row=1, col=2, marker={'color': self.ModelCols, 'opacity': 1, 'symbol': 'star', 'size': 18, 'line': {'color': '#FFF', 'width': 1}},
                        mode='markers+text', showlegend=False)
        fig.add_vline(np.nanmean(self.DF_ind['IT c80 2100']), line=dict(color='black', width=1, dash='dot'), row=1, col=2)

        fig.update_layout(height=600, width=1200)
        fig.show()

    def fig_cav(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('(a) Costs vs abatement', '(b) CAV per model (2050)'), horizontal_spacing = 0.2, column_widths=[0.6, 0.4])
        fig.update_xaxes(title_text="Policy costs (perc of GDP)", row=1, col=1, showgrid=False)
        fig.update_xaxes(title_text="CAV", row=1, range=(0, 1.4), col=2)
        fig.update_yaxes(row=1, col=2, tickfont_family="Arial Black", tickfont_size = 14)
        fig.update_yaxes(title_text='Relative abatement index', row=1, col=1, showgrid=False)
        self.ModelCols = ['black', 'orange', 'darkgreen', 'tomato', 'brown', 'steelblue', 'magenta', 'grey', 'turquoise']
        v1list = []
        v2list = []
        v1blist = []
        v2blist = []
        for year in self.Timeline:
            year = str(year)
            v1 = f'Policy cost {year} per GDP'
            v2 = f'RAI c80 {year} CO2 FFI'
            v1list.append(v1)
            v2list.append(v2)
            if year in ['2050', '2100']:
                v1blist.append(v1)
                v2blist.append(v2)
        for m in range(len(self.ModelList)):
            model = self.ModelList[m]
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1list])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2list])[0]
            nonans = np.where((~np.isnan(x)) | (~np.isnan(y)))[0]
            x = x[nonans]
            y = y[nonans]
            fig.add_scatter(x=x, y=y,
                        marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 6, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='lines+markers', showlegend=False)
            x = np.array(self.DF_ind[self.DF_ind.index == model][v1blist])[0]
            y = np.array(self.DF_ind[self.DF_ind.index == model][v2blist])[0]
            fig.add_scatter(x=x, y=y, text=['2050', '2100'],
                        textposition='top center', marker={'color': self.ModelCols[m], 'opacity': 1, 'symbol': 'circle', 'size': 15, 'line': {'color': '#FFF', 'width': 1}},
                        row=1, col=1, mode='markers+text', showlegend=False)

        fig.add_scatter(x=list(self.DF_ind['CAV 2050']), y=[color(self.ModelCols[i], self.DF_ind.index[i]) for i in range(len(self.DF_ind.index))], text=list(self.DF_ind['CAV 2050'].round(2)),
                        textposition='top center', textfont=dict(color=self.ModelCols),
                        row=1, col=2, marker={'color': self.ModelCols, 'opacity': 1, 'symbol': 'star', 'size': 18, 'line': {'color': '#FFF', 'width': 1}},
                        mode='markers+text', showlegend=False)
        fig.add_vline(np.nanmean(self.DF_ind['CAV 2050']), line=dict(color='black', width=1, dash='dot'), row=1, col=2)
        fig.update_layout(height=600, width=1200)
        fig.show()