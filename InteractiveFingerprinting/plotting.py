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
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class class_plotting:
    ''' Class object that does the plotting'''

    def __init__(self):
        print('STARTING PLOTTING')
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.xr_data = xr.open_dataset('Data/xr_variables.nc').sel(Region=self.settings['regions'])
        self.xr_ind = xr.open_dataset('Data/xr_indicators.nc').sel(Region=self.settings['regions'])
        self.models_all = self.xr_data.Model.values
        self.models_ref = self.settings['models']
        # find the model that is in self.models_all but not in self.models_ref
        self.model_ind = [x for x in self.models_all if x not in self.models_ref][0]
        filename = 'MyScenario_rahel.csv'
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
        self.key_region = str(self.xr_data.Region[np.where(~np.isnan(self.xr_data.sel(Model=df.Model[0]).Value))[1]][0].values)
    
    def plot_variables(self):
        print('- Plotting variables into /Figures/VariableData.html')
        scen = 'ELV-SSP2-NDC-D0'
        start_var = "Primary Energy|Wind"
        available_var = np.array([x for x in self.settings['required_variables'] if x in self.xr_data['Variable'].values])

        fig = make_subplots(rows=3, cols=4,
                            subplot_titles=[f'{r}' for r in np.array(self.xr_data.Region)],
                            horizontal_spacing = 0.04, vertical_spacing=0.10)

        for m_i, m in enumerate(self.models_ref):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=self.xr_data.sel(Variable=start_var,
                                                            Region=r,
                                                            Model=m,
                                                            Scenario="ELV-SSP2-CP-D0").Value,
                                        showlegend=False,
                                        line=dict(dash='dash', color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=int(np.floor(r_i/4)+1),
                                        col=int(1+np.mod(r_i, 4)))
            
        for r_i, r in enumerate(np.array(self.xr_data.Region)):
            fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                    y=self.xr_data.sel(Variable=start_var,
                                                        Region=r,
                                                        Model=self.model_ind,
                                                        Scenario="ELV-SSP2-CP-D0").Value,
                                    showlegend=False,
                                    line=dict(width=5, dash='dash', color ='black'),
                                    name=self.model_ind),
                                    row=int(np.floor(r_i/4)+1),
                                    col=int(1+np.mod(r_i, 4)))

        for m_i, m in enumerate(self.models_ref):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=self.xr_data.sel(Variable=start_var,
                                                            Region=r,
                                                            Model=m,
                                                            Scenario=scen).Value,
                                        showlegend=([True]+[False]*10)[r_i],
                                        line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=int(np.floor(r_i/4)+1),
                                        col=int(1+np.mod(r_i, 4)))
                
        for r_i, r in enumerate(np.array(self.xr_data.Region)):
            fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                    y=self.xr_data.sel(Variable=start_var,
                                                        Region=r,
                                                        Model=self.model_ind,
                                                        Scenario=scen).Value,
                                    showlegend=([True]+[False]*10)[r_i],
                                    line=dict(width=5, color ='black'),
                                    name=self.model_ind),
                                    row=int(np.floor(r_i/4)+1),
                                    col=int(1+np.mod(r_i, 4)))

        fig.update_layout(title=f'<b>{start_var}</b> under <b>{scen}</b> (including your scenario) compared to current policies (dashed)',)
        # DROPDOWN
        fig.update_layout(
            updatemenus=[
                {'x':-0.05,
                'xanchor':"right",
                'yanchor':"top",
                'y':1.0,
                'direction':"down",
                'active':int(np.where(available_var == start_var)[0][0]),
                'showactive':True,
                "buttons": [dict(
                            label=var,
                            method="update",
                            args=[{"y": [np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario="ELV-SSP2-CP-D0").Value) for m in self.models_ref for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=self.model_ind, Scenario="ELV-SSP2-CP-D0").Value) for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario=scen).Value) for m in self.models_ref for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=self.model_ind, Scenario=scen).Value) for reg in np.array(self.xr_data.Region)
                                        ]},
                                {'title.text': f'<b>{var}</b> under <b>{scen}</b> (including your scenario) compared to current policies (dashed)', "yaxis.title.text": ''}], #
                        ) for var in available_var]
                },
            ]
        )

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.50,
            xanchor="right",
            x=-0.15,
            font=dict( 
                size=14,
                color="black"
            ),
        ))

        fig.update_layout(height=1100, width=2000, template='plotly_white')

        # import os
        try:
            os.remove('Figures/VariableData.html')
        except:
            3
        def html_w(typ):
            return '<html> '+typ+' <p style="font-family: Arial">'

        with open('Figures/VariableData.html', 'a') as f:
            f.write(html_w('<h1>')+'Variable data</p></h1>')
            f.write(html_w('<body>')+'This page contains the data for the a selection of variables from the ELEVATE project</p></body>')
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    def plot_variables_norm(self):
        print('- Plotting normalized variables into /Figures/VariableData_norm.html')
        scen = 'ELV-SSP2-NDC-D0'
        start_var = "Primary Energy|Wind"
        available_var = np.array([x for x in self.settings['required_variables'] if x in self.xr_data['Variable'].values])

        fig = make_subplots(rows=3, cols=4,
                            subplot_titles=[f'{r}' for r in np.array(self.xr_data.Region)],
                            horizontal_spacing = 0.04, vertical_spacing=0.10)

        for m_i, m in enumerate(self.models_ref):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                y = self.xr_data.sel(Variable=start_var,
                                     Region=r,
                                     Model=m,
                                     Scenario="ELV-SSP2-CP-D0")
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=(y/y.sel(Time=2015)).Value,
                                        showlegend=False,
                                        line=dict(dash='dash', color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=int(np.floor(r_i/4)+1),
                                        col=int(1+np.mod(r_i, 4)))
            
        for r_i, r in enumerate(np.array(self.xr_data.Region)):
            y = self.xr_data.sel(Variable=start_var,
                                    Region=r,
                                    Model=self.model_ind,
                                    Scenario="ELV-SSP2-CP-D0")
            fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                    y=(y/y.sel(Time=2015)).Value,
                                    showlegend=False,
                                    line=dict(width=5, dash='dash', color ='black'),
                                    name=self.model_ind),
                                    row=int(np.floor(r_i/4)+1),
                                    col=int(1+np.mod(r_i, 4)))

        for m_i, m in enumerate(self.models_ref):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                y = self.xr_data.sel(Variable=start_var,
                                     Region=r,
                                     Model=m,
                                     Scenario=scen)
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=(y/y.sel(Time=2015)).Value,
                                        showlegend=([True]+[False]*10)[r_i],
                                        line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=int(np.floor(r_i/4)+1),
                                        col=int(1+np.mod(r_i,4)))
                
        for r_i, r in enumerate(np.array(self.xr_data.Region)):
            y = self.xr_data.sel(Variable=start_var,
                                    Region=r,
                                    Model=self.model_ind,
                                    Scenario=scen)
            fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                    y=(y/y.sel(Time=2015)).Value,
                                    showlegend=([True]+[False]*10)[r_i],
                                    line=dict(width=5, color ='black'),
                                    name=self.model_ind),
                                    row=int(np.floor(r_i/4)+1),
                                    col=int(1+np.mod(r_i, 4)))

        fig.update_layout(title=f'<b>{start_var}</b> under <b>{scen}</b> (including your scenario) compared to current policies (dashed)',)
        # DROPDOWN
        fig.update_layout(
            updatemenus=[
                {'x':-0.05,
                'xanchor':"right",
                'yanchor':"top",
                'y':1.0,
                'direction':"down",
                'active':int(np.where(available_var == start_var)[0][0]),
                'showactive':True,
                "buttons": [dict(
                            label=var,
                            method="update",
                            args=[{"y": [np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario="ELV-SSP2-CP-D0").Value / self.xr_data.sel(Time=2015, Variable=var, Region=reg, Model=m, Scenario="ELV-SSP2-CP-D0").Value) for m in self.models_ref for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=self.model_ind, Scenario="ELV-SSP2-CP-D0").Value / self.xr_data.sel(Time=2015, Variable=var, Region=reg, Model=self.model_ind, Scenario="ELV-SSP2-CP-D0").Value) for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario=scen).Value / self.xr_data.sel(Time=2015, Variable=var, Region=reg, Model=m, Scenario=scen).Value) for m in self.models_ref for reg in np.array(self.xr_data.Region)
                                         ]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=self.model_ind, Scenario=scen).Value / self.xr_data.sel(Time=2015, Variable=var, Region=reg, Model=self.model_ind, Scenario=scen).Value) for reg in np.array(self.xr_data.Region)
                                        ]},{'title.text': f'<b>{var}</b> under <b>{scen}</b> (including your scenario) compared to current policies (dashed)', "yaxis.title.text": ''}], #
                        ) for var in available_var]
                },
            ]
        )

        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.50,
            xanchor="right",
            x=-0.15,
            font=dict( 
                size=14,
                color="black"
            ),
        ))

        fig.update_layout(height=1100, width=2000, template='plotly_white')

        # import os
        try:
            os.remove('Figures/VariableData_norm.html')
        except:
            3
        def html_w(typ):
            return '<html> '+typ+' <p style="font-family: Arial">'

        with open('Figures/VariableData_norm.html', 'a') as f:
            f.write(html_w('<h1>')+'Variable data (normalized)</p></h1>')
            f.write(html_w('<body>')+'This page contains the data for the a selection of variables from the ELEVATE project. All data is normalized to the 2015 value to make regional aggregation differences comparable.</p></body>')
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    def plot_indicators(self):
        print('- Plotting variables into /Figures/Indicators.html')
        # Define indicators to include as all but the ones in indicators_to_exclude
        all_indicators = np.array(list(self.settings['indicators'].keys()))
        indicators_to_include = [x for x in all_indicators if x not in self.settings['indicators_to_exclude']]

        scen = 'ELV-SSP2-NDC-D0'
        xrset = self.xr_ind.sel(Time=np.arange(2010, 2051))

        # Only 2050
        sum_fig = make_subplots(rows=1, cols=1,
                            horizontal_spacing = 0.04, vertical_spacing=0.10)

        med = xrset.median(dim='Model')
        std = xrset.std(dim='Model')
        xr_normed = (xrset - med) / std
        a=0
        for ind_i, ind in enumerate(indicators_to_include[::-1]):
            xr_use = xr_normed.sel(Indicator=ind, Time=2050)
            name = self.settings['indicators'][ind]['name']
            sum_fig.add_trace(go.Box(
                                x=np.array(xr_use.sel(Scenario=scen, Region=self.key_region).Value),
                                y=[name]*len(xr_use.Model),
                                showlegend=False, line=dict(color='silver')), row=1, col=1)
            for m_i, m in enumerate(self.models_ref):
                sum_fig.add_trace(go.Scatter(y=[name],
                                        x=[xr_use.sel(Scenario=scen, Region=self.key_region, Model=m).Value],
                                        showlegend=([True]+[False]*30)[ind_i],
                                        mode='markers',
                                        marker=dict(size=13),
                                        line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=1,
                                        col=1)
            sum_fig.add_trace(go.Scatter(y=[name],
                                    x=[xr_use.sel(Scenario=scen, Region=self.key_region, Model=self.model_ind).Value],
                                    showlegend=([True]+[False]*30)[ind_i],
                                    mode='markers',
                                    marker=dict(size=25, symbol='x'),
                                    line=dict(color = 'black'),
                                    name=self.model_ind),
                                    row=1,
                                    col=1)
            if ind_i in [1, 6, 13, 17, 19]:
                st = ['<b>Responsiveness</b>', '<b>Mitigation strategy</b>', '<b>Energy supply</b>', '<b>Energy demand</b>', '<b>Cost and effort</b>'][::-1][a]
                sum_fig.add_trace(go.Scatter(y=[st]*2,
                                                x=[-3, 3],
                                                mode='lines',
                                                line=dict(color='black', width=3),
                                                showlegend=False),
                                                row=1, col=1)
                a +=1
                
        sum_fig.update_traces(orientation='h')
            
        sum_fig.add_trace(go.Scatter(y=[self.settings['indicators'][ind]['name'] for ind in indicators_to_include],
                                x=[0]*len(indicators_to_include),
                                mode='lines',
                                line=dict(color='black', dash='dash'),
                                showlegend=False),
                                row=1, col=1)

        sum_fig.update_layout(legend=dict(
            yanchor="top",
            orientation='h',
            y=-0.05,
            xanchor="center",
            x=0.50,
            font=dict( 
                size=10,
                color="black"
            ),
        ))
        sum_fig.update_layout(width=1000, template='plotly_white')
        sum_fig.update_xaxes(range=[-3, 3])
        sum_fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [-2, -1, 0, 1, 2],
                ticktext = ['η-2σ', 'η-σ', 'η', 'η+σ', 'η+2σ']
            )
        )

        # All individual indicators
        figs = []
        for ind_i, ind in enumerate(indicators_to_include):
            if self.settings['indicators'][ind]['plottype'] == 'bar':
                subplot_titles = ['']
                rows = 1
                cols = 1
            else:
                subplot_titles = [f'{r}' for r in np.array(self.xr_data.Region)]
                rows = 2
                cols = 5
            if self.settings['indicators'][ind]['axisnorm'] == 'n': shared_yaxes = False
            else: shared_yaxes = 'all'

            fig = make_subplots(rows=rows, cols=cols,
                                subplot_titles=subplot_titles,
                                horizontal_spacing = 0.04, vertical_spacing=0.10, shared_yaxes=shared_yaxes, shared_xaxes=True)
            
            if ind_i in ['ED1_etrans', 'ED2_eindus', 'ED3_ebuild']:
                fig.update_yaxes(range=[0, 1])
            if ind in ['M4_nonco2']:
                fig.update_yaxes(range=[-2, 2])


            if self.settings['indicators'][ind]['plottype'] == 'line':
                for m_i, m in enumerate(self.models_ref):
                    for r_i, r in enumerate(np.array(xrset.Region)):
                        fig.add_trace(go.Scatter(x=xrset.Time,
                                                y=xrset.sel(Indicator=ind, Region=r, Model=m, Scenario=scen).Value,
                                                showlegend=([True]+[False]*10)[r_i],
                                                line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                                name=m),
                                                row=1+np.floor(r_i/5).astype(int),
                                                col=np.mod(r_i, 5)+1)
                
                for r_i, r in enumerate(np.array(xrset.Region)):
                    fig.add_trace(go.Scatter(x=xrset.Time,
                                            y=xrset.sel(Indicator=ind, Region=r, Model=self.model_ind, Scenario=scen).Value,
                                            showlegend=([True]+[False]*10)[r_i],
                                            line=dict(color = 'black', width=5),
                                            name=self.model_ind),
                                            row=1+np.floor(r_i/5).astype(int),
                                            col=np.mod(r_i, 5)+1)
                    if self.settings['indicators'][ind]['horline'] == 'y':
                        fig.add_shape(type="line",
                                    x0=2010, y0=0, x1=2050, y1=0,
                                    line=dict(color="black", width=1, dash="dash"),
                                    row=1+np.floor(r_i/5).astype(int),
                                                col=np.mod(r_i, 5)+1)
                    
            else:
                for r_i, r in enumerate(np.array(xrset.Region)):
                    fig.add_trace(go.Box(x=[r]*len(self.models_ref),
                                        y=np.array(xrset.sel(Indicator=ind, Region=r, Model=self.models_ref, Scenario=scen, Time=2050).Value),
                                        showlegend=False, line=dict(color='silver')), row=1, col=1)
                    for m_i, m in enumerate(self.models_ref):
                        fig.add_trace(go.Scatter(x=[r],
                                                y=[xrset.sel(Indicator=ind, Region=r, Model=m, Scenario=scen, Time=2050).Value],
                                                showlegend=([True]+[False]*10)[r_i],
                                                mode='markers',
                                                marker=dict(size=13),
                                                line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                                name=m),
                                                row=1,
                                                col=1)
                for r_i, r in enumerate(np.array(xrset.Region)):
                    fig.add_trace(go.Scatter(x=[r],
                                            y=[xrset.sel(Indicator=ind, Region=r, Model=self.model_ind, Scenario=scen, Time=2050).Value],
                                            showlegend=([True]+[False]*10)[r_i],
                                            mode='markers',
                                            marker=dict(size=25, symbol='x'),
                                            line=dict(color = 'black'),
                                            name=self.model_ind),
                                            row=1,
                                            col=1)
                    
                if self.settings['indicators'][ind]['horline'] == 'y':
                    fig.add_trace(go.Scatter(x=np.array(xrset.Region),
                                             y=[0]*len(xrset.Region),
                                                mode='lines',
                                                line=dict(color='black', dash='dash'),
                                                showlegend=False),
                                                row=1, col=1)

            if shared_yaxes == 'all':
                # unhide y-axis ticks
                fig.update_layout(yaxis1=dict(showticklabels=True))
                fig.update_layout(yaxis2=dict(showticklabels=True))
                fig.update_layout(yaxis3=dict(showticklabels=True))
                fig.update_layout(yaxis4=dict(showticklabels=True))
                fig.update_layout(yaxis5=dict(showticklabels=True))
                fig.update_layout(yaxis6=dict(showticklabels=True))
                fig.update_layout(yaxis7=dict(showticklabels=True))
                fig.update_layout(yaxis8=dict(showticklabels=True))
                fig.update_layout(yaxis9=dict(showticklabels=True))
                fig.update_layout(yaxis10=dict(showticklabels=True))

            fig.update_layout(legend=dict(
                yanchor="top",
                orientation='h',
                y=-0.05,
                xanchor="center",
                x=0.50,
                font=dict( 
                    size=10,
                    color="black"
                ),
            ))
            fig.update_layout(height=600, width=1600, template='plotly_white')
            figs.append(fig)
        try:
            os.remove('Figures/Indicators.html')
        except:
            3
        def html_w(typ):
            return '<html> '+typ+' <p style="font-family: Arial">'

        with open('Figures/Indicators.html', 'a') as f:
            f.write(html_w('<h1>')+'Indicators</p></h1>')
            f.write(html_w('<body>')+'This page contains the results of the indicators from the ELEVATE poject. The ELV-SSP2-NDC-D0 scenario is used here (together with your own scenario). Please note that in some plots, the panels have similar axes, but not in all of them.</p></body>')
    
            f.write(html_w('<h1>')+'Summary</p></h1>')
            f.write(html_w('<body>')+'2050 values of all indicators in region <b>'+self.key_region+'</b> (for this plot we automatically pick the first region in your file).</p></body>')
            f.write(sum_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            for n_i in range(len(figs)):
                ind = indicators_to_include[n_i]
                f.write('<hr>')
                f.write(html_w('<h1>')+'Indicator '+ind.split('_')[0]+': '+self.settings['indicators'][ind]['name']+'</p></h1>')
                f.write(html_w('<body>')+self.settings['indicators'][ind]['explanation']+'</p></body>')
                f.write(figs[n_i].to_html(full_html=False, include_plotlyjs='cdn'))

        self.xr_data.close()
        self.xr_ind.close()