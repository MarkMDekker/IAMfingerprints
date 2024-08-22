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
        self.xr_data = xr.open_dataset('Data/xr_variables.nc')
        self.xr_ind = xr.open_dataset('Data/xr_indicators.nc')
        self.models_all = self.xr_data.Model.values
        self.models_ref = self.settings['models']
        # find the model that is in self.models_all but not in self.models_ref
        self.model_ind = [x for x in self.models_all if x not in self.models_ref][0]
    
    def plot_variables(self):
        print('- Plotting variables into /Figures/VariableData.html')
        start_reg = 'China'
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

        fig.update_layout(title=f'<b>{start_var}</b> under <b>{scen}</b> compared to current policies (dashed)',)
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
                                {'title.text': f'<b>{var}</b> under <b>{scen}</b> compared to current policies (dashed)', "yaxis.title.text": ''}], #
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
        start_reg = 'China'
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

        fig.update_layout(title=f'<b>{start_var}</b> under <b>{scen}</b> compared to current policies (dashed)',)
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
                                        ]},{'title.text': f'<b>{var}</b> under <b>{scen}</b> compared to current policies (dashed)', "yaxis.title.text": ''}], #
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
        scen = 'ELV-SSP2-NDC-D0'
        inds = np.array(list(self.settings['indicators'].keys()))
        xrset = self.xr_ind.sel(Time=np.arange(2010, 2051))
        normaxis = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23]
        barplots = [1, 2, 8, 21, 22, 23]
        key_region = np.array(self.xr_data.Region)[np.where(~np.isnan(self.xr_data.sel(Model=self.model_ind, Time=2050, Scenario=scen, Variable='Primary Energy|Coal').Value))[0]][0]

        # Only 2050
        sum_fig = make_subplots(rows=1, cols=1,
                            horizontal_spacing = 0.04, vertical_spacing=0.10)

        xr_normed = xrset / xrset.mean(dim='Model')
        for ind_i, ind in enumerate(inds):
            xr_use = xr_normed.sel(Indicator=ind, Time=2050)
            sum_fig.add_trace(go.Box(x=[ind.replace('_', '<br>')]*len(xr_use.Model),
                                y=np.array(xr_use.sel(Scenario=scen, Region=key_region).Value),
                                showlegend=False, line=dict(color='silver')), row=1, col=1)
            for m_i, m in enumerate(self.models_ref):
                sum_fig.add_trace(go.Scatter(x=[ind.replace('_', '<br>')],
                                        y=[xr_use.sel(Scenario=scen, Region=key_region, Model=m).Value],
                                        showlegend=([True]+[False]*30)[ind_i],
                                        mode='markers',
                                        marker=dict(size=13),
                                        line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m),
                                        row=1,
                                        col=1)
            sum_fig.add_trace(go.Scatter(x=[ind.replace('_', '<br>')],
                                    y=[xr_use.sel(Scenario=scen, Region=key_region, Model=self.model_ind).Value],
                                    showlegend=([True]+[False]*30)[ind_i],
                                    mode='markers',
                                    marker=dict(size=25, symbol='x'),
                                    line=dict(color = 'black'),
                                    name=self.model_ind),
                                    row=1,
                                    col=1)

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
            sum_fig.update_layout(height=800, template='plotly_white')
        # All individual indicators
        figs = []
        for ind_i, ind in enumerate(inds):
            if ind_i in barplots:
                subplot_titles = ['']
                rows = 1
                cols = 1
            else:
                subplot_titles = [f'{r}' for r in np.array(self.xr_data.Region)]
                rows = 2
                cols = 5
            if ind_i not in normaxis: shared_yaxes = False
            else: shared_yaxes = True

            fig = make_subplots(rows=rows, cols=cols,
                                subplot_titles=subplot_titles,
                                horizontal_spacing = 0.04, vertical_spacing=0.10, shared_yaxes=shared_yaxes, shared_xaxes=True)
            
            if ind_i in [16, 17, 18]:
                fig.update_yaxes(range=[0, 1])
            if ind_i in [8]:
                fig.update_yaxes(range=[-2, 2])


            if ind_i not in barplots:
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
            f.write(html_w('<body>')+'This page contains the results of the indicators from the ELEVATE project. The ELV-SSP2-NDC-D0 scenario is used here.</p></body>')
    
            f.write(html_w('<h1>')+'Summary</p></h1>')
            f.write(html_w('<body>')+'2050 values of all indicators in scenario <b>'+scen+'</b> and region <b>'+key_region+'</b></p></body>')
            f.write(sum_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            for n_i in range(len(figs)):
                ind = list(self.settings['indicators'].keys())[n_i]
                f.write('<hr>')
                f.write(html_w('<h1>')+'Indicator '+ind.split('_')[0]+': '+self.settings['indicators'][ind]['name']+'</p></h1>')
                f.write(html_w('<body>')+self.settings['indicators'][ind]['explanation']+'</p></body>')
                f.write(figs[n_i].to_html(full_html=False, include_plotlyjs='cdn'))