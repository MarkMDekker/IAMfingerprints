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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class class_plotting:
    ''' Class object that does the plotting'''

    def __init__(self):
        with open("config.yaml", "r") as stream:
            self.settings = yaml.load(stream, Loader=yaml.Loader)
        self.xr_data = xr.open_dataset('Data/xr_variables.nc')
        self.xr_ind = xr.open_dataset('Data/xr_indicators.nc')
    
    def plot_variables(self):
        start_reg = 'China'
        scen = 'ELV-SSP2-NDC-D0'
        start_var = "Primary Energy|Wind"
        available_var = np.array([x for x in self.settings['required_variables'] if x in self.xr_data['Variable'].values])

        fig = make_subplots(rows=2, cols=3,
                            specs=[[{}, {}, {}], [{}, {}, {}]],
                            subplot_titles=[f'{r}' for r in np.array(self.xr_data.Region)],
                            horizontal_spacing = 0.04, vertical_spacing=0.10)

        for m_i, m in enumerate(np.array(self.xr_data.Model)):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=self.xr_data.sel(Variable=start_var,
                                                            Region=r,
                                                            Model=m,
                                                            Scenario="ELV-SSP2-CP-D0").Value,
                                        showlegend=False,
                                        line=dict(dash='dash', color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m), row=[1, 1, 1, 2, 2, 2][r_i], col=[1, 2, 3, 1, 2, 3][r_i])

        for m_i, m in enumerate(np.array(self.xr_data.Model)):
            for r_i, r in enumerate(np.array(self.xr_data.Region)):
                fig.add_trace(go.Scatter(x=self.xr_data.Time,
                                        y=self.xr_data.sel(Variable=start_var,
                                                            Region=r,
                                                            Model=m,
                                                            Scenario=scen).Value,
                                        showlegend=[True, False, False, False, False, False][r_i],
                                        line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                        name=m), row=[1, 1, 1, 2, 2, 2][r_i], col=[1, 2, 3, 1, 2, 3][r_i])

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
                            args=[{"y": [np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario="ELV-SSP2-CP-D0").Value) for m in self.settings['models'] for reg in np.array(self.xr_data.Region)]+[np.array(self.xr_data.sel(Variable=var, Region=reg, Model=m, Scenario=scen).Value) for m in self.settings['models'] for reg in np.array(self.xr_data.Region)]},
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

        fig.update_layout(height=700, width=1600, template='plotly_white')

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

    def plot_indicators(self):
        start_reg = 'China'
        scen = 'ELV-SSP2-NDC-D0'
        inds = np.array(list(self.settings['indicators'].keys()))
        xrset = self.xr_ind.sel(Time=np.arange(2010, 2051))
        normaxis = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23]
        barplots = [1, 2, 8, 21, 22, 23]

        figs = []
        for ind_i, ind in enumerate(inds):
            if ind_i in barplots:
                specs = [[{}]]
                subplot_titles = ['']
            else:
                specs = [[{}, {}, {}, {}, {}]]
                subplot_titles = [f'{r}' for r in np.array(self.xr_data.Region)]
            if ind_i not in normaxis: shared_yaxes = False
            else: shared_yaxes = True

            fig = make_subplots(rows=1, cols=len(specs[0]),
                                specs=specs,
                                subplot_titles=subplot_titles,
                                horizontal_spacing = 0.04, vertical_spacing=0.10, shared_yaxes=shared_yaxes)
            
            if ind_i in [16, 17, 18]:
                fig.update_yaxes(range=[0, 1])

            if ind_i not in barplots:
                for m_i, m in enumerate(np.array(xrset.Model)):
                    for r_i, r in enumerate(np.array(xrset.Region)):
                        fig.add_trace(go.Scatter(x=xrset.Time,
                                                y=xrset.sel(Indicator=ind, Region=r, Model=m, Scenario=scen).Value,
                                                showlegend=[True, False, False, False, False, False][r_i],
                                                line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                                name=m), row=1, col=[1, 2, 3, 4, 5][r_i])
            else:
                for r_i, r in enumerate(np.array(xrset.Region)):
                    fig.add_trace(go.Box(x=[r]*len(np.array(xrset.Model)),
                                        y=np.array(xrset.sel(Indicator=ind, Region=r, Model=np.array(xrset.Model), Scenario=scen, Time=2050).Value),
                                        showlegend=False, line=dict(color='silver')), row=1, col=1)
                    for m_i, m in enumerate(np.array(xrset.Model)):
                        fig.add_trace(go.Scatter(x=[r],
                                                y=[xrset.sel(Indicator=ind, Region=r, Model=m, Scenario=scen, Time=2050).Value],
                                                showlegend=[True, False, False, False, False, False][r_i],
                                                mode='markers',
                                                marker=dict(size=15),
                                                line=dict(color = ['steelblue', 'tomato', 'forestgreen', 'goldenrod', 'brown', 'purple'][m_i]),
                                                name=m), row=1, col=1)

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
            for n_i in range(len(figs)):
                ind = list(self.settings['indicators'].keys())[n_i]
                if n_i > 0:
                    f.write('<hr>')
                f.write(html_w('<h1>')+'Indicator '+ind.split('_')[0]+': '+self.settings['indicators'][ind]['name']+'</p></h1>')
                f.write(html_w('<body>')+self.settings['indicators'][ind]['explanation']+'</p></body>')
                f.write(figs[n_i].to_html(full_html=False, include_plotlyjs='cdn'))