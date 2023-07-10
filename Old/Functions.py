# HELPERFUNCTIONS

# ========================================================================================================================== #
# REQUIRED PACKAGES
# ========================================================================================================================== #

import pandas as pd
import numpy as np

# ========================================================================================================================== #
# FUNCTIONS
# ========================================================================================================================== #

def interpolate_missing_5years(data):
    col_2010 = list(data.columns).index('2010')
    for i, year in enumerate(range(2015, 2105, 10)):
        data.insert(col_2010+2*i+1, str(year), data[[str(year-5), str(year+5)]].mean(axis=1))

def get(data, scenario, variable, year=None):
    selection = data[(data['Scenario'] == scenario) & (data['Variable'] == variable)].set_index('Model')
    if year is None:
        return selection.loc[:,'2010':]
    else:
        return selection[year]

def set_value_from_var_column(data, meta, metacol, col, year, scenario):
    """
    Adds a column to the meta df using the `data` df. Which variable to use
    is taken from a `metacol` column in the meta df. For example, the GDP
    metric is sometimes GDP|PPP and sometimes GDP|MER, depending on the model.
    
    In this example, metacol would be 'GDP_metric'.
    """
    meta[col] = np.nan
    for model, info in meta.iterrows():
        var = info[metacol]
        selection = data[
            (data['Variable'] == var)
            & (data['Model'] == model)
            & (data['Scenario'] == scenario)
        ]
        if len(selection) > 0:
            meta.loc[model, col] = selection.iloc[0][year]

def make_rows(model, scenario, variable, timeline, data):
    dfs = []
    for k in range(len(data)):
        dic = {}
        dic['Model'] = model
        dic['Scenario'] = scenario
        dic['Year'] = timeline[k]
        dic[variable] = data[k]
        dfs.append(pd.DataFrame(dic, index=[100]))
    return dfs

def add_legend_item(fig, name='', mode='markers', **kwargs):
    """
    In Plotly, a legend item can be added manually by adding an empty trace
    """
    fig.add_scatter(x=[None], y=[None], name=name, mode=mode, **kwargs)

def calc_relative_abatement_index(data, year, var, pol='diag-c80-gr5', base='diag-base'):
    CO2_FFI_base = get(data, base, var, year)
    CO2_FFI_pol  = get(data, pol, var, year)
    dat = (CO2_FFI_base - CO2_FFI_pol) / (CO2_FFI_base)
    dat[CO2_FFI_base == 0] = 0
    return dat

def calc_carbon_and_energy_intensity(data, year, scenario, meta, emisvar):
    
    # For each model, get either GDP|PPP or GDP|MER (depending on column `GDP_metric`)
    GDP_column = f'GDP {year} {scenario}'
    set_value_from_var_column(data, meta, 'GDP_metric', GDP_column, year, scenario)
    
    CO2_FFI      = get(data, scenario, emisvar, year)
    final_energy = get(data, scenario, 'Final Energy', year)
    GDP_PPP      = meta[GDP_column]
    
    carbon_intensity = CO2_FFI / final_energy
    energy_intensity = final_energy / GDP_PPP
    
    return carbon_intensity, energy_intensity

def calc_normalised_carbon_and_energy_intensity(data, year, meta, emisvar):
    CI_pol, EI_pol           = calc_carbon_and_energy_intensity(data, year, 'diag-c80-gr5', meta, emisvar)
    CI_baseline, EI_baseline = calc_carbon_and_energy_intensity(data, year, 'diag-base', meta, emisvar)
    return CI_pol / CI_baseline, EI_pol / EI_baseline

def ellipse(a, b, npoints):
    x = np.linspace(-a, a, npoints)
    y1 = b * np.sqrt(1-(x/a)**2)
    y2 = -y1
    return np.concatenate([x,x[::-1]]), np.concatenate([y1,y2[::-1]])

def rotate(x, y, theta):
    return x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta)

def confidence_ellipse(x_values, y_values, nsigma, npoints=300):
    # Calculate center of confidence ellipse
    mu_x, mu_y = np.mean(x_values), np.mean(y_values)
    
    # Calculate correlation coefficient and covariances
    cov_matrix = np.cov([x_values, y_values])
    cov_xy = cov_matrix[0,1]
    sigma_x, sigma_y = np.sqrt(cov_matrix[0,0]), np.sqrt(cov_matrix[1,1])
    rho = cov_xy / (sigma_x * sigma_y)
    
    # Get the x-y points for the default ellipse with a=sqrt(1+rho), b=sqrt(1-rho)
    ellipse_x, ellipse_y = ellipse(np.sqrt(1+rho), np.sqrt(1-rho), npoints)
    
    # Rotate ellipse 45 degrees counter-clockwise
    ellipse_x, ellipse_y = rotate(ellipse_x, ellipse_y, np.pi/4)
    
    # Scale ellipse horizontally by (2*n*sigma_x) and vertically by (2*n*sigma_y)
    # Note: scaling by 2*n*sigma_x means that the x_values (centered around 0) should
    # be multiplied by n*sigma_x, not 2*n*sigma_x
    ellipse_x = nsigma*sigma_x * ellipse_x
    ellipse_y = nsigma*sigma_y * ellipse_y
    
    # Shift ellipse such that its center is situated at the point mu_x, mu_y
    ellipse_x += mu_x
    ellipse_y += mu_y
    
    return ellipse_x, ellipse_y

def calc_fossil_fuel_reduction(data, year, pol, base='diag-base', var='Primary Energy|Fossil'):
    
    prim_energy_fossil_2020 = get(data, base, var, '2020')
    prim_energy_fossil_pol  = get(data, pol, var, year)
    
    return (prim_energy_fossil_2020 - prim_energy_fossil_pol) / prim_energy_fossil_2020

def color(color, text):
    return f"<span style='color:{str(color)}'> {str(text)} </span>"

def linearInterp(row):
    years = row.index.astype(float)
    return np.trapz(row, x=years)

def cp80gr5(t):
    return 80*1.05**(t-2040)

def set_value_from_var_column(data, meta, metacol, col, year, scenario):
    meta[col] = np.nan
    for model, info in meta.iterrows():
        var = info[metacol]
        selection = data[
            (data['Variable'] == var)
            & (data['Model'] == model)
            & (data['Scenario'] == scenario)
        ]
        if len(selection) > 0:
            meta.loc[model, col] = selection.iloc[0][year]