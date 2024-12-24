import streamlit as st
import pandas as pd
import plotly.express as px
from eurostatapiclient import EurostatAPIClient
import plotly.graph_objects as go
import numpy as np
st.title('Overview of raw milk prices and milk yield in Ireland and EU')
#choose version, format and language
VERSION = '1.0'
FORMAT = 'json'
LANGUAGE = 'en'

def get_eurostat_data(dataset_code, params={}):
    """
    Retrieves data from Eurostat using the EurostatAPIClient with error handling.

    Args:
        dataset_code (str): The Eurostat dataset code.
        params (dict, optional): Additional parameters for the API call.
            Defaults to {}.

    Returns:
        raw data, or None if an error occurs.
    """

    try:
        client = EurostatAPIClient(VERSION, FORMAT, LANGUAGE)
        dataset = client.get_dataset(dataset_code, params=params)
        return dataset

    except Exception as e:
        st.error(f"Error fetching data from Eurostat: {e}")
        return None

dataset_code = "tag00070"
selected_countries = st.multiselect("Select Countries",                                     
                                   ['IE', 'DK', 'NL', 'AT', 'BE', 'BG', 
                                    'CY', 'CZ', 'DE', 'EE', 'EL', 'ES', 
                                    'FI', 'FR', 'HR', 'HU', 'IT', 'LT', 
                                    'LU', 'LV', 'MT', 'PL', 'PT', 'RO', 
                                    'SE', 'SI', 'SK'], 
                                    default=['IE']
                                   )  # Allow user selection

data_raw = get_eurostat_data(dataset_code, params={'geo': selected_countries})
label = data_raw.label 
data = data_raw.to_dataframe()

colors = ['#8B4513', '#A9A9A9', '#D2B48C', '#696969', '#D2691E', '#808080', '#A0522D']
fig1 = px.line(data, 
               x='time', 
               y='values', 
               color='geo', 
               title=f"Eurostat Data: {label}", 
               color_discrete_sequence=colors, 
               hover_data={'time': True, 'values': True, 'geo': True} ) 

# Customize hovertext labels 
fig1.update_traces(hovertemplate='<b>Country:</b> %{customdata[0]}<br><b>Year:</b> %{x}<br><b>Price:</b> %{y}')

st.plotly_chart(fig1)

data_raw_ie = get_eurostat_data(dataset_code, params={'geo': ['IE']})
data_ie = data_raw_ie.to_dataframe()
data_ie['price_change'] = data_ie['values'].pct_change() * 100

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=data_ie['time'],
        y=data_ie['values'],
        mode='lines+markers',
        name='Raw Milk Price in Ireland',
        line=dict(color='brown'),
        marker=dict(size=6),
        customdata=data_ie[['time', 'values', 'price_change']].values
    )
)

# Bar plot for percentage price change
fig2.add_trace(
    go.Bar(
        x=data_ie['time'],
        y=data_ie['price_change'],
        name='Price Change (%)',
        marker=dict(color='lightgray',
                    opacity=0.7),
        yaxis='y2'
    )
)

fig2.update_layout(
    title='Raw Milk Price in Ireland Over Time',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Raw Milk Price (per 100 kg)', 
               titlefont=dict(color='black'), 
               showgrid=True,
              ),
    yaxis2=dict(
        title='Percentage Change (%)',
        overlaying='y',
        side='right',
        titlefont=dict(color='gray'),
        showgrid=False
    ),
    legend=dict(orientation='h', 
                yanchor='bottom', 
                y=1.02, 
                xanchor='right', x=1)
)

# Adding annotations for non-zero price changes
for i, row in data_ie.iterrows():
    if pd.notna(row['price_change']):        
        fig2.add_annotation(
                x=row['time'],
                y=row['price_change'],
                text=f"{row['price_change']:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=20 if row['price_change'] > 0 else -20,
                font=dict(size=10)
            )
fig2.update_traces(hovertemplate='<b>Year:</b> %{x}<br><b>Raw Milk Price:</b> %{customdata[1]}<br><b>Price Change:</b> %{customdata[2]}%')

st.plotly_chart(fig2)

gdp_code = "tipsna40"
collection_code = "tag00041"
cows_code = "tag00014"
years = ['2020', '2021', '2022', '2023']
countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI',
       'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL',
       'PT', 'RO', 'SE', 'SI', 'SK']
year = st.selectbox("Select Year", years)
gdp_raw = get_eurostat_data(gdp_code, params={#'geo': countries, 
    "time": year})
milk_raw = get_eurostat_data(collection_code, params={#'geo': countries, 
                                                      "time": year})
cows_raw = get_eurostat_data(cows_code, params={#'geo': countries, 
                                                "time": year})
gdp = pd.read_csv('GDP per capita EU.csv')#gdp_raw.to_dataframe()
gdp.rename(columns={'year': 'time', 'country': 'geo', 'value': 'values'}, inplace=True)
gdp['time'] = gdp['time'].astype(str)
collection = milk_raw.to_dataframe()
cows = cows_raw.to_dataframe()

EDA = pd.merge(collection.drop(columns=['freq', 'milkitem', 'dairyprod']), 
                 cows.drop(columns=['freq', 'animals', 'month', 'unit']),
                on=['geo', 'time'],
              suffixes=('_milk_on_farms', '_num_cows')
              )

EDA = pd.merge(EDA,
               gdp.drop(columns=['unit', 'category']),
               on=['geo', 'time']
              )

EDA = EDA[EDA['geo'].isin(countries)]

EDA['milk_yield'] = EDA['values_milk_on_farms'] / EDA['values_num_cows']
apparent_milk_yield = EDA['milk_yield'].mean()*1000
# Calculate additional variables
EDA['milk_prod_mln'] = EDA['values_milk_on_farms'] / 1000
dairy_cows_range = np.linspace(min(EDA['values_num_cows']), max(EDA['values_num_cows']), 100)
milk_production_curve = (dairy_cows_range / 1000) * (apparent_milk_yield/ 1000)
marker_size = (EDA['values'] / EDA['values'].max()) * 30

# Create Plotly figure
fig3 = go.Figure()

# Scatter plot for countries
fig3.add_trace(go.Scatter(
    x=EDA['values_num_cows'],
    y=EDA['milk_prod_mln'],
    mode='markers+text',
    hovertext=EDA['geo'],
    marker=dict(
        size=marker_size,
        color='brown',
        opacity=0.8,
        #line=dict(width=1, color='black')
    ),
    name='Country',
    customdata=EDA[['geo', 'values_num_cows', 'milk_prod_mln']]
))


# Add line for average milk yield
fig3.add_trace(go.Scatter(
    x=dairy_cows_range,
    y=milk_production_curve,
    mode='lines',
    line=dict(color='lightgray', dash='dash'),
    name=f'Average apparent milk yield ({year})'
))

for index, row in EDA.iterrows():
    if row['geo'] in ['IE', 'DK', 'NL']:
        fig3.add_annotation(
            x=np.log10(row['values_num_cows']),
            y=row['milk_prod_mln'],
            text=row['geo'],
            arrowhead=2,
            ax=15, ay=-20,
            font=dict(size=10)
        )

# Set log scale for x-axis
fig3.update_xaxes(
    title_text='Dairy cows (thousand heads)',
    type='log',
    tickvals=[1, 10, 100, 1000],
    ticktext=['1', '10', '100', '1000'],
    range=[np.log10(min(EDA['values_num_cows']) * 0.9), np.log10(max(EDA['values_num_cows']) * 1.1)]
)

# Set y-axis
fig3.update_yaxes(
    title_text='Raw milk produced (mln t)',
    range=[0, 40]
)

# Update layout
fig3.update_layout(
    title=f'Dairy Cows and Milk Production with Apparent Milk Yield in EU ({year})',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    template='plotly_white'
)
fig3.update_traces(hovertemplate='<b>Country:</b> %{customdata[0]}<br><b>Dairy Cows:</b> %{customdata[1]}<br><b>Raw Milk Produced:</b> %{customdata[2]} mln t')
st.plotly_chart(fig3)

st.markdown("Data source: [Eurostat](https://ec.europa.eu/eurostat/en/web/main/data/database)")
st.markdown("The copyright for the editorial content of the Eurostat website, which is owned by the EU, is licensed under the Creative Commons Attribution 4.0 International licence.[link](https://creativecommons.org/licenses/by/4.0/)") 
