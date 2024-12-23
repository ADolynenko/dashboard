import streamlit as st
import pandas as pd
import plotly.express as px
from eurostatapiclient import EurostatAPIClient
import plotly.graph_objects as go
import numpy as np

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
#code for the raw milk dataset
dataset_code = "tag00070"
selected_countries = st.multiselect("Select Countries",                                     
                                   ['IE', 'DK', 'NL']
                                   )  # Allow user selection

data_raw = get_eurostat_data(dataset_code, params={'geo': selected_countries})
label = data_raw.label 
data = data_raw.to_dataframe()

if data is not None:
    try:
        if 'time' in data.columns and 'geo' in data.columns:            
            fig = px.line(data, x='time', y='values', color='geo',
                          title=f"Eurostat Data: {label}")
            st.plotly_chart(fig)
        else:
            st.warning("The dataset doesn't contain required columns ('time' or 'geo'). Adapt the plot accordingly.")

    except KeyError as e:
        st.error(f"Error creating plot: Column '{e}' not found. Please inspect the raw data to see the available columns.")
    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")
else:
    st.write("Failed to retrieve data. Check the dataset code and internet connection.")





data_raw_ie = get_eurostat_data(dataset_code, params={'geo': ['IE']})
data_ie = data_raw_ie.to_dataframe()
data_ie['price_change'] = data_ie['values'].pct_change() * 100

# Create the figure
fig = go.Figure()

# Line plot 
fig.add_trace(
    go.Scatter(
        x=data_ie['time'],
        y=data_ie['values'],
        mode='lines+markers',
        name='Raw Milk Price in Ireland',
        line=dict(color='black'),
        marker=dict(size=6)
    )
)

# Bar plot for percentage price change
fig.add_trace(
    go.Bar(
        x=data_ie['time'],
        y=data_ie['price_change'],
        name='Price Change (%)',
        marker=dict(color='lightgray',
                    opacity=0.7),
        yaxis='y2'
    )
)

fig.update_layout(
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
    if row['price_change'] != 0:        
        fig.add_annotation(
                x=row['time'],
                y=row['price_change'],
                text=f"{row['price_change']:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=20 if row['price_change'] > 0 else -20,
                font=dict(size=10)
            )

# Show in Streamlit
st.plotly_chart(fig)

gdp_code = "tipsna40"
collection_code = "tag00041"
cows_code = "tag00014"
years = ['2020', '2021', '2022', '2023']
countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI',
       'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL',
       'PT', 'RO', 'SE', 'SI', 'SK']
year = st.selectbox("Select Year", years)
gdp_raw = get_eurostat_data(gdp_code, params={'geo': countries, "time": year})
milk_raw = get_eurostat_data(collection_code, params={'geo': countries, "time": year})
cows_raw = get_eurostat_data(cows_code, params={'geo': countries, "time": year})
gdp = gdp_raw.to_dataframe()
collection = milk_raw.to_dataframe()
cows = cows_raw.to_dataframe()

EDA = pd.merge(collection.drop(columns=['freq', 'milkitem', 'dairyprod']), 
                 cows.drop(columns=['freq', 'animals', 'month', 'unit']),
                on=['geo', 'time'],
              suffixes=('_milk_on_farms', '_num_cows')
              )

EDA = pd.merge(EDA,
               gdp.drop(columns=['freq', 'unit', 'na_item']),
               on=['geo', 'time']
              )
EDA['milk_yield'] = EDA['values_milk_on_farms'] / EDA['values_num_cows']
EDA['apparent_milk_yield'] = EDA['milk_yield'] * 1000

# Calculate additional variables
EDA['milk_prod_mln'] = EDA['values_milk_on_farms'] / 1000
dairy_cows_range = np.linspace(min(EDA['values_num_cows']), max(EDA['values_num_cows']), 100)
milk_production_curve = (dairy_cows_range / 1000) * (EDA['apparent_milk_yield'].mean() / 1000)
marker_size = (EDA['values'] / EDA['values'].max()) * 30


# Create Plotly figure
fig = go.Figure()

# Scatter plot for countries
fig.add_trace(go.Scatter(
    x=EDA['values_num_cows'],
    y=EDA['milk_prod_mln'],
    mode='markers+text',
    #text=EDA['geo'],
    #textposition='top center',
    marker=dict(
        size=marker_size,
        color='red',
        opacity=0.8,
        line=dict(width=1, color='black')
    ),
    name='Countries'
))
# Add annotations for specific countries
for i, row in EDA.iterrows():
    if row['geo'] in ['IE', 'DK', 'NL']:
        fig.add_annotation(
            x=row['values_num_cows'],
            y=row['milk_prod_mln'],
            text=row['geo'],
            showarrow=True,
            arrowhead=2,
            ax=row['values_num_cows'] * 0.05,  # Adjust arrow x-offset
            ay=row['milk_prod_mln'] * 0.05  # Adjust arrow y-offset
        )
# Add line for average milk yield
fig.add_trace(go.Scatter(
    x=dairy_cows_range,
    y=milk_production_curve,
    mode='lines',
    line=dict(color='green', dash='dash'),
    name=f'Average apparent milk yield ({year})'
))

# Set log scale for x-axis
fig.update_xaxes(
    title_text='Dairy cows (thousand heads)',
    type='log',
    tickvals=[1, 10, 100, 1000],
    ticktext=['1', '10', '100', '1000'],
    
)

# Set y-axis
fig.update_yaxes(
    title_text='Raw milk produced (mln t)',
    range=[0, 40]
)

# Update layout
fig.update_layout(
    title=f'Dairy Cows and Milk Production with Apparent Milk Yield in EU ({year})',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    template='plotly_white'
)

# Display in Streamlit
st.plotly_chart(fig)


