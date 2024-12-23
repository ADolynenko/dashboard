import streamlit as st
import pandas as pd
import plotly.express as px
from eurostatapiclient import EurostatAPIClient
import plotly.graph_objects as go

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
        marker=dict(color='lightgray'),
        opacity=0.7
    )
)

# Adding annotations for non-zero price changes
annotations = []
for i, row in data_ie.iterrows():
    if row['price_change'] != 0:
        annotations.append(
            dict(
                x=row['year'],
                y=row['price_change'],
                text=f"{row['price_change']:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(size=10)
            )
        )

fig.update_layout(
    title='Raw Milk Price in Ireland Over Time',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Raw Milk Price (per 100 kg)', titlefont=dict(color='black')),
    yaxis2=dict(
        title='Percentage Change (%)',
        overlaying='y',
        side='right',
        titlefont=dict(color='gray')
    ),
    annotations=annotations,
    legend=dict(x=0.1, y=0.9),
    template='plotly_white',
    bargap=0.4
)

# Add gridlines
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Show in Streamlit
st.plotly_chart(fig)
