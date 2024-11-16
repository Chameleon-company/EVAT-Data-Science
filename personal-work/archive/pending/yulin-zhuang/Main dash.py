import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load and preprocess data
df = pd.read_csv('EVCS_Usage_With_Google_Coordinates.csv')

# Remove rows with negative charging amounts
df = df[df['Total kWh'] >= 0]

# Remove rows with specific site
df = df[df['Site'] != '***TEST SITE*** Charge Your Car HQ']

# Convert 'Start Time' to datetime and extract the hour
df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.hour

# Calculate daily energy usage
df['Start Date'] = pd.to_datetime(df['Start Date'], format='%Y-%m-%d')
daily_energy = df.groupby(df['Start Date'].dt.date)['Total kWh'].sum().reset_index()
daily_energy.columns = ['Date', 'Total kWh']

# Calculate the frequency of each site
site_freq = df.groupby('Site').size().reset_index(name='Frequency')

# Merge frequency data back to original dataframe
df = pd.merge(df, site_freq, on='Site')

# Map for charging station locations and frequency
map_fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="Site", hover_data=["Frequency"],
                            size="Frequency", color="Frequency",
                            title="EV Charging Station Usage Frequency",
                            color_continuous_scale=px.colors.sequential.Plasma, size_max=15, zoom=10)
map_fig.update_layout(mapbox_style="open-street-map")

# Histogram for charging times
avg_hour = df['Start Time'].mean()
hist_fig = px.histogram(df, x='Start Time', nbins=24, title='Charging Time Histogram (24-Hour Format)', labels={'Start Time': 'Hour of the Day'})
hist_fig.update_xaxes(title_text='Hour of the Day', tickvals=list(range(24)), ticktext=[f'{i}:00' for i in range(24)])
hist_fig.update_yaxes(title_text='Frequency')
hist_fig.add_vline(x=avg_hour, line_dash='dash', line_color='red', annotation_text='Average Hour', annotation_position='top left')

# Pie chart for charging amount distribution
bins = [0, 20, 40, 60, 80, 100, 120, 140]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120+']
df['Charging Amount Bin'] = pd.cut(df['Total kWh'], bins=bins, labels=labels, right=False)
pie_fig = px.pie(df, names='Charging Amount Bin', title='Charging Amount Distribution', labels={'Charging Amount Bin': 'Charging Amount Bin'})

# Daily energy usage line chart
daily_energy_fig = px.line(daily_energy, x='Date', y='Total kWh', title='Daily Energy Usage (Total kWh)', labels={'Date': 'Date', 'Total kWh': 'Total kWh'})

# Heatmap for charging amount vs. time
heatmap_data = df.groupby(['Start Time', 'Total kWh']).size().reset_index(name='Count')
heatmap_fig = px.density_heatmap(heatmap_data, x='Start Time', y='Total kWh', z='Count',
                                color_continuous_scale='OrRd',  # Gradient from purple (low frequency) to yellow (high frequency)
                                title='Charging Amount vs. Charging Time Heatmap',
                                labels={'Start Time': 'Hour of the Day', 'Total kWh': 'Charging Amount (kWh)', 'Count': 'Frequency'})
heatmap_fig.update_xaxes(title_text='Hour of the Day', tickvals=list(range(24)), ticktext=[f'{i}:00' for i in range(24)])
heatmap_fig.update_yaxes(title_text='Charging Amount (kWh)')

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout with adjusted spacing
app.layout = html.Div([
    html.H1('EV Charging Dashboard'),
    dcc.Graph(figure=map_fig, style={'height': '50vh'}),  # Map at the top
    html.Div([
        dcc.Graph(figure=pie_fig, style={'width': '50%','height': '50vh'}),  # Pie chart on the left
        dcc.Graph(figure=heatmap_fig, style={'width': '50%','height': '50vh'})  # Heatmap on the right
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(figure=daily_energy_fig, style={'height': '50vh'}),  # Daily energy usage chart
    dcc.Graph(figure=hist_fig, style={'height': '50vh'})  # Histogram at the bottom
], style={'padding': '10px'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
