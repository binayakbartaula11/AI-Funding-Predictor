import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Application Configuration
# Sets the browser tab title, icon, and overall layout for optimal viewing
st.set_page_config(
    page_title="AI/ML Startup Funding Forecast",
    page_icon="📈",
    layout="wide"
)

# Main Application Header
# Presents the app title and explains the methodology of each forecasting technique
st.title("📊 AI/ML Startup Funding Forecast (2018-2025)")
st.markdown("""
This app calculates and visualizes forecasts for AI/ML startup funding using different moving average methods:

**Simple Moving Average (SMA)**  
`SMA(t) = (a + b + c) / 3`  
Where a, b, and c represent the values from the 3 most recent years.

**Weighted Moving Average (WMA)**  
`WMA(t) = (a×1 + b×2 + c×4) / (1 + 2 + 4)`  
Where more recent years get higher weights (oldest=1, middle=2, newest=4).

**Exponential Moving Average (EMA)**  
`EMA(t) = Previous_EMA + α(Current_Value - Previous_EMA)`  
Where α is the smoothing factor between 0 and 1.
""")

# Configuration Panel
# Sidebar containing all user-adjustable parameters for customizing the forecast
st.sidebar.header("Forecast Parameters")

# Historical Dataset
# Default funding data in billions USD from 2018-2024 for AI/ML startups
years_default = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
funding_default = [22.1, 26.6, 33.0, 66.8, 48.0, 50.0, 57.5]  # Using midpoint of 55-60 for 2024

# Data Input Controls
# Allows users to use default dataset or input custom historical values
st.sidebar.subheader("Historical Data")
use_default = st.sidebar.checkbox("Use Default Data", value=True)

if use_default:
    years = years_default
    funding = funding_default
else:
    # Custom input fields for each historical year
    funding = []
    for i, year in enumerate(years_default):
        funding.append(st.sidebar.number_input(f"Funding for {year} ($B)", 
                                              min_value=0.0, 
                                              max_value=500.0, 
                                              value=funding_default[i],
                                              step=0.1))
    years = years_default

# EMA Configuration
# Controls the responsiveness of EMA to recent data changes
smoothing_factor = st.sidebar.slider("EMA Smoothing Factor (α)", min_value=0.1, max_value=0.9, value=0.5, step=0.1)

# Data Structure Creation
# Organizes input data into pandas DataFrame for analysis
df = pd.DataFrame({
    'Year': years,
    'Funding': funding
})

# SMA Implementation
# Calculates simple moving average with explicit window handling
def calculate_sma(data, window_size=3):
    """Calculate SMA with specified window size
    
    SMA(t) = (a + b + c) / 3
    Where a, b, c are values from the last 3 years
    """
    sma_values = []
    
    # First window_size values should be NaN since we don't have enough history
    for i in range(window_size - 1):
        sma_values.append(np.nan)
    
    # Calculate SMA for remaining positions starting from window_size index
    for i in range(window_size - 1, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        sma_values.append(np.mean(window_values))
    
    return pd.Series(sma_values, index=data.index)

# WMA Implementation
# Calculates weighted moving average with customizable weights
def calculate_wma(data, window_size=3, weights=None):
    """Calculate WMA with specified window size and weights
    
    WMA(t) = (a×1 + b×2 + c×4) / (1 + 2 + 4)
    Where a, b, c are values from oldest to newest with corresponding weights
    """
    if weights is None:
        weights = np.array([1, 2, 4])  # Default weights for oldest to newest
    
    # Normalize weights to sum to 1
    norm_weights = weights / weights.sum()
    
    wma_values = []
    
    # First window_size values should be NaN since we don't have enough history
    for i in range(window_size):
        wma_values.append(np.nan)
    
    # Calculate WMA for remaining positions starting from window_size index
    for i in range(window_size, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        wma_values.append(np.sum(window_values * norm_weights))
    
    return pd.Series(wma_values, index=data.index)

# WMA Implementation
# Calculates weighted moving average with customizable weights
def calculate_wma(data, window_size=3, weights=None):
    """Calculate WMA with specified window size and weights
    
    WMA(t) = (a×1 + b×2 + c×4) / (1 + 2 + 4)
    Where a, b, c are values from oldest to newest with corresponding weights
    """
    if weights is None:
        weights = np.array([1, 2, 4])  # Default weights favor recent data
    
    # Normalize weights to ensure they sum to 1.0
    norm_weights = weights / weights.sum()
    
    wma_values = []
    
    # First n values are NaN since we need at least window_size points for calculation
    for i in range(window_size - 1):
        wma_values.append(np.nan)
    
    # Calculate WMA for each position once we have enough history
    for i in range(window_size - 1, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        wma_values.append(np.sum(window_values * norm_weights))
    
    return pd.Series(wma_values, index=data.index)

# EMA Implementation
# Calculates exponential moving average with adaptive smoothing factor
def calculate_ema(data, alpha=0.5):
    """Calculate EMA with specified smoothing factor (alpha)
    
    EMA(t) = Previous_EMA + α(Current_Value - Previous_EMA)
    Where α is the smoothing factor between 0 and 1
    """
    ema_values = [data.iloc[0]]  # Initialize with first observed value
    
    # Calculate EMA recursively for each subsequent position
    for i in range(1, len(data)):
        current_value = data.iloc[i]
        previous_ema = ema_values[-1]
        # Apply EMA formula: previous forecast + portion of current error
        ema_t = previous_ema + alpha * (current_value - previous_ema)
        ema_values.append(ema_t)
    
    return pd.Series(ema_values, index=data.index)

# Moving Average Calculation
# Apply each forecasting method to the historical data
df['SMA'] = calculate_sma(df['Funding'])
df['WMA'] = calculate_wma(df['Funding'], weights=np.array([1, 2, 4]))
df['EMA'] = calculate_ema(df['Funding'], alpha=smoothing_factor)

# Generate 2025 Forecasts
# Calculate next-year projections based on most recent moving averages
if len(funding) >= 3:
    # Use most recent values as forecasts for next period
    sma_forecast = df['SMA'].iloc[-1]
    wma_forecast = df['WMA'].iloc[-1]
else:
    # Handle insufficient data case
    sma_forecast = np.nan
    wma_forecast = np.nan

# EMA requires at least 1 value to calculate
if not np.isnan(df['EMA'].iloc[-1]):
    ema_forecast = df['EMA'].iloc[-1]
else:
    ema_forecast = np.nan

# Extend Dataset with Forecasts
# Add 2025 row with forecast values to the DataFrame
forecast_2025 = pd.DataFrame({
    'Year': [2025],
    'Funding': [None],  # No actual value for 2025 yet
    'SMA': [sma_forecast],
    'WMA': [wma_forecast],
    'EMA': [ema_forecast]
})

df_with_forecast = pd.concat([df, forecast_2025], ignore_index=True)

# Forecast Display
# Show forecast metrics with comparison to previous year
st.header("2025 Forecasts")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("SMA Forecast", f"${sma_forecast:.2f}B", 
              f"{((sma_forecast/funding[-1])-1)*100:.1f}% vs 2024")

with col2:
    st.metric("WMA Forecast", f"${wma_forecast:.2f}B", 
              f"{((wma_forecast/funding[-1])-1)*100:.1f}% vs 2024")

with col3:
    st.metric("EMA Forecast", f"${ema_forecast:.2f}B", 
              f"{((ema_forecast/funding[-1])-1)*100:.1f}% vs 2024")

# Data Table
# Display complete dataset with actual values and calculated moving averages
st.header("Historical Data and Moving Averages")
styled_df = df_with_forecast.style.format({
    'Funding': '${:.2f}B',
    'SMA': '${:.2f}B',
    'WMA': '${:.2f}B',
    'EMA': '${:.2f}B'
})
st.dataframe(styled_df)

# Visualization Section
# Interactive chart showing historical trends and forecast projections
st.header("Visualization")
fig = make_subplots(specs=[[{"secondary_y": False}]])

# Historical funding line with markers
fig.add_trace(
    go.Scatter(x=df_with_forecast['Year'], y=df_with_forecast['Funding'], 
               mode='lines+markers', name='Actual Funding',
               line=dict(color='royalblue', width=3))
)

# Moving average trend lines
fig.add_trace(
    go.Scatter(x=df_with_forecast['Year'], y=df_with_forecast['SMA'], 
               mode='lines', name='SMA (3-year)',
               line=dict(color='red', width=2, dash='dash'))
)

fig.add_trace(
    go.Scatter(x=df_with_forecast['Year'], y=df_with_forecast['WMA'], 
               mode='lines', name='WMA (weights: 1,2,4)',
               line=dict(color='green', width=2, dash='dash'))
)

fig.add_trace(
    go.Scatter(x=df_with_forecast['Year'], y=df_with_forecast['EMA'], 
               mode='lines', name=f'EMA (α={smoothing_factor})',
               line=dict(color='purple', width=2, dash='dash'))
)

# 2025 forecast markers
fig.add_trace(
    go.Scatter(x=[2025], y=[sma_forecast], mode='markers',
               marker=dict(color='red', size=12, symbol='star'),
               name='SMA Forecast')
)

fig.add_trace(
    go.Scatter(x=[2025], y=[wma_forecast], mode='markers',
               marker=dict(color='green', size=12, symbol='star'),
               name='WMA Forecast')
)

fig.add_trace(
    go.Scatter(x=[2025], y=[ema_forecast], mode='markers',
               marker=dict(color='purple', size=12, symbol='star'),
               name='EMA Forecast')
)

# Chart layout and styling
fig.update_layout(
    title_text='AI/ML Startup Funding Trends and Forecasts',
    xaxis_title='Year',
    yaxis_title='Funding (Billions $)',
    legend_title='Legend',
    hovermode='x unified',  # Shows all values for a given year on hover
    height=600
)

# Ensure x-axis shows all years including forecast year
fig.update_xaxes(tickvals=list(range(2018, 2026)))

st.plotly_chart(fig, use_container_width=True)

# Forecast Accuracy Analysis
# Evaluates historical performance of each forecasting method
if len(years) > 3:
    st.header("Forecast Error Analysis")
    
    # Prepare error analysis dataset from historical periods
    error_df = pd.DataFrame()
    
    # Use only data points where we have both actual and predicted values
    # Starting from 2021 (index 3) where all three models have predictions
    valid_indices = df.iloc[3:].index
    
    error_df['Year'] = df['Year'].iloc[valid_indices]
    error_df['Actual'] = df['Funding'].iloc[valid_indices].values
    error_df['SMA_Pred'] = df['SMA'].iloc[valid_indices].values
    error_df['WMA_Pred'] = df['WMA'].iloc[valid_indices].values
    error_df['EMA_Pred'] = df['EMA'].iloc[valid_indices].values
    
    # Calculate prediction errors (actual - predicted)
    error_df['SMA_Error'] = error_df['Actual'] - error_df['SMA_Pred']
    error_df['WMA_Error'] = error_df['Actual'] - error_df['WMA_Pred']
    error_df['EMA_Error'] = error_df['Actual'] - error_df['EMA_Pred']
    
    # Calculate squared errors for MSE computation
    error_df['SMA_Sq_Error'] = error_df['SMA_Error'] ** 2
    error_df['WMA_Sq_Error'] = error_df['WMA_Error'] ** 2
    error_df['EMA_Sq_Error'] = error_df['EMA_Error'] ** 2
    
    # Display formatted error metrics table
    styled_error_df = error_df.style.format({
        'Actual': '${:.2f}B',
        'SMA_Pred': '${:.2f}B',
        'WMA_Pred': '${:.2f}B',
        'EMA_Pred': '${:.2f}B',
        'SMA_Error': '${:.2f}B',
        'WMA_Error': '${:.2f}B',
        'EMA_Error': '${:.2f}B',
        'SMA_Sq_Error': '${:.2f}B²',
        'WMA_Sq_Error': '${:.2f}B²',
        'EMA_Sq_Error': '${:.2f}B²',
    })
    
    st.dataframe(styled_error_df)
    
    # Calculate mean squared error for each model
    mse_sma = error_df['SMA_Sq_Error'].mean()
    mse_wma = error_df['WMA_Sq_Error'].mean()
    mse_ema = error_df['EMA_Sq_Error'].mean()
    
    # Display MSE comparison
    st.subheader("Mean Squared Error (MSE)")
    mse_col1, mse_col2, mse_col3 = st.columns(3)
    
    with mse_col1:
        st.metric("SMA MSE", f"${mse_sma:.2f}B²")
        
    with mse_col2:
        st.metric("WMA MSE", f"${mse_wma:.2f}B²")
        
    with mse_col3:
        st.metric("EMA MSE", f"${mse_ema:.2f}B²")
    
    # Model recommendation based on historical accuracy
    best_model = min([(mse_sma, "SMA"), (mse_wma, "WMA"), (mse_ema, "EMA")], key=lambda x: x[0])[1]
    st.success(f"Based on historical forecast accuracy, the **{best_model}** model performs best with the lowest Mean Squared Error.")

# Detailed Methodology Section
# Explains the mathematical calculations behind each forecast model
st.header("Forecast Formula Details")

# SMA Calculation Explanation
st.subheader("Simple Moving Average (SMA)")
sma_calc = ""
if len(funding) >= 3:
    sma_calc = f"""
For our 2025 forecast:
```
SMA(2025) = SMA(2024) = (Funding_2022 + Funding_2023 + Funding_2024) / 3
          = (${funding[-3]:.1f}B + ${funding[-2]:.1f}B + ${funding[-1]:.1f}B) / 3
          = ${sum(funding[-3:]):.1f}B / 3
          = ${sma_forecast:.2f}B
```
"""

st.markdown(f"""
The SMA calculates the arithmetic mean of values over the last 3 years:

```
SMA(t) = (a + b + c) / 3
```
Where a, b, and c are the funding values for the past 3 years.
{sma_calc}
""")

# WMA Calculation Explanation
st.subheader("Weighted Moving Average (WMA)")
wma_calc = ""
if len(funding) >= 3:
    # Step-by-step WMA calculation for transparency
    a, b, c = funding[-3], funding[-2], funding[-1]
    weighted_sum = a*1 + b*2 + c*4
    weights_sum = 1 + 2 + 4
    manual_wma = weighted_sum / weights_sum
    
    wma_calc = f"""
For our 2025 forecast:
```
WMA(2025) = WMA(2024) = (Funding_2022×1 + Funding_2023×2 + Funding_2024×4) / (1 + 2 + 4)
          = (${a:.1f}B×1 + ${b:.1f}B×2 + ${c:.1f}B×4) / 7
          = (${a*1:.1f}B + ${b*2:.1f}B + ${c*4:.1f}B) / 7
          = ${weighted_sum:.1f}B / 7
          = ${manual_wma:.2f}B
```
"""

st.markdown(f"""
The WMA assigns higher weights to more recent data points:

```
WMA(t) = (a×1 + b×2 + c×4) / (1 + 2 + 4)
```
Where a, b, and c are the funding values for the past 3 years, with weights 1, 2, and 4 respectively.
{wma_calc}
""")

# EMA Calculation Explanation
st.subheader("Exponential Moving Average (EMA)")
ema_calc = ""
if len(funding) >= 2:
    # Demonstrate most recent EMA calculation step
    current = funding[-1]
    if len(funding) >= 3:
        previous_ema_value = df['EMA'].iloc[-2]
        ema_calc = f"""
For our most recent EMA calculation (2024):
```
EMA(2024) = EMA(2023) + α × (Funding_2024 - EMA(2023))
          = ${previous_ema_value:.2f}B + {smoothing_factor} × (${current:.2f}B - ${previous_ema_value:.2f}B)
          = ${previous_ema_value:.2f}B + {smoothing_factor} × ${current - previous_ema_value:.2f}B
          = ${previous_ema_value:.2f}B + ${smoothing_factor * (current - previous_ema_value):.2f}B
          = ${ema_forecast:.2f}B
```

For 2025 forecast, since we don't have an actual 2025 value yet, we use the last calculated EMA value:
```
EMA(2025) = EMA(2024) = ${ema_forecast:.2f}B
```
"""

st.markdown(f"""
The EMA applies exponentially decreasing weights to past observations:

```
EMA(t) = Previous_EMA + α(Current_Value - Previous_EMA)
```

Where:
* α is the smoothing factor ({smoothing_factor})
* Previous_EMA is the last calculated EMA value
* Current_Value is the latest observed value

{ema_calc}
""")

# Footer and Attribution
# Displays copyright information and company social media links
st.markdown("---")
st.caption(
    "Built with Streamlit · AI/ML Startup Funding Forecast Tool"
)
