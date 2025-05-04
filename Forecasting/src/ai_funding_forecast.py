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

# Add data source reference
st.info("""
**Data Source:** Historical funding data (2018-2024) obtained from Perplexity research.  
[View Research Source](https://www.perplexity.ai/search/please-provide-structured-data-TuW86UxiRCykQcOQAXFIUg)
""")

# Configuration Panel
# Sidebar containing all user-adjustable parameters for customizing the forecast
st.sidebar.header("Forecast Parameters")

# Historical Dataset
# Funding data (2018-2024) obtained from research via Perplexity, in billions USD for AI/ML startups
years_default = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
funding_default = [9.3, 26.6, 36, 68, 45.8, 42.5, 100.4]

# Data Input Controls
# Allows users to use research data or input custom historical values
st.sidebar.subheader("Historical Data")
use_research = st.sidebar.checkbox("Use Research Data (Perplexity)", value=True)

if use_research:
    years = years_default
    funding = funding_default
else:
    # Custom input fields for each historical year - FIX: Ensure all types are float
    funding = []
    for i, year in enumerate(years_default):
        funding.append(st.sidebar.number_input(f"Funding for {year} ($B)", 
                                              min_value=0.0, 
                                              max_value=500.0, 
                                              value=float(funding_default[i]),  # Convert to float
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

# Calculate year-over-year growth percentages
df['YoY_Growth'] = df['Funding'].pct_change() * 100

# Calculate Simple Moving Average (SMA) with 3-year window
def calculate_sma(data, window_size=3):
    """Calculate SMA with specified window size
    
    SMA(t) = (a + b + c) / 3
    Where a, b, and c are the 3 most recent values
    """
    sma_values = []
    
    # First window_size values should be NaN since we don't have enough history
    for i in range(window_size):
        sma_values.append(np.nan)
    
    # Calculate SMA for remaining positions starting from window_size index
    for i in range(window_size, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        sma_values.append(np.mean(window_values))
    
    return pd.Series(sma_values, index=data.index)

# Calculate Weighted Moving Average (WMA) with specific weights
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
    # Calculate SMA forecast using the last 3 years
    last_3_years = df['Funding'].iloc[-3:].values
    sma_forecast = np.mean(last_3_years)
    
    # Calculate WMA forecast using the last 3 years with weights [1,2,4]
    weights = np.array([1, 2, 4])
    wma_forecast = np.sum(last_3_years * weights) / np.sum(weights)
else:
    # Handle insufficient data case
    sma_forecast = np.nan
    wma_forecast = np.nan

# EMA forecast is the last EMA value
ema_forecast = df['EMA'].iloc[-1] if len(df) > 0 else np.nan

# Extend Dataset with Forecasts
# Add 2025 row with forecast values to the DataFrame
forecast_2025 = pd.DataFrame({
    'Year': [2025],
    'Funding': [None],  # No actual value for 2025 yet
    'YoY_Growth': [None],  # No YoY growth for forecast yet
    'SMA': [sma_forecast],
    'WMA': [wma_forecast],
    'EMA': [ema_forecast]
})

df_with_forecast = pd.concat([df, forecast_2025], ignore_index=True)

# Calculate growth rates for forecasts compared to 2024
if len(funding) > 0:
    last_actual = funding[-1]
    sma_growth = ((sma_forecast / last_actual) - 1) * 100 if not np.isnan(sma_forecast) else np.nan
    wma_growth = ((wma_forecast / last_actual) - 1) * 100 if not np.isnan(wma_forecast) else np.nan
    ema_growth = ((ema_forecast / last_actual) - 1) * 100 if not np.isnan(ema_forecast) else np.nan
else:
    sma_growth = wma_growth = ema_growth = np.nan

# Forecast Display
# Show forecast metrics with comparison to previous year
st.header("2025 Forecasts")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("SMA Forecast", f"${sma_forecast:.2f}B", 
              f"{sma_growth:.1f}% vs 2024" if not np.isnan(sma_growth) else "N/A")

with col2:
    st.metric("WMA Forecast", f"${wma_forecast:.2f}B", 
              f"{wma_growth:.1f}% vs 2024" if not np.isnan(wma_growth) else "N/A")

with col3:
    st.metric("EMA Forecast", f"${ema_forecast:.2f}B", 
              f"{ema_growth:.1f}% vs 2024" if not np.isnan(ema_growth) else "N/A")

# Data Table
# Display complete dataset with actual values and calculated moving averages
st.header("Historical Data and Moving Averages")
styled_df = df_with_forecast.style.format({
    'Funding': '${:.2f}B',
    'YoY_Growth': '{:.1f}%',
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

# Year-over-Year Growth Visualization
st.header("Year-over-Year Growth Rates")
growth_fig = go.Figure()

growth_fig.add_trace(
    go.Bar(x=df['Year'], y=df['YoY_Growth'], 
           name='YoY Growth', 
           marker_color='teal')
)

growth_fig.update_layout(
    title_text='Annual Growth in AI/ML Startup Funding',
    xaxis_title='Year',
    yaxis_title='Growth Rate (%)',
    height=400,
    yaxis=dict(
        range=[0, 240],
        tickmode='array',
        tickvals=[0, 80, 160, 240],
        ticktext=['0', '80', '160', '240']
    )
)

# Add a horizontal line at 0%
growth_fig.add_shape(
    type='line',
    x0=min(df['Year'])-0.5,
    y0=0,
    x1=max(df['Year'])+0.5,
    y1=0,
    line=dict(color='gray', width=1, dash='dash')
)

# Display growth labels on bars
growth_fig.update_traces(
    texttemplate='%{y:.1f}%',
    textposition='outside'
)

st.plotly_chart(growth_fig, use_container_width=True)

# Forecast Error Analysis with multiple metrics
# Evaluates historical performance of each forecasting method
if len(years) > 3:
    st.header("Forecast Error Analysis")
    
    # Prepare error analysis dataset from historical periods
    error_df = pd.DataFrame()
    
    # Use only data points where we have both actual and predicted values
    valid_indices = df.index[~df['SMA'].isna() & ~df['WMA'].isna() & ~df['EMA'].isna()]
    
    error_df['Year'] = df['Year'].iloc[valid_indices]
    error_df['Actual'] = df['Funding'].iloc[valid_indices].values
    error_df['SMA_Pred'] = df['SMA'].iloc[valid_indices].values
    error_df['WMA_Pred'] = df['WMA'].iloc[valid_indices].values
    error_df['EMA_Pred'] = df['EMA'].iloc[valid_indices].values
    
    # Calculate absolute errors
    error_df['SMA_Error'] = error_df['Actual'] - error_df['SMA_Pred']
    error_df['WMA_Error'] = error_df['Actual'] - error_df['WMA_Pred']
    error_df['EMA_Error'] = error_df['Actual'] - error_df['EMA_Pred']
    
    # Calculate absolute errors (for MAE)
    error_df['SMA_Abs_Error'] = np.abs(error_df['SMA_Error'])
    error_df['WMA_Abs_Error'] = np.abs(error_df['WMA_Error'])
    error_df['EMA_Abs_Error'] = np.abs(error_df['EMA_Error'])
    
    # Calculate squared errors (for MSE)
    error_df['SMA_Sq_Error'] = error_df['SMA_Error'] ** 2
    error_df['WMA_Sq_Error'] = error_df['WMA_Error'] ** 2
    error_df['EMA_Sq_Error'] = error_df['EMA_Error'] ** 2
    
    # Calculate percentage errors (for MAPE)
    error_df['SMA_Pct_Error'] = np.abs(error_df['SMA_Error'] / error_df['Actual']) * 100
    error_df['WMA_Pct_Error'] = np.abs(error_df['WMA_Error'] / error_df['Actual']) * 100
    error_df['EMA_Pct_Error'] = np.abs(error_df['EMA_Error'] / error_df['Actual']) * 100
    
    # Display formatted error metrics table
    styled_error_df = error_df[['Year', 'Actual', 'SMA_Pred', 'WMA_Pred', 'EMA_Pred', 
                                'SMA_Error', 'WMA_Error', 'EMA_Error']].style.format({
        'Actual': '${:.2f}B',
        'SMA_Pred': '${:.2f}B',
        'WMA_Pred': '${:.2f}B',
        'EMA_Pred': '${:.2f}B',
        'SMA_Error': '${:.2f}B',
        'WMA_Error': '${:.2f}B',
        'EMA_Error': '${:.2f}B',
    })
    
    st.subheader("Error Values")
    st.dataframe(styled_error_df)
    
    # Calculate comprehensive error metrics
    # 1. Mean Squared Error (MSE)
    mse_sma = error_df['SMA_Sq_Error'].mean()
    mse_wma = error_df['WMA_Sq_Error'].mean()
    mse_ema = error_df['EMA_Sq_Error'].mean()
    
    # 2. Mean Absolute Error (MAE)
    mae_sma = error_df['SMA_Abs_Error'].mean()
    mae_wma = error_df['WMA_Abs_Error'].mean()
    mae_ema = error_df['EMA_Abs_Error'].mean()
    
    # 3. Mean Absolute Percentage Error (MAPE)
    mape_sma = error_df['SMA_Pct_Error'].mean()
    mape_wma = error_df['WMA_Pct_Error'].mean()
    mape_ema = error_df['EMA_Pct_Error'].mean()
    
    # 4. Theil's U Statistic (measure of forecast accuracy)
    # U = sqrt(sum((forecast_t - actual_t)²) / sum(actual_t²))
    u_sma = np.sqrt(np.sum(error_df['SMA_Sq_Error']) / np.sum(error_df['Actual']**2))
    u_wma = np.sqrt(np.sum(error_df['WMA_Sq_Error']) / np.sum(error_df['Actual']**2))
    u_ema = np.sqrt(np.sum(error_df['EMA_Sq_Error']) / np.sum(error_df['Actual']**2))
    
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 
                  'Mean Absolute Percentage Error (MAPE)', "Theil's U Statistic"],
        'SMA': [mse_sma, mae_sma, mape_sma, u_sma],
        'WMA': [mse_wma, mae_wma, mape_wma, u_wma],
        'EMA': [mse_ema, mae_ema, mape_ema, u_ema]
    })
    
    # Format the metrics table
    st.subheader("Error Metrics Summary")
    metrics_styled = metrics_df.style.format({
        'SMA': lambda x: f'${x:.2f}B²' if x > 1 else f'{x:.4f}',
        'WMA': lambda x: f'${x:.2f}B²' if x > 1 else f'{x:.4f}',
        'EMA': lambda x: f'${x:.2f}B²' if x > 1 else f'{x:.4f}',
    }, subset=['SMA', 'WMA', 'EMA'])
    
    st.dataframe(metrics_styled)
    
    # Display MSE comparison
    st.subheader("Mean Squared Error (MSE)")
    mse_col1, mse_col2, mse_col3 = st.columns(3)
    
    with mse_col1:
        st.metric("SMA MSE", f"${mse_sma:.2f}B²")
        
    with mse_col2:
        st.metric("WMA MSE", f"${mse_wma:.2f}B²")
        
    with mse_col3:
        st.metric("EMA MSE", f"${mse_ema:.2f}B²")
    
    # Model recommendation based on multiple metrics
    best_models = {}
    best_models['MSE'] = min([(mse_sma, "SMA"), (mse_wma, "WMA"), (mse_ema, "EMA")], key=lambda x: x[0])[1]
    best_models['MAE'] = min([(mae_sma, "SMA"), (mae_wma, "WMA"), (mae_ema, "EMA")], key=lambda x: x[0])[1]
    best_models['MAPE'] = min([(mape_sma, "SMA"), (mape_wma, "WMA"), (mape_ema, "EMA")], key=lambda x: x[0])[1]
    best_models['Theil'] = min([(u_sma, "SMA"), (u_wma, "WMA"), (u_ema, "EMA")], key=lambda x: x[0])[1]
    
    # Count model occurrences to find overall best
    model_counts = {"SMA": 0, "WMA": 0, "EMA": 0}
    for metric, model in best_models.items():
        model_counts[model] += 1
    
    overall_best = max(model_counts.items(), key=lambda x: x[1])[0]
    
    st.success(f"""
    Based on historical forecast accuracy:
    - Best model by MSE: **{best_models['MSE']}**
    - Best model by MAE: **{best_models['MAE']}**  
    - Best model by MAPE: **{best_models['MAPE']}**
    - Best model by Theil's U: **{best_models['Theil']}**
    
    **Overall recommendation:** The **{overall_best}** model performs best across multiple error metrics.
    """)

# Error Visualization
if len(years) > 3:
    st.header("Error Visualization")
    
    error_fig = go.Figure()
    
    # Add error bars for each model
    error_fig.add_trace(
        go.Bar(x=error_df['Year'], y=error_df['SMA_Error'], 
               name='SMA Error', marker_color='red', opacity=0.7)
    )
    
    error_fig.add_trace(
        go.Bar(x=error_df['Year'], y=error_df['WMA_Error'], 
               name='WMA Error', marker_color='green', opacity=0.7)
    )
    
    error_fig.add_trace(
        go.Bar(x=error_df['Year'], y=error_df['EMA_Error'], 
               name='EMA Error', marker_color='purple', opacity=0.7)
    )
    
    # Add horizontal line at zero
    error_fig.add_shape(
        type='line',
        x0=min(error_df['Year'])-0.5,
        y0=0,
        x1=max(error_df['Year'])+0.5,
        y1=0,
        line=dict(color='black', width=1.5)
    )
    
    error_fig.update_layout(
        title_text='Forecast Errors by Model (Actual - Forecast)',
        xaxis_title='Year',
        yaxis_title='Error (Billions $)',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(error_fig, use_container_width=True)

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
SMA(2025) = (Funding_2022 + Funding_2023 + Funding_2024) / 3
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
WMA(2025) = (Funding_2022×1 + Funding_2023×2 + Funding_2024×4) / (1 + 2 + 4)
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

# Forecast Methodology Comparison
st.subheader("Forecast Methodology Comparison")
st.markdown("""
| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **SMA** | Simple to calculate and understand | Equal weight to all periods; Slow to respond to recent trends | Stable markets with minimal volatility |
| **WMA** | Emphasizes recent data; Better trend responsiveness | More complex than SMA; Still limited by fixed weights | Markets with gradual trend changes |
| **EMA** | Highly responsive to recent changes; Adaptive over time | Can overreact to outliers; Requires tuning α parameter | Fast-changing markets with clear trends |
""")

# Footer and Attribution
# Displays copyright information and company social media links
st.markdown("---")
st.caption("© 2025 Binayak Bartaula | Built with Streamlit · AI/ML Startup Funding Forecast Tool")

# Add proper links to resources
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("[GitHub Repository](https://github.com/binayakbartaula11)", unsafe_allow_html=True)
with col2:
    st.markdown("[Excel Data](https://github.com/binayakbartaula11/ai-funding-predictor/blob/main/ai_funding_forecast.xlsx)", unsafe_allow_html=True)
with col3:
    st.markdown("[Data Source](https://www.perplexity.ai/search/please-provide-structured-data-TuW86UxiRCykQcOQAXFIUg)", unsafe_allow_html=True)
