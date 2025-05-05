# AI/ML Startup Funding Forecast: Technical Whitepaper

**Author**: Binayak Bartaula  
**Department**: Department of Computer Engineering  
**Institution**: Nepal College of Information Technology  
**Date**: May 2025

## Executive Summary

This technical whitepaper documents the methodologies, algorithms, and implementation details of the AI/ML Startup Funding Forecast application. The tool provides data-driven forecasts of venture capital funding in the artificial intelligence and machine learning sector, utilizing three distinct time series forecasting techniques: Simple Moving Average (SMA), Weighted Moving Average (WMA), and Exponential Moving Average (EMA). This document outlines the mathematical foundations, implementation considerations, forecast evaluation metrics, and technical architecture of the application.

## 1. Introduction

Forecasting venture capital funding in the rapidly evolving AI/ML sector presents unique challenges due to market volatility, technological breakthroughs, and shifting investor sentiment. This whitepaper explains the technical approach to predicting funding trends while accounting for these dynamics.

### 1.1 Purpose

The AI/ML Startup Funding Forecast application serves to:

1. Provide transparent, methodologically sound predictions of future funding levels
2. Compare multiple forecasting techniques using quantifiable error metrics
3. Empower stakeholders to test alternative scenarios via parameter adjustments
4. Visualize historical funding patterns against predicted trends

### 1.2 Historical Data Overview

The application utilizes historical funding data from 2018-2024, showing the following annual venture capital investments in AI/ML startups (in billions USD):

| Year | Funding ($B) |
|------|--------------|
| 2018 | 9.3          |
| 2019 | 26.6         |
| 2020 | 36.0         |
| 2021 | 68.0         |
| 2022 | 45.8         |
| 2023 | 42.5         |
| 2024 | 100.4        |

These figures demonstrate considerable volatility, with notable growth (2018-2021), a market correction (2022-2023), and substantial recovery (2024). This volatility underscores the need for robust forecasting methods that can adaptively respond to changing market conditions.

## 2. Forecast Methodologies

The application implements three established time series forecasting techniques, each with different strengths and mathematical properties. This section details their formulations, implementation nuances, and relative advantages.

### 2.1 Simple Moving Average (SMA)

#### 2.1.1 Mathematical Formulation

The SMA calculates the arithmetic mean of values over a specified lookback window:

$$\text{SMA}_t = \frac{1}{n} \sum_{i=t-n+1}^{t} x_i$$

Where:
- $\text{SMA}_t$ is the Simple Moving Average at time $t$
- $n$ is the window size (3 years in our implementation)
- $x_i$ represents the funding value at time $i$

#### 2.1.2 Implementation Details

The SMA implementation handles edge cases where insufficient historical data exists:

```python
def calculate_sma(data, window_size=3):
    """Calculate SMA with specified window size"""
    sma_values = []
    
    # First window_size-1 values should be NaN since we don't have enough history
    for i in range(window_size - 1):
        sma_values.append(np.nan)
    
    # Calculate SMA for remaining positions starting from window_size-1 index
    for i in range(window_size - 1, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        sma_values.append(np.mean(window_values))
    
    return pd.Series(sma_values, index=data.index)
```

#### 2.1.3 Strengths and Limitations

**Strengths:**
- Simple to calculate and understand
- Reduces the impact of short-term fluctuations
- Equal weighting creates stability in predictions

**Limitations:**
- Lags behind in responsive adaptation to recent trends
- Equal weights fail to prioritize more recent (potentially more relevant) data
- Cannot adapt to structural market changes quickly

### 2.2 Weighted Moving Average (WMA)

#### 2.2.1 Mathematical Formulation

The WMA applies custom weights to each period within the window, prioritizing more recent observations:

$$\text{WMA}_t = \frac{\sum_{i=t-n+1}^{t} w_i \cdot x_i}{\sum_{i=t-n+1}^{t} w_i}$$

Where:
- $\text{WMA}_t$ is the Weighted Moving Average at time $t$
- $n$ is the window size (3 years in our implementation)
- $x_i$ represents the funding value at time $i$
- $w_i$ represents the weight applied to time $i$ (weights of [1,2,4] in our implementation)

#### 2.2.2 Implementation Details

The WMA implementation normalizes weights and handles edge cases:

```python
def calculate_wma(data, window_size=3, weights=None):
    """Calculate WMA with specified window size and weights"""
    if weights is None:
        weights = np.array([1, 2, 4])  # Default weights favor recent data
    
    # Normalize weights to ensure they sum to 1.0
    norm_weights = weights / weights.sum()
    
    wma_values = []
    
    # First n-1 values are NaN since we need at least window_size points
    for i in range(window_size - 1):
        wma_values.append(np.nan)
    
    # Calculate WMA for each position once we have enough history
    for i in range(window_size - 1, len(data)):
        window_values = data[i-(window_size-1):i+1].values
        wma_values.append(np.sum(window_values * norm_weights))
    
    return pd.Series(wma_values, index=data.index)
```

#### 2.2.3 Strengths and Limitations

**Strengths:**
- Prioritizes recent observations while still considering historical context
- More responsive to emerging trends than SMA
- Fixed weights provide predictable behavior

**Limitations:**
- Requires selection of appropriate weight distribution
- Fixed weights lack adaptive properties over time
- Still affected by outliers in the most heavily weighted period

### 2.3 Exponential Moving Average (EMA)

#### 2.3.1 Mathematical Formulation

The EMA calculates a weighted average that applies exponentially decreasing weights to past observations:

$$\text{EMA}_t = \text{EMA}_{t-1} + \alpha \cdot (x_t - \text{EMA}_{t-1})$$

Where:
- $\text{EMA}_t$ is the Exponential Moving Average at time $t$
- $\alpha$ is the smoothing factor between 0 and 1
- $x_t$ is the funding value at time $t$
- $\text{EMA}_{t-1}$ is the previously calculated EMA value

This can be expanded to show the exponential weighting explicitly:

$$\text{EMA}_t = \alpha \cdot x_t + \alpha(1-\alpha) \cdot x_{t-1} + \alpha(1-\alpha)^2 \cdot x_{t-2} + ... + \alpha(1-\alpha)^{t-1} \cdot x_1 + (1-\alpha)^t \cdot \text{EMA}_0$$

#### 2.3.2 Implementation Details

The EMA implementation initializes with the first observed value and recursively applies the EMA formula:

```python
def calculate_ema(data, alpha=0.5):
    """Calculate EMA with specified smoothing factor (alpha)"""
    ema_values = [data.iloc[0]]  # Initialize with first observed value
    
    # Calculate EMA recursively for each subsequent position
    for i in range(1, len(data)):
        current_value = data.iloc[i]
        previous_ema = ema_values[-1]
        # Apply EMA formula: previous forecast + portion of current error
        ema_t = previous_ema + alpha * (current_value - previous_ema)
        ema_values.append(ema_t)
    
    return pd.Series(ema_values, index=data.index)
```

#### 2.3.3 Strengths and Limitations

**Strengths:**
- Adaptive to recent changes with tunable responsiveness
- Incorporates entire history with diminishing weights
- Computationally efficient (requires storing only the previous EMA value)

**Limitations:**
- Smoothing factor selection significantly impacts performance
- Can overreact to outliers with high α values
- Complex interpretation due to exponential weighting

## 3. Forecast Evaluation Metrics

The application implements multiple quantitative metrics to evaluate forecast accuracy and select the most reliable forecasting method based on historical performance.

### 3.1 Mean Squared Error (MSE)

MSE measures the average of squared differences between predicted and actual values:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ is the actual funding value
- $\hat{y}_i$ is the forecasted funding value
- $n$ is the number of forecasts being evaluated

Implementation:
```python
mse_sma = error_df['SMA_Sq_Error'].mean()
mse_wma = error_df['WMA_Sq_Error'].mean()
mse_ema = error_df['EMA_Sq_Error'].mean()
```

### 3.2 Mean Absolute Error (MAE)

MAE measures the average absolute differences between predicted and actual values:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Implementation:
```python
mae_sma = error_df['SMA_Abs_Error'].mean()
mae_wma = error_df['WMA_Abs_Error'].mean()
mae_ema = error_df['EMA_Abs_Error'].mean()
```

### 3.3 Mean Absolute Percentage Error (MAPE)

MAPE expresses accuracy as a percentage of the error:

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

Implementation:
```python
mape_sma = error_df['SMA_Pct_Error'].mean()
mape_wma = error_df['WMA_Pct_Error'].mean()
mape_ema = error_df['EMA_Pct_Error'].mean()
```

### 3.4 Theil's U Statistic

Theil's U provides a normalized measure of forecast accuracy:

$$U = \sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}y_i^2}}$$

Implementation:
```python
u_sma = np.sqrt(np.sum(error_df['SMA_Sq_Error']) / np.sum(error_df['Actual']**2))
u_wma = np.sqrt(np.sum(error_df['WMA_Sq_Error']) / np.sum(error_df['Actual']**2))
u_ema = np.sqrt(np.sum(error_df['EMA_Sq_Error']) / np.sum(error_df['Actual']**2))
```

## 4. Application Architecture

### 4.1 Technology Stack

The application is built using the following technologies:

- **Streamlit**: Front-end framework providing interactive UI components
- **Pandas**: Data manipulation and time series operations
- **NumPy**: Numerical computations and array operations
- **Plotly**: Interactive visualization library for charts and graphs

### 4.2 Component Overview

The application's architecture consists of these major components:

1. **Configuration & Input Processing**
   - Parameter settings via sidebar
   - Data validation and preprocessing

2. **Computational Core**
   - Implementation of forecasting algorithms
   - Error metrics calculation
   - 2025 forecast generation

3. **Visualization Engine**
   - Time series plots of historical and forecast values
   - Error visualization
   - YoY growth representation

4. **Documentation Layer**
   - Formula explanations
   - Methodology comparison
   - Code transparency

### 4.3 Data Flow

The application follows this data flow pattern:

1. User supplies or confirms historical data inputs
2. Forecasting algorithms process historical data
3. Predictions for 2025 are generated
4. Error analysis evaluates historical forecast accuracy
5. Visualizations render the predictions and analysis
6. Recommendations are made based on error metrics

## 5. Future Enhancements

This section outlines planned technical improvements to enhance forecast accuracy and application functionality:

### 5.1 Advanced Forecasting Methods

- **ARIMA Models**: Implement Autoregressive Integrated Moving Average models
- **Machine Learning Approaches**: Neural networks for complex pattern recognition
- **Bayesian Forecasting**: Incorporate uncertainty quantification

### 5.2 External Variables Integration

- **Macroeconomic Indicators**: GDP growth, interest rates, tech sector performance
- **Funding Stage Analysis**: Separate forecasts by funding stage (seed, Series A, etc.)
- **Geographic Segmentation**: Regional funding analysis and prediction

### 5.3 Technical Improvements

- **Automated Parameter Optimization**: Grid search for optimal smoothing factors
- **Confidence Intervals**: Statistical bounds for forecast uncertainty
- **API Integration**: Real-time data ingestion from funding databases

## 6. Conclusion

The AI/ML Startup Funding Forecast application provides a robust, transparent framework for predicting venture capital trends in the artificial intelligence and machine learning sector. By implementing multiple forecasting methodologies and rigorous error analysis, the tool offers valuable insights for investors, founders, and market analysts in this dynamic industry.

The combination of Simple Moving Average, Weighted Moving Average, and Exponential Moving Average techniques enables users to balance forecast stability with responsiveness to recent market shifts. The comprehensive error metrics framework ensures that forecasts can be evaluated objectively and continuously improved.

---

## Appendix A: Mathematical Derivations

### A.1 EMA Recursive Formula Derivation

Starting with the definition:

$$\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}$$

Expanding for $\text{EMA}_{t-1}$:

$$\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot [\alpha \cdot x_{t-1} + (1-\alpha) \cdot \text{EMA}_{t-2}]$$

$$\text{EMA}_t = \alpha \cdot x_t + \alpha(1-\alpha) \cdot x_{t-1} + (1-\alpha)^2 \cdot \text{EMA}_{t-2}$$

Continuing this expansion leads to the complete formula:

$$\text{EMA}_t = \alpha \cdot x_t + \alpha(1-\alpha) \cdot x_{t-1} + \alpha(1-\alpha)^2 \cdot x_{t-2} + ... + \alpha(1-\alpha)^{t-1} \cdot x_1 + (1-\alpha)^t \cdot \text{EMA}_0$$

Where $\text{EMA}_0$ is the initialization value.

## Appendix B: References

1. Hyndman, R.J., & Athanasopoulos, G. (2018). [*Forecasting: Principles and Practice*](https://otexts.com/fpp2/). OTexts.

2. Box, G.E., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). [*Time Series Analysis: Forecasting and Control*](https://doi.org/10.1002/9781118619193). John Wiley & Sons.

3. Perplexity Research. (2024). [*Historical AI/ML Startup Funding Dataset 2018-2024*](https://www.perplexity.ai/search/please-provide-structured-data-TuW86UxiRCykQcOQAXFIUg).

4. Chatfield, C. (2000). [*Time-Series Forecasting*](https://doi.org/10.1201/9781420036206). CRC Press.

5. Montgomery, D.C., Jennings, C.L., & Kulahci, M. (2015). [*Introduction to Time Series Analysis and Forecasting*](https://doi.org/10.1002/9781119264034). John Wiley & Sons.

6. Taylor, S.J., & Letham, B. (2018). [*Forecasting at Scale*](https://doi.org/10.1080/00031305.2017.1380080). The American Statistician, 72(1), 37-45.

7. Diebold, F.X. (2017). [*Forecasting in Economics, Business, Finance and Beyond*](https://www.sas.upenn.edu/~fdiebold/Textbooks.html). Department of Economics, University of Pennsylvania.

8. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). [*The M4 Competition: 100,000 time series and 61 forecasting methods*](https://doi.org/10.1016/j.ijforecast.2019.04.014). International Journal of Forecasting, 36(1), 54-74.

9. Seabold, S., & Perktold, J. (2010). [*Statsmodels: Econometric and Statistical Modeling with Python*](https://conference.scipy.org/proceedings/scipy2010/seabold.html). Proceedings of the 9th Python in Science Conference.

10. McKinney, W. (2010). [*Data Structures for Statistical Computing in Python*](https://conference.scipy.org/proceedings/scipy2010/mckinney.html). Proceedings of the 9th Python in Science Conference.

## Appendix C: Online Resources

- [Project GitHub Repository](https://github.com/binayakbartaula11/ai-funding-predictor)
- [Interactive Web Application](https://ai-ml-funding-forecast.streamlit.app)
- [Raw Data & Excel Models](https://github.com/binayakbartaula11/ai-funding-predictor/blob/main/ai_funding_forecast.xlsx)
- [Data Source](https://www.perplexity.ai/search/please-provide-structured-data-TuW86UxiRCykQcOQAXFIUg)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)

---

*Copyright © 2025 Binayak Bartaula, Nepal College of Information Technology. All rights reserved.*
