# 📖 TCS Stock Analysis - User Guide

> **Complete user manual for the TCS Stock Analysis Dashboard and all its features**

## Table of Contents

- [Getting Started](#getting-started)
- [Dashboard Overview](#dashboard-overview)
- [Feature Walkthroughs](#feature-walkthroughs)
- [Data Upload Guide](#data-upload-guide)
- [Analysis Techniques](#analysis-techniques)
- [Model Usage](#model-usage)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### 🚀 Quick Start (5 Minutes)

1. **Launch the Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Access the Interface**
   - Open your browser to `http://localhost:8501`
   - You'll see the TCS Stock Analysis Dashboard

3. **Start Exploring**
   - Use the default TCS dataset (recommended for first-time users)
   - Navigate through the 4 main tabs: Overview, EDA, Predictions, Insights

### 🎯 What You Can Do

- **📊 Analyze Stock Performance**: View comprehensive price trends, volume patterns, and statistical summaries
- **🔬 Explore Data**: Interactive charts for dividends, stock splits, correlations, and distributions
- **🤖 Make Predictions**: Train machine learning models (Linear Regression & LSTM) for price forecasting
- **💡 Get Insights**: Market trend analysis, trading signals, and risk assessments
- **💾 Export Data**: Download predictions and analysis results as CSV files

---

## Dashboard Overview

### 🎛️ Sidebar Controls

The left sidebar contains all the main controls for customizing your analysis:

#### **📁 Dataset Options**
- **Use Default Dataset**: Pre-loaded TCS historical data (recommended)
- **Upload New Dataset**: Upload your own CSV file with stock data

#### **📅 Date Range Selection**
- **Start Date**: Choose the beginning of your analysis period
- **End Date**: Choose the end of your analysis period
- **💡 Tip**: Longer periods provide more comprehensive analysis but may slow down processing

#### **🤖 Model Selection**
- **Linear Regression**: Fast, interpretable model good for trend analysis
- **LSTM Neural Network**: Advanced deep learning model for complex patterns

#### **📊 Visualization Options**
- **Chart Type**: Line Chart, Candlestick, or OHLC
- **Analysis Type**: Price, Volume, Corporate Actions, or Technical Analysis

### 🏠 Main Interface Tabs

#### Tab 1: 📊 Overview
Quick summary and key metrics dashboard

#### Tab 2: 🔬 EDA (Exploratory Data Analysis)
Detailed charts and statistical analysis

#### Tab 3: 🤖 Predictions
Machine learning model training and forecasting

#### Tab 4: 💡 Insights
Market analysis, trends, and trading signals

---

## Feature Walkthroughs

### 📊 Tab 1: Overview - Your Dashboard Home

**Purpose**: Get a quick snapshot of TCS stock performance and key metrics.

#### **Key Metrics Cards**
1. **💰 Current Price**: Latest closing price in ₹
2. **📈/📉 Price Change**: Daily change in price and percentage
3. **📊 Total Volume**: Trading volume for the selected period
4. **📈 Average Volume**: Mean trading volume
5. **⚡ Volatility**: Annualized volatility percentage

#### **Summary Statistics Table**
- **Comprehensive Statistics**: Count, Mean, Standard Deviation, Min, Max, Quartiles
- **All Columns**: Open, High, Low, Close, Volume (if available)
- **Easy Interpretation**: Color-coded for quick insights

#### **Main Price Chart**
- **Interactive Plotly Chart**: Zoom, pan, hover for details
- **Multiple Chart Types**: 
  - 📈 **Line Chart**: Clean trend visualization
  - 🕯️ **Candlestick**: Traditional OHLC view with volume
  - 📊 **OHLC**: Bar-style OHLC representation

#### **🎯 How to Use Overview Tab**
1. **Quick Assessment**: Look at the metric cards for immediate insights
2. **Check the Chart**: Identify overall trends and patterns
3. **Review Statistics**: Understand the data distribution and characteristics
4. **Adjust Date Range**: Use sidebar to focus on specific periods

---

### 🔬 Tab 2: EDA - Deep Dive Analysis

**Purpose**: Comprehensive exploratory data analysis with advanced visualizations.

#### **Corporate Actions Analysis**

##### **💰 Dividends vs Close Price**
- **Top Panel**: Stock price line with dividend event markers (red diamonds)
- **Bottom Panel**: Bar chart showing dividend amounts over time
- **📍 Key Insights**: 
  - Correlation between dividend announcements and price movements
  - Dividend payment patterns and amounts
  - Impact of dividend policy on stock performance

##### **📈 Stock Splits vs Close Price**
- **Price Chart**: With stock split events marked as orange stars
- **Split Information**: Hover over markers to see split ratios
- **📍 Key Insights**:
  - Effect of stock splits on price and volume
  - Historical split patterns
  - Market reaction to corporate restructuring

#### **Statistical Analysis**

##### **📊 Daily % Change Distribution**
- **Histogram**: Distribution of daily percentage returns
- **Normal Curve Overlay**: Compare to theoretical normal distribution
- **Statistics Box**: Mean, standard deviation, skewness, kurtosis
- **📍 Key Insights**:
  - Risk assessment through return distribution
  - Market efficiency indicators
  - Outlier identification (extreme gain/loss days)

##### **🎯 Moving Average Crossover Signals**
- **Price Line**: Daily closing prices
- **50-Day MA**: Medium-term trend (blue line)
- **200-Day MA**: Long-term trend (red line)
- **Buy Signals**: Green triangles (50 MA crosses above 200 MA)
- **Sell Signals**: Red triangles (50 MA crosses below 200 MA)
- **Strategy Stats**: Number of signals and performance summary
- **📍 Key Insights**:
  - Golden Cross (bullish) and Death Cross (bearish) signals
  - Trend following strategy effectiveness
  - Entry and exit points for long-term investors

#### **Advanced Charts**

##### **📈 Price Trends Analysis (Multi-Panel)**
- **Panel 1**: Price with moving averages
- **Panel 2**: Trading volume with color coding
- **Panel 3**: Daily returns as bar chart
- **📍 Key Insights**:
  - Volume-price relationship
  - Return volatility patterns
  - Support and resistance levels

##### **🔥 Correlation Heatmap**
- **Interactive Heatmap**: OHLCV correlations
- **Color Scale**: Blue (negative) to Red (positive)
- **Correlation Values**: Displayed in each cell
- **📍 Key Insights**:
  - Relationship between different price points
  - Volume-price correlations
  - Market efficiency indicators

#### **🎯 How to Use EDA Tab**
1. **Start with Dividends**: Check corporate action impacts
2. **Analyze Returns**: Review distribution for risk assessment
3. **Study Signals**: Understand trend-following opportunities
4. **Examine Correlations**: Identify relationships between variables
5. **Look for Patterns**: Use all charts together for comprehensive view

---

### 🤖 Tab 3: Predictions - Machine Learning Models

**Purpose**: Train and evaluate machine learning models for stock price prediction.

#### **Model Selection & Training**

##### **🚀 Train Model Button**
- **Single Click Training**: Automated model training pipeline
- **Progress Indicators**: Real-time feedback during training
- **Error Handling**: Clear error messages if training fails

#### **Linear Regression Model**

##### **⚡ Fast & Interpretable**
- **Training Time**: ~2 seconds
- **Features Used**:
  - Previous day's OHLC values
  - 10-day and 30-day moving averages
  - Daily returns
  - Volume indicators

##### **📊 Performance Metrics Cards**
- **🎯 R² Score**: Coefficient of determination (0-1, higher better)
- **📉 MSE**: Mean Squared Error (lower better)
- **📊 MAE**: Mean Absolute Error in ₹ (lower better)

##### **📈 Prediction Visualization**
- **Blue Line**: Actual prices
- **Red Line**: Predicted prices
- **Green Section**: Training data
- **Orange Section**: Testing data
- **Interactive Hover**: See exact values and dates

#### **LSTM Neural Network**

##### **🧠 Advanced Deep Learning**
- **Training Time**: ~30 seconds (optimized)
- **Architecture**:
  - LSTM Layer: 25 neurons
  - Dropout: 10% (prevents overfitting)
  - Dense Layer: 10 neurons
  - Output: Single price prediction

##### **📊 Enhanced Performance**
- **Sequence Learning**: Uses 30-day price sequences
- **Pattern Recognition**: Captures complex temporal relationships
- **Better Accuracy**: Typically higher R² scores than linear models

#### **💾 Download Predictions**

##### **CSV Export Features**
- **Complete Dataset**: All predictions with dates
- **Data Split Labels**: Train/Test identification
- **Actual vs Predicted**: Side-by-side comparison
- **Timestamp Format**: Ready for further analysis

##### **📁 File Naming Convention**
```
TCS_Linear_Regression_Predictions_YYYYMMDD_HHMMSS.csv
TCS_LSTM_Predictions_YYYYMMDD_HHMMSS.csv
```

#### **🎯 How to Use Predictions Tab**
1. **Choose Your Model**: Linear for speed, LSTM for accuracy
2. **Click Train**: Wait for model training to complete
3. **Review Metrics**: Check R², MSE, MAE values
4. **Analyze Chart**: Look at prediction vs actual alignment
5. **Download Results**: Export for external analysis
6. **Compare Models**: Train both to see performance differences

#### **📊 Model Performance Interpretation**

##### **R² Score (Coefficient of Determination)**
- **0.90-1.00**: Excellent fit
- **0.80-0.89**: Good fit
- **0.70-0.79**: Acceptable fit
- **Below 0.70**: Poor fit, consider more data or features

##### **Mean Absolute Error (MAE)**
- **<₹20**: Very good for TCS stock
- **₹20-₹50**: Acceptable range
- **>₹50**: May need model improvement

---

### 💡 Tab 4: Insights - Market Intelligence

**Purpose**: Get actionable insights about market trends, risk assessment, and trading signals.

#### **📈 Price Analysis Panel**

##### **Market Trend Assessment**
- **📈 Bullish**: Current price > 50-day MA > 200-day MA
- **📉 Bearish**: Current price < 50-day MA < 200-day MA
- **📊 Neutral**: Mixed signals or sideways movement

##### **Moving Average Analysis**
- **Current Price vs 50-Day MA**: Short-term trend indicator
- **Current Price vs 200-Day MA**: Long-term trend indicator
- **MA Relationship**: Trend strength assessment

#### **📊 Risk Metrics Panel**

##### **Volatility Measures**
- **Annual Volatility**: Annualized standard deviation of returns
  - **<20%**: Low volatility
  - **20-30%**: Medium volatility
  - **>30%**: High volatility

##### **Risk Assessment**
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Risk Level**: Categorical risk assessment

#### **🎯 Trading Signals**

##### **Golden Cross / Death Cross**
- **✅ Golden Cross**: 50-day MA crosses above 200-day MA (Bullish)
- **❌ Death Cross**: 50-day MA crosses below 200-day MA (Bearish)

##### **Volume Analysis**
- **High Volume**: Current volume > 1.5x average (Strong interest)
- **Normal Volume**: Regular trading activity

##### **RSI Signals**
- **⚠️ RSI > 70**: Overbought condition (Consider taking profits)
- **💚 RSI < 30**: Oversold condition (Potential buying opportunity)
- **📊 RSI 30-70**: Normal trading range

#### **🏢 Company Information Panel**

##### **Fundamental Data** (If Available)
- **🏭 Sector**: Industry classification
- **🔧 Industry**: Specific business area
- **👥 Employees**: Total workforce
- **💰 Market Cap**: Market capitalization

#### **🎯 How to Use Insights Tab**
1. **Check Trend**: Start with price analysis for overall direction
2. **Assess Risk**: Review volatility and drawdown metrics
3. **Look for Signals**: Identify potential trading opportunities
4. **Consider Context**: Factor in company fundamentals
5. **Make Decisions**: Use insights for investment planning

---

## Data Upload Guide

### 📁 Supported File Formats

#### **CSV File Requirements**
- **Format**: Comma-separated values (.csv)
- **Encoding**: UTF-8 (recommended)
- **Size Limit**: 200MB maximum

#### **Required Columns**
Your CSV file must contain these exact column names:

| Column | Description | Example |
|--------|-------------|---------|
| `Date` | Trading date | 2024-01-15 |
| `Open` | Opening price | 3245.50 |
| `High` | Highest price | 3267.80 |
| `Low` | Lowest price | 3232.15 |
| `Close` | Closing price | 3255.25 |
| `Volume` | Trading volume | 2456789 |

#### **Optional Columns**
- `Dividends`: Dividend payments
- `Stock Splits`: Stock split ratios
- `Adj Close`: Adjusted closing price

### 📝 Data Format Examples

#### **Correct Format**
```csv
Date,Open,High,Low,Close,Volume
2024-01-15,3245.50,3267.80,3232.15,3255.25,2456789
2024-01-16,3255.25,3278.90,3241.10,3265.40,1987654
2024-01-17,3265.40,3289.75,3258.30,3275.85,2134567
```

#### **Date Format Options**
- `YYYY-MM-DD` (Recommended): 2024-01-15
- `MM/DD/YYYY`: 01/15/2024
- `DD/MM/YYYY`: 15/01/2024
- `YYYY/MM/DD`: 2024/01/15

### 🔄 Upload Process

#### **Step-by-Step Guide**
1. **Select Upload Option**: Choose "Upload New Dataset" in sidebar
2. **Click Browse**: Select your CSV file
3. **Wait for Processing**: File validation and cleaning
4. **Check Success**: Green message confirms successful upload
5. **Start Analysis**: All features now work with your data

#### **Validation Checks**
- ✅ **Column Names**: Verifies required columns exist
- ✅ **Data Types**: Ensures numeric columns are properly formatted
- ✅ **Date Parsing**: Converts date strings to datetime objects
- ✅ **Missing Values**: Identifies and handles gaps in data
- ✅ **Data Quality**: Checks for obvious errors or outliers

### ⚠️ Common Upload Issues

#### **Missing Columns Error**
```
Error: Required columns missing: ['Date', 'Close']
```
**Solution**: Ensure your CSV has all required column names (case-sensitive)

#### **Date Parsing Error**
```
Error: Unable to parse dates in 'Date' column
```
**Solution**: Check date format consistency throughout the file

#### **Invalid Data Types**
```
Error: Non-numeric values found in price columns
```
**Solution**: Remove text characters from numeric columns (commas, currency symbols)

### 🎯 Data Quality Tips

#### **Before Upload**
- **Remove Headers**: Ensure only one header row exists
- **Clean Data**: Remove any non-numeric characters from price/volume columns
- **Check Dates**: Ensure consistent date format throughout
- **Sort Data**: Arrange by date (oldest to newest)

#### **Data Sources**
- **Yahoo Finance**: Direct download compatible
- **Google Finance**: May need column renaming
- **Bloomberg**: Check date format
- **Custom Data**: Ensure proper formatting

---

## Analysis Techniques

### 📊 Technical Analysis Methods

#### **Moving Average Strategies**

##### **Golden Cross Strategy**
- **Signal**: 50-day MA crosses above 200-day MA
- **Interpretation**: Strong bullish signal for long-term uptrend
- **Best Use**: Long-term investment decisions
- **Caution**: Can generate false signals in sideways markets

##### **Death Cross Strategy**
- **Signal**: 50-day MA crosses below 200-day MA
- **Interpretation**: Strong bearish signal for long-term downtrend
- **Best Use**: Risk management and exit timing
- **Caution**: May be late signal in fast-moving markets

#### **Volume Analysis**

##### **Volume-Price Relationship**
- **High Volume + Rising Price**: Strong buying pressure
- **High Volume + Falling Price**: Strong selling pressure
- **Low Volume + Price Movement**: Weak signal, may reverse
- **Volume Spikes**: Often precede significant price movements

##### **Volume Indicators**
- **Above Average Volume**: 1.5x+ normal volume indicates strong interest
- **Volume Trends**: Increasing volume confirms price trends
- **Volume Divergence**: Price moves without volume may be unsustainable

#### **Risk Assessment Techniques**

##### **Volatility Analysis**
- **Low Volatility (<20%)**: Stable, conservative investment
- **Medium Volatility (20-30%)**: Moderate risk/reward
- **High Volatility (>30%)**: High risk/reward, requires active management

##### **Maximum Drawdown**
- **<15%**: Low drawdown risk
- **15-30%**: Moderate drawdown risk
- **>30%**: High drawdown risk, consider position sizing

### 🔍 Pattern Recognition

#### **Price Patterns**
- **Support Levels**: Price floors where buying typically emerges
- **Resistance Levels**: Price ceilings where selling typically emerges
- **Trend Lines**: Connect highs/lows to identify trend direction
- **Breakouts**: Price moves beyond established support/resistance

#### **Distribution Analysis**
- **Normal Distribution**: Typical market behavior
- **Fat Tails**: Indicates higher probability of extreme moves
- **Skewness**: Bias toward positive or negative returns
- **Kurtosis**: Measure of extreme event frequency

---

## Model Usage

### 🎯 When to Use Linear Regression

#### **Best For**
- **Quick Analysis**: Fast training and prediction
- **Trend Following**: Identifying linear relationships
- **Feature Importance**: Understanding which factors drive price
- **Baseline Model**: Starting point for analysis

#### **Limitations**
- **Linear Relationships Only**: Cannot capture complex patterns
- **Feature Engineering Required**: Needs well-designed features
- **Short-term Predictions**: Best for near-term forecasting

#### **Interpretation**
- **R² Score**: Percentage of price variance explained by features
- **Coefficients**: Impact of each feature on price prediction
- **Residuals**: Unexplained price movements

### 🧠 When to Use LSTM

#### **Best For**
- **Complex Patterns**: Capturing non-linear relationships
- **Sequence Learning**: Understanding temporal dependencies
- **Long-term Predictions**: Better for extended forecasts
- **High Accuracy**: When precision is more important than speed

#### **Limitations**
- **Black Box**: Difficult to interpret predictions
- **Training Time**: Requires more computational resources
- **Data Requirements**: Needs larger datasets for best performance
- **Overfitting Risk**: Can memorize patterns that don't generalize

#### **Architecture Understanding**
- **LSTM Neurons**: Remember long-term dependencies
- **Dropout**: Prevents overfitting by randomly ignoring neurons
- **Sequence Length**: How many days of history to consider
- **Dense Layer**: Final processing before prediction

### 📊 Model Comparison Guidelines

#### **Accuracy Metrics**
- **R² Score**: Higher is better (max 1.0)
- **MAE**: Lower is better (in ₹)
- **MSE**: Lower is better (squared errors)

#### **Performance Expectations**
- **Linear Regression**: R² typically 0.75-0.85
- **LSTM**: R² typically 0.80-0.90
- **Real-world Use**: Consider transaction costs and slippage

#### **Model Selection Criteria**
1. **Speed vs Accuracy**: Linear for speed, LSTM for accuracy
2. **Interpretability**: Linear if you need to understand predictions
3. **Data Size**: LSTM works better with more data
4. **Use Case**: Short-term (Linear) vs Long-term (LSTM) predictions

---

## Troubleshooting

### 🔧 Common Issues & Solutions

#### **Dashboard Won't Load**
```bash
# Error: No module named 'streamlit'
pip install streamlit

# Error: Port already in use
streamlit run dashboard/app.py --server.port 8502
```

#### **Data Loading Errors**
```python
# Error: File not found
# Solution: Check file paths in src/ directory
# Ensure data/ folder contains CSV files
```

#### **Memory Issues**
```python
# Error: MemoryError during LSTM training
# Solutions:
# 1. Reduce date range in sidebar
# 2. Use Linear Regression instead
# 3. Close other applications
```

#### **Model Training Failures**
```python
# Error: Insufficient data for model training
# Solution: Ensure at least 100 data points
# Select longer date range

# Error: NaN values in features
# Solution: Data will be automatically cleaned
# Check for extreme outliers in uploaded data
```

### 📊 Performance Optimization

#### **Faster Loading**
- **Reduce Date Range**: Smaller datasets load faster
- **Close Browser Tabs**: Free up memory
- **Use Default Data**: Pre-optimized for performance

#### **Better Predictions**
- **More Data**: Longer time periods improve model accuracy
- **Clean Data**: Remove outliers and inconsistencies
- **Appropriate Model**: Match model complexity to data size

### 🔍 Data Quality Issues

#### **Missing Values**
- **Automatic Handling**: Forward and backward fill applied
- **Manual Check**: Review data for excessive gaps
- **Impact**: Large gaps may affect model performance

#### **Outliers**
- **Automatic Removal**: Statistical outlier detection applied
- **Manual Review**: Check for data entry errors
- **Market Events**: Some outliers may be legitimate (news, earnings)

---

## Tips & Best Practices

### 🎯 Analysis Best Practices

#### **Data Selection**
- **Sufficient History**: Use at least 2 years of data for reliable analysis
- **Recent Data**: Include recent periods for current relevance
- **Market Conditions**: Consider different market cycles (bull/bear)

#### **Model Usage**
- **Start Simple**: Begin with Linear Regression for understanding
- **Validate Results**: Cross-check predictions with market knowledge
- **Multiple Models**: Compare different approaches for robustness
- **Regular Updates**: Retrain models with new data periodically

#### **Risk Management**
- **Diversification**: Don't rely on single stock analysis
- **Position Sizing**: Consider volatility for position allocation
- **Stop Losses**: Use technical signals for risk management
- **Market Context**: Consider broader market conditions

### 📈 Investment Applications

#### **Long-term Investing**
- **Focus on Trends**: Use moving average crossovers
- **Ignore Noise**: Look at longer time periods
- **Fundamental Alignment**: Combine with company analysis
- **Regular Review**: Monitor positions quarterly

#### **Active Trading**
- **Volume Confirmation**: Ensure trades have volume support
- **Risk-Reward**: Maintain favorable risk-reward ratios
- **Quick Decisions**: Use faster models for timely signals
- **Market Hours**: Consider timing of signals

#### **Portfolio Analysis**
- **Correlation**: Check how TCS relates to other holdings
- **Sector Exposure**: Consider IT sector concentration
- **Volatility Budget**: Allocate based on risk capacity
- **Rebalancing**: Use signals for portfolio adjustments

### 🔍 Advanced Features

#### **Export Capabilities**
- **CSV Downloads**: All predictions can be exported
- **External Analysis**: Use with Excel, R, Python
- **Backup Data**: Save analysis results for later review
- **Sharing**: Export for team collaboration

#### **Customization Options**
- **Date Ranges**: Focus on specific periods of interest
- **Chart Types**: Choose visualization style preference
- **Model Parameters**: Adjust sequence length for LSTM
- **Analysis Depth**: Select specific analysis types

### ⚠️ Important Disclaimers

#### **Educational Purpose**
- **Learning Tool**: Designed for educational use
- **Not Financial Advice**: Predictions are not investment recommendations
- **Market Risk**: All investments carry risk of loss
- **Professional Advice**: Consult qualified advisors for investment decisions

#### **Model Limitations**
- **Past Performance**: Historical patterns may not continue
- **Market Changes**: Models may not adapt to regime changes
- **Black Swan Events**: Unexpected events not captured in historical data
- **Transaction Costs**: Real trading involves costs not modeled

### 📚 Learning Resources

#### **Technical Analysis**
- Study moving average strategies
- Learn about volume analysis
- Understand risk metrics
- Practice pattern recognition

#### **Machine Learning**
- Linear regression fundamentals
- Time series analysis
- Neural network basics
- Model evaluation techniques

#### **Financial Markets**
- Stock market mechanics
- Corporate actions impact
- Risk management principles
- Portfolio theory basics

---

**🎓 Congratulations!** You now have comprehensive knowledge to use the TCS Stock Analysis Dashboard effectively. Start with the Overview tab, explore the EDA features, experiment with the prediction models, and use the insights for informed decision-making.

**📞 Need Help?** Refer to the API Reference for technical details or the Technical Specifications for system requirements.