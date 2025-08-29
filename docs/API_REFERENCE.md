# 📚 TCS Stock Analysis - API Reference

> **Complete documentation for all modules, classes, and functions in the TCS Stock Analysis project**

## Table of Contents

- [Dashboard Module](#dashboard-module)
- [EDA Module](#eda-module) 
- [Preprocessing Module](#preprocessing-module)
- [Feature Engineering Module](#feature-engineering-module)
- [Model Training Module](#model-training-module)
- [Utility Functions](#utility-functions)

---

## Dashboard Module

### `app.py` - Main Streamlit Application

#### Core Functions

##### `load_data() -> Dict`
```python
def load_data() -> Dict:
    """
    Load TCS stock data from CSV files
    
    Returns:
        Dict: Dictionary containing 'history', 'info', and 'actions' DataFrames
        
    Raises:
        FileNotFoundError: If data files are not found
    """
```

**Purpose**: Loads historical stock data, company information, and corporate actions data.

**Returns**:
- `history`: Historical OHLCV data with datetime index
- `info`: Company fundamental information
- `actions`: Dividend and stock split data

##### `process_uploaded_file(uploaded_file) -> pd.DataFrame`
```python
def process_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Process user-uploaded CSV file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Processed stock data with datetime index
        
    Raises:
        ValueError: If required columns are missing
    """
```

**Purpose**: Validates and processes user-uploaded stock data files.

**Required Columns**: Date, Open, High, Low, Close, Volume

##### `create_summary_statistics_table(df: pd.DataFrame) -> pd.DataFrame`
```python
def create_summary_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics table for stock data
    
    Args:
        df (pd.DataFrame): Stock data with OHLCV columns
        
    Returns:
        pd.DataFrame: Summary statistics table
    """
```

**Purpose**: Creates comprehensive statistical summary of stock data.

**Output Metrics**:
- Count, Mean, Std, Min, 25%, 50%, 75%, Max for all numeric columns
- Current price, price change, volatility calculations

#### Machine Learning Functions

##### `prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`
```python
def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML models
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
```

**Features Created**:
- Previous day's OHLC values
- Moving averages (10, 30 day)
- Daily returns
- Technical indicators

##### `train_linear_regression_model(X: pd.DataFrame, y: pd.Series) -> Dict`
```python
def train_linear_regression_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Train Linear Regression model for stock price prediction
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (Close price)
        
    Returns:
        Dict: Model results including metrics, predictions, and model object
    """
```

**Model Configuration**:
- Algorithm: Linear Regression with Ridge regularization
- Features: OHLC data, moving averages, returns
- Train/Test Split: 80/20
- Metrics: R², MSE, MAE

##### `train_lstm_model(df: pd.DataFrame, sequence_length: int = 30) -> Dict`
```python
def train_lstm_model(df: pd.DataFrame, sequence_length: int = 30) -> Dict:
    """
    Train LSTM Neural Network for stock price prediction
    
    Args:
        df (pd.DataFrame): Stock data
        sequence_length (int): Length of input sequences
        
    Returns:
        Dict: Model results including metrics and predictions
    """
```

**Model Architecture**:
- LSTM Layer: 25 neurons
- Dropout: 0.1 (10%)
- Dense Layer: 10 neurons
- Output: Single price prediction
- Optimizer: Adam
- Loss: Mean Squared Error

#### Visualization Functions

##### `plot_model_predictions(results: Dict, model_name: str) -> go.Figure`
```python
def plot_model_predictions(results: Dict, model_name: str) -> go.Figure:
    """
    Create interactive plot of model predictions vs actual prices
    
    Args:
        results (Dict): Model training results
        model_name (str): Name of the model
        
    Returns:
        go.Figure: Plotly figure with predictions
    """
```

**Plot Features**:
- Actual vs predicted prices
- Training/testing data separation
- Performance metrics display
- Interactive hover information

---

## EDA Module

### `TCSDataAnalyzer` Class

#### Initialization
```python
class TCSDataAnalyzer:
    def __init__(self, df: pd.DataFrame, actions_df: pd.DataFrame = None, logger=None):
        """
        Initialize TCS Data Analyzer
        
        Args:
            df (pd.DataFrame): Stock price data with datetime index
            actions_df (pd.DataFrame): Corporate actions data (dividends, splits)
            logger: Optional logger instance
        """
```

#### Core Analysis Methods

##### `generate_summary_statistics() -> Dict`
```python
def generate_summary_statistics(self) -> Dict:
    """
    Generate comprehensive summary statistics
    
    Returns:
        Dict: Complete statistical summary including:
            - basic_info: Records count, date range, missing values
            - price_stats: Price metrics (current, high, low, avg, std)
            - return_stats: Daily return analysis and Sharpe ratio
            - volume_stats: Volume analysis (if available)
    """
```

##### `analyze_moving_average_crossover() -> Dict`
```python
def analyze_moving_average_crossover(self) -> Dict:
    """
    Analyze moving average crossover trading strategy
    
    Returns:
        Dict: Strategy analysis including:
            - total_buy_signals: Number of buy signals generated
            - total_sell_signals: Number of sell signals generated
            - total_trades: Complete trades executed
            - avg_return_per_trade: Average return percentage
            - win_rate: Percentage of profitable trades
            - best_trade: Highest return trade
            - worst_trade: Lowest return trade
    """
```

##### `analyze_volatility_patterns() -> Dict`
```python
def analyze_volatility_patterns(self) -> Dict:
    """
    Analyze volatility patterns in stock data
    
    Returns:
        Dict: Volatility analysis including:
            - current_volatility_30d: 30-day rolling volatility
            - current_volatility_90d: 90-day rolling volatility
            - avg_volatility: Average volatility
            - max_volatility_period: Date of highest volatility
            - volatility_trend: Current trend direction
    """
```

#### Visualization Methods

##### `plot_price_trends(show_ma: bool = True) -> go.Figure`
```python
def plot_price_trends(self, show_ma: bool = True) -> go.Figure:
    """
    Create comprehensive price trend analysis
    
    Args:
        show_ma (bool): Whether to include moving averages
        
    Returns:
        go.Figure: Multi-subplot figure with:
            - Price chart with moving averages
            - Volume chart
            - Daily returns chart
    """
```

##### `plot_correlation_heatmap() -> go.Figure`
```python
def plot_correlation_heatmap(self) -> go.Figure:
    """
    Create correlation heatmap for OHLCV data
    
    Returns:
        go.Figure: Interactive heatmap showing correlations between:
            - Open, High, Low, Close prices
            - Volume (if available)
            - Color scale: Blue (negative) to Red (positive)
    """
```

##### `plot_dividends_vs_close_price() -> go.Figure`
```python
def plot_dividends_vs_close_price(self) -> go.Figure:
    """
    Analyze dividend payments vs stock price
    
    Returns:
        go.Figure: Two-panel chart showing:
            - Top: Stock price with dividend event markers
            - Bottom: Dividend amounts over time
    """
```

##### `plot_stock_splits_vs_close_price() -> go.Figure`
```python
def plot_stock_splits_vs_close_price(self) -> go.Figure:
    """
    Analyze stock splits vs price movements
    
    Returns:
        go.Figure: Price chart with stock split event markers
    """
```

##### `plot_moving_average_crossover_signals() -> go.Figure`
```python
def plot_moving_average_crossover_signals(self) -> go.Figure:
    """
    Visualize moving average crossover trading signals
    
    Returns:
        go.Figure: Chart showing:
            - Stock price line
            - 50-day and 200-day moving averages
            - Buy signal markers (green triangles up)
            - Sell signal markers (red triangles down)
            - Strategy performance summary
    """
```

##### `plot_daily_change_distribution() -> go.Figure`
```python
def plot_daily_change_distribution(self) -> go.Figure:
    """
    Analyze distribution of daily price changes
    
    Returns:
        go.Figure: Histogram with:
            - Daily return percentage distribution
            - Normal distribution overlay
            - Statistical metrics (mean, std, skewness, kurtosis)
    """
```

##### `plot_candlestick_chart(days: int = 100) -> go.Figure`
```python
def plot_candlestick_chart(self, days: int = 100) -> go.Figure:
    """
    Create candlestick chart for recent trading data
    
    Args:
        days (int): Number of recent days to display
        
    Returns:
        go.Figure: Two-panel candlestick chart:
            - Top: OHLC candlestick chart
            - Bottom: Volume bars with color coding
    """
```

---

## Preprocessing Module

### `TCSDataPreprocessor` Class

#### Initialization
```python
class TCSDataPreprocessor:
    def __init__(self, logger=None):
        """
        Initialize TCS Data Preprocessor
        
        Args:
            logger: Optional logger instance for tracking operations
        """
```

#### Core Methods

##### `load_and_clean_data(file_path: str) -> pd.DataFrame`
```python
def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
    """
    Comprehensive data loading and cleaning pipeline
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Cleaned and validated stock data
        
    Processing Steps:
        1. Load CSV and handle date parsing
        2. Remove duplicate records
        3. Handle missing values (forward/backward fill)
        4. Validate OHLC data consistency
        5. Clean volume data (handle zeros/negatives)
        6. Process corporate actions data
        7. Remove statistical outliers
        8. Optimize data types
    """
```

##### `add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame`
```python
def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock data
    
    Args:
        df (pd.DataFrame): Clean stock data
        
    Returns:
        pd.DataFrame: Data with technical indicators
        
    Indicators Added:
        - Moving Averages: 5, 10, 20, 50, 200 day
        - Returns: Daily return percentage and price change
        - Price Ratios: Open/Close, High/Low percentages
        - Volume: 20-day volume MA and volume ratio
        - Volatility: 10 and 30-day rolling volatility
    """
```

##### `prepare_for_ml(df: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]`
```python
def prepare_for_ml(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning models
    
    Args:
        df (pd.DataFrame): Stock data with indicators
        target_column (str): Target variable column name
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
        
    Processing:
        - Remove non-predictive columns
        - Handle remaining missing values
        - Create lag features
        - Scale numerical features
    """
```

#### Utility Methods

##### `get_preprocessing_summary() -> Dict`
```python
def get_preprocessing_summary(self) -> Dict:
    """
    Get summary of preprocessing operations performed
    
    Returns:
        Dict: Summary including:
            - steps_performed: List of all preprocessing steps
            - final_shape: Final dataset dimensions
            - columns: List of column names in processed data
    """
```

---

## Feature Engineering Module

### `TCSFeatureEngineer` Class

#### Technical Indicators

##### Moving Averages
- Simple Moving Averages (SMA): 5, 10, 20, 50, 100, 200 days
- Exponential Moving Averages (EMA): 12, 26 days
- Moving Average Convergence Divergence (MACD)

##### Price-based Indicators
- Relative Strength Index (RSI): 14-day period
- Bollinger Bands: 20-day with 2 standard deviations
- Price Rate of Change (ROC): Multiple periods
- Average True Range (ATR): Volatility measure

##### Volume Indicators
- Volume Rate of Change
- On Balance Volume (OBV)
- Volume Moving Averages

---

## Model Training Module

### `TCSModelTrainer` Class

#### Supported Models

##### Linear Models
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1 + L2 regularization)

##### Tree-based Models
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (if available)

##### Neural Networks
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Dense Neural Networks

#### Model Evaluation

##### Metrics
- R-squared Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)

##### Cross-validation
- Time Series Split validation
- Walk-forward validation
- Expanding window validation

---

## Utility Functions

### Data Loading
```python
def load_stock_data(file_path: str) -> pd.DataFrame:
    """Quick data loading with basic validation"""

def validate_stock_data(df: pd.DataFrame) -> bool:
    """Validate stock data format and completeness"""
```

### Visualization Helpers
```python
def create_subplot_layout(rows: int, cols: int) -> go.Figure:
    """Create standardized subplot layouts"""

def apply_theme(fig: go.Figure, theme: str = 'plotly_white') -> go.Figure:
    """Apply consistent theming to plots"""
```

### Performance Utilities
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for return series"""

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown percentage"""
```

---

## Error Handling

### Common Exceptions

#### `DataValidationError`
Raised when stock data fails validation checks.

#### `InsufficientDataError`
Raised when insufficient data for analysis or modeling.

#### `ModelTrainingError`
Raised when machine learning model training fails.

---

## Usage Examples

### Basic EDA Analysis
```python
from src.eda import TCSDataAnalyzer

# Load data
df = pd.read_csv('data/TCS_stock_history.csv')
actions = pd.read_csv('data/TCS_stock_actions.csv')

# Initialize analyzer
analyzer = TCSDataAnalyzer(df, actions)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report()

# Create visualizations
price_fig = analyzer.plot_price_trends()
correlation_fig = analyzer.plot_correlation_heatmap()
```

### Model Training
```python
from dashboard.app import train_linear_regression_model, prepare_features_and_target

# Prepare data
X, y = prepare_features_and_target(df)

# Train model
results = train_linear_regression_model(X, y)

# Access metrics
print(f"R² Score: {results['metrics']['test']['r2']:.4f}")
print(f"MAE: {results['metrics']['test']['mae']:.2f}")
```

### Data Preprocessing
```python
from src.preprocess import TCSDataPreprocessor

# Initialize preprocessor
preprocessor = TCSDataPreprocessor()

# Clean and process data
clean_data = preprocessor.load_and_clean_data('raw_data.csv')

# Add technical indicators
enhanced_data = preprocessor.add_technical_indicators(clean_data)

# Get summary
summary = preprocessor.get_preprocessing_summary()
```

---

## Performance Considerations

### Memory Usage
- Large datasets are processed in chunks when possible
- DataFrames are optimized for memory usage with appropriate dtypes
- Unnecessary columns are dropped early in the pipeline

### Computation Speed
- Vectorized operations using NumPy and Pandas
- Efficient algorithms for technical indicator calculations
- Parallel processing for independent operations

### Scalability
- Modular design allows for easy extension
- Configuration-driven approach for parameters
- Support for different data sources and formats

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-08 | Initial release with core functionality |
| 1.1.0 | 2024-08 | Added LSTM models and enhanced visualizations |
| 1.2.0 | 2024-08 | Improved error handling and performance |

---

**📝 Note**: This API reference is automatically generated from code documentation. For the most up-to-date information, refer to the inline documentation in the source code.