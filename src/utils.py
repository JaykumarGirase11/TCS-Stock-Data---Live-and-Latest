# TCS Stock Data Analysis & Prediction Project
# Core utility functions and configurations

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

warnings.filterwarnings('ignore')

# Project Configuration
PROJECT_CONFIG = {
    'DATA_PATH': 'data/',
    'MODELS_PATH': 'models/',
    'DASHBOARD_PATH': 'dashboard/',
    'RESULTS_PATH': 'results/',
    'ASSETS_PATH': 'assets/',
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'VALIDATION_SIZE': 0.1
}

# Logging Configuration
def setup_logging():
    """Setup logging configuration for the project"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tcs_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Data Loading Utilities
def load_tcs_data(file_path: str = None) -> pd.DataFrame:
    """
    Load TCS stock data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and preprocessed data
    """
    if file_path is None:
        file_path = os.path.join(PROJECT_CONFIG['DATA_PATH'], 'TCS_stock_history.csv')
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert Date column to datetime
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.sort_values(date_cols[0])
            df.set_index(date_cols[0], inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def validate_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLC data
    
    Args:
        df (pd.DataFrame): Stock data with OHLC columns
        
    Returns:
        pd.DataFrame: Validated data
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Fix OHLC inconsistencies
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    return df

def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate various return metrics
    
    Args:
        df (pd.DataFrame): Stock data
        price_col (str): Price column name
        
    Returns:
        pd.DataFrame: Data with return calculations
    """
    df = df.copy()
    
    # Daily returns
    df['Daily_Return'] = df[price_col].pct_change() * 100
    
    # Weekly returns
    df['Weekly_Return'] = df[price_col].pct_change(5) * 100
    
    # Monthly returns  
    df['Monthly_Return'] = df[price_col].pct_change(21) * 100
    
    # Cumulative returns
    df['Cumulative_Return'] = ((df[price_col] / df[price_col].iloc[0]) - 1) * 100
    
    return df

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        Dict: Summary statistics
    """
    if 'Close' not in df.columns:
        raise ValueError("Close price column not found")
    
    close_price = df['Close']
    daily_returns = df.get('Daily_Return', close_price.pct_change() * 100)
    
    stats = {
        'basic_stats': {
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'total_days': len(df),
            'current_price': close_price.iloc[-1],
            'min_price': close_price.min(),
            'max_price': close_price.max(),
            'avg_price': close_price.mean(),
        },
        'return_metrics': {
            'total_return': ((close_price.iloc[-1] / close_price.iloc[0]) - 1) * 100,
            'annualized_return': (1 + daily_returns.mean()/100) ** 252 - 1,
            'annualized_volatility': daily_returns.std() * np.sqrt(252),
            'max_daily_gain': daily_returns.max(),
            'max_daily_loss': daily_returns.min(),
        },
        'risk_metrics': {
            'sharpe_ratio': 0,  # Will calculate with risk-free rate
            'max_drawdown': 0,  # Will calculate separately
            'var_95': np.percentile(daily_returns.dropna(), 5),
            'var_99': np.percentile(daily_returns.dropna(), 1),
        }
    }
    
    # Calculate Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_return = stats['return_metrics']['annualized_return'] - risk_free_rate
    if stats['return_metrics']['annualized_volatility'] > 0:
        stats['risk_metrics']['sharpe_ratio'] = excess_return / (stats['return_metrics']['annualized_volatility'] / 100)
    
    # Calculate maximum drawdown
    cumulative = (1 + daily_returns/100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    stats['risk_metrics']['max_drawdown'] = drawdown.min()
    
    return stats

def create_directories():
    """Create necessary project directories"""
    directories = [
        PROJECT_CONFIG['MODELS_PATH'],
        PROJECT_CONFIG['RESULTS_PATH'],
        PROJECT_CONFIG['ASSETS_PATH'],
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Model Evaluation Utilities
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression performance metrics
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict: Performance metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# Initialize project
if __name__ == "__main__":
    create_directories()
    logger = setup_logging()
    logger.info("TCS Stock Analysis Project Initialized")