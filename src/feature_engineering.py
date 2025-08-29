# TCS Stock Data Analysis & Prediction Project
# Feature Engineering Module

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

class TCSFeatureEngineer:
    """
    Advanced feature engineering class for TCS stock data
    """
    
    def __init__(self, df: pd.DataFrame, logger=None):
        self.df = df.copy()
        self.logger = logger or logging.getLogger(__name__)
        self.feature_names = []
        
    def create_technical_indicators(self) -> pd.DataFrame:
        """
        Create comprehensive technical indicators
        
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        df = self.df.copy()
        
        # 1. Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
            self.feature_names.extend([f'MA_{window}', f'MA_{window}_ratio'])
        
        # 2. Exponential Moving Averages
        for span in [12, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
            self.feature_names.append(f'EMA_{span}')
        
        # 3. Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['High_Close_Pct'] = (df['High'] - df['Close']) / df['Close'] * 100
        df['Low_Close_Pct'] = (df['Close'] - df['Low']) / df['Close'] * 100
        
        self.feature_names.extend(['High_Low_Pct', 'Open_Close_Pct', 'High_Close_Pct', 'Low_Close_Pct'])
        
        # 4. RSI (Relative Strength Index)
        df = self._calculate_rsi(df, window=14)
        df = self._calculate_rsi(df, window=30)
        
        # 5. MACD (Moving Average Convergence Divergence)
        df = self._calculate_macd(df)
        
        # 6. Bollinger Bands
        df = self._calculate_bollinger_bands(df, window=20, num_std=2)
        
        # 7. Stochastic Oscillator
        df = self._calculate_stochastic(df, k_window=14, d_window=3)
        
        # 8. Average True Range (ATR)
        df = self._calculate_atr(df, window=14)
        
        # 9. Williams %R
        df = self._calculate_williams_r(df, window=14)
        
        # 10. Price Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
            self.feature_names.append(f'ROC_{period}')
        
        self.logger.info(f"✅ Created {len(self.feature_names)} technical indicators")
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df[f'RSI_{window}'] = rsi
        self.feature_names.append(f'RSI_{window}')
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        self.feature_names.extend(['MACD', 'MACD_signal', 'MACD_histogram'])
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * num_std)
        df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * num_std)
        df[f'BB_width_{window}'] = df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']
        df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / df[f'BB_width_{window}']
        
        self.feature_names.extend([f'BB_upper_{window}', f'BB_lower_{window}', 
                                  f'BB_width_{window}', f'BB_position_{window}'])
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_window).min()
        highest_high = df['High'].rolling(window=k_window).max()
        
        df[f'Stoch_K_{k_window}'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        df[f'Stoch_D_{d_window}'] = df[f'Stoch_K_{k_window}'].rolling(window=d_window).mean()
        
        self.feature_names.extend([f'Stoch_K_{k_window}', f'Stoch_D_{d_window}'])
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df[f'ATR_{window}'] = true_range.rolling(window=window).mean()
        
        self.feature_names.append(f'ATR_{window}')
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        
        df[f'Williams_R_{window}'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        
        self.feature_names.append(f'Williams_R_{window}')
        return df
    
    def create_volume_features(self) -> pd.DataFrame:
        """
        Create volume-based features
        
        Returns:
            pd.DataFrame: Data with volume features
        """
        if 'Volume' not in self.df.columns:
            self.logger.warning("Volume column not found, skipping volume features")
            return self.df
        
        df = self.df.copy()
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']
            self.feature_names.extend([f'Volume_MA_{window}', f'Volume_ratio_{window}'])
        
        # Price-Volume features
        df['Price_Volume'] = df['Close'] * df['Volume']
        df['Volume_Price_Trend'] = df['Volume'] * df['Close'].pct_change()
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume Rate of Change
        for period in [5, 10]:
            df[f'Volume_ROC_{period}'] = df['Volume'].pct_change(periods=period) * 100
            self.feature_names.append(f'Volume_ROC_{period}')
        
        self.feature_names.extend(['Price_Volume', 'Volume_Price_Trend', 'OBV'])
        
        self.logger.info("✅ Created volume-based features")
        return df
    
    def create_lag_features(self, target_col: str = 'Close', lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            target_col (str): Column to create lags for
            lags (List[int]): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        df = self.df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            df[f'{target_col}_change_lag_{lag}'] = df[target_col].pct_change(lag) * 100
            self.feature_names.extend([f'{target_col}_lag_{lag}', f'{target_col}_change_lag_{lag}'])
        
        # Rolling statistics for lags
        for window in [3, 5, 10]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            
            self.feature_names.extend([
                f'{target_col}_rolling_mean_{window}', f'{target_col}_rolling_std_{window}',
                f'{target_col}_rolling_min_{window}', f'{target_col}_rolling_max_{window}'
            ])
        
        self.logger.info(f"✅ Created lag features for {target_col}")
        return df
    
    def create_time_features(self) -> pd.DataFrame:
        """
        Create time-based features
        
        Returns:
            pd.DataFrame: Data with time features
        """
        df = self.df.copy()
        
        # Basic time features
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # Cyclical features (sine/cosine encoding)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Market timing features
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsMonthStart'] = df.index.is_month_start.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        time_features = [
            'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'DayOfYear', 'WeekOfYear',
            'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
            'DayOfYear_sin', 'DayOfYear_cos',
            'IsMonday', 'IsFriday', 'IsMonthEnd', 'IsMonthStart', 'IsQuarterEnd'
        ]
        
        self.feature_names.extend(time_features)
        
        self.logger.info("✅ Created time-based features")
        return df
    
    def create_volatility_features(self) -> pd.DataFrame:
        """
        Create volatility-based features
        
        Returns:
            pd.DataFrame: Data with volatility features
        """
        df = self.df.copy()
        
        # Calculate returns if not present
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Rolling volatility
        for window in [5, 10, 20, 30, 60]:
            df[f'Volatility_{window}'] = df['Daily_Return'].rolling(window=window).std()
            df[f'Volatility_ratio_{window}'] = df[f'Volatility_{window}'] / df['Volatility_30']
            self.feature_names.extend([f'Volatility_{window}', f'Volatility_ratio_{window}'])
        
        # GARCH-like features
        df['Squared_Returns'] = df['Daily_Return'] ** 2
        for window in [5, 20]:
            df[f'EWMA_Vol_{window}'] = df['Squared_Returns'].ewm(span=window).mean().apply(np.sqrt)
            self.feature_names.append(f'EWMA_Vol_{window}')
        
        # Volatility regime features
        df['High_Vol_Regime'] = (df['Volatility_20'] > df['Volatility_20'].rolling(60).mean()).astype(int)
        df['Vol_Trend'] = df['Volatility_20'].diff()
        
        self.feature_names.extend(['Squared_Returns', 'High_Vol_Regime', 'Vol_Trend'])
        
        self.logger.info("✅ Created volatility features")
        return df
    
    def create_momentum_features(self) -> pd.DataFrame:
        """
        Create momentum-based features
        
        Returns:
            pd.DataFrame: Data with momentum features
        """
        df = self.df.copy()
        
        # Price momentum
        for period in [3, 5, 10, 20, 50]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            self.feature_names.append(f'Momentum_{period}')
        
        # Moving average convergence/divergence
        for fast, slow in [(5, 20), (10, 30), (20, 50)]:
            df[f'MA_Conv_{fast}_{slow}'] = df[f'MA_{fast}'] / df[f'MA_{slow}'] - 1
            self.feature_names.append(f'MA_Conv_{fast}_{slow}')
        
        # Price position within recent range
        for window in [10, 20, 50]:
            rolling_min = df['Close'].rolling(window=window).min()
            rolling_max = df['Close'].rolling(window=window).max()
            df[f'Price_Position_{window}'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min)
            self.feature_names.append(f'Price_Position_{window}')
        
        self.logger.info("✅ Created momentum features")
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Create all features in one go
        
        Returns:
            pd.DataFrame: Data with all engineered features
        """
        self.logger.info("🔧 Starting comprehensive feature engineering...")
        
        # Start with original data
        df = self.df.copy()
        
        # Create all feature types
        df = self.create_technical_indicators()
        df = self.create_volume_features()
        df = self.create_lag_features()
        df = self.create_time_features()
        df = self.create_volatility_features()
        df = self.create_momentum_features()
        
        # Store the enriched dataframe
        self.df = df
        
        self.logger.info(f"✅ Feature engineering completed! Created {len(self.feature_names)} features")
        self.logger.info(f"📊 Dataset shape: {df.shape}")
        
        return df
    
    def get_feature_importance_ready_data(self, target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for feature importance analysis
        
        Args:
            target_col (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        df = self.df.copy()
        
        # Create target (next day's price)
        target = df[target_col].shift(-1)
        
        # Remove non-numeric and target columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        # Remove rows with NaN values
        valid_mask = ~(df[feature_cols].isna().any(axis=1) | target.isna())
        
        X = df.loc[valid_mask, feature_cols]
        y = target[valid_mask]
        
        self.logger.info(f"✅ Prepared {X.shape[1]} features for ML models")
        return X, y
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of created features
        
        Returns:
            Dict: Feature summary
        """
        return {
            'total_features': len(self.feature_names),
            'feature_categories': {
                'technical_indicators': len([f for f in self.feature_names if any(indicator in f for indicator in ['MA_', 'EMA_', 'RSI_', 'MACD', 'BB_', 'Stoch_', 'ATR_', 'Williams_'])]),
                'volume_features': len([f for f in self.feature_names if 'Volume' in f or 'OBV' in f]),
                'lag_features': len([f for f in self.feature_names if 'lag' in f or 'rolling' in f]),
                'time_features': len([f for f in self.feature_names if any(time_feat in f for time_feat in ['Year', 'Month', 'Day', 'Quarter', 'sin', 'cos', 'Is'])]),
                'volatility_features': len([f for f in self.feature_names if 'Volatility' in f or 'Vol' in f or 'EWMA' in f]),
                'momentum_features': len([f for f in self.feature_names if 'Momentum' in f or 'Conv' in f or 'Position' in f])
            },
            'feature_names': self.feature_names
        }

# Convenience function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick feature engineering function
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    engineer = TCSFeatureEngineer(df)
    return engineer.engineer_all_features()