"""
TCS Stock Data Analysis & Prediction Project
============================================
Comprehensive data preprocessing module with robust cleaning, validation, and technical indicator functions
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

class TCSDataPreprocessor:
    """
    Comprehensive data preprocessing class for TCS stock data
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.processed_data = None
        self.preprocessing_steps = []
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and perform initial cleaning of TCS stock data
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            self.logger.info(f"✅ Data loaded successfully: {df.shape}")
            
            # Store original shape
            original_shape = df.shape
            
            # 1. Handle date column
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])  # Remove invalid dates
                df = df.sort_values(date_col)
                df.set_index(date_col, inplace=True)
                self.preprocessing_steps.append(f"✅ Date column '{date_col}' processed and set as index")
            
            # 2. Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                self.preprocessing_steps.append(f"✅ Removed {duplicates_removed} duplicate rows")
            
            # 3. Handle missing values
            missing_before = df.isnull().sum().sum()
            if missing_before > 0:
                # Forward fill then backward fill for time series data
                df = df.fillna(method='ffill').fillna(method='bfill')
                missing_after = df.isnull().sum().sum()
                self.preprocessing_steps.append(f"✅ Missing values handled: {missing_before} → {missing_after}")
            
            # 4. Validate and fix OHLC data
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                df = self._validate_ohlc_data(df)
                self.preprocessing_steps.append("✅ OHLC data validated and corrected")
            
            # 5. Handle volume data
            if 'Volume' in df.columns:
                # Handle zero or negative volumes
                zero_volume_count = (df['Volume'] <= 0).sum()
                if zero_volume_count > 0:
                    df.loc[df['Volume'] <= 0, 'Volume'] = df['Volume'].replace(0, np.nan).interpolate()
                    self.preprocessing_steps.append(f"✅ Fixed {zero_volume_count} invalid volume records")
            
            # 6. Handle dividends and stock splits
            for col in ['Dividends', 'Stock Splits']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)  # Fill NaN with 0 for these columns
                    self.preprocessing_steps.append(f"✅ {col} column cleaned")
            
            # 7. Remove outliers (using IQR method for price columns)
            price_columns = ['Open', 'High', 'Low', 'Close']
            outliers_removed = 0
            for col in price_columns:
                if col in df.columns:
                    outliers_before = len(df)
                    df = self._remove_outliers_iqr(df, col)
                    outliers_removed += outliers_before - len(df)
            
            if outliers_removed > 0:
                self.preprocessing_steps.append(f"✅ Removed {outliers_removed} outlier records")
            
            # 8. Data type optimization
            df = self._optimize_dtypes(df)
            self.preprocessing_steps.append("✅ Data types optimized")
            
            # Store processed data
            self.processed_data = df
            
            self.logger.info(f"🧹 Data preprocessing completed: {original_shape} → {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error in data preprocessing: {str(e)}")
            raise
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLC data inconsistencies
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Validated dataframe
        """
        # Ensure High is the maximum of OHLC
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        
        # Ensure Low is the minimum of OHLC
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Check for impossible values (negative prices)
        for col in ['Open', 'High', 'Low', 'Close']:
            if (df[col] <= 0).any():
                # Replace with interpolated values
                df[col] = df[col].replace(0, np.nan).interpolate()
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to check for outliers
            factor (float): IQR factor for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Remove outliers
        outlier_mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        return df[outlier_mask]
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for memory efficiency
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Optimized dataframe
        """
        # Convert float64 to float32 for price columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert int64 to int32 for volume
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Price change indicators
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Volatility
        df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
        df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
        
        self.preprocessing_steps.append("✅ Technical indicators added")
        return df
    
    def prepare_for_ml(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning models
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Create feature dataset
        features_df = df.copy()
        
        # Add lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'{target_column}_lag_{lag}'] = features_df[target_column].shift(lag)
        
        # Add date-based features
        features_df['Year'] = features_df.index.year
        features_df['Month'] = features_df.index.month
        features_df['Day'] = features_df.index.day
        features_df['DayOfWeek'] = features_df.index.dayofweek
        features_df['Quarter'] = features_df.index.quarter
        
        # Create target (next day's closing price)
        target = features_df[target_column].shift(-1)
        
        # Remove rows with NaN values
        valid_mask = ~(features_df.isna().any(axis=1) | target.isna())
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        # Select only numeric columns for ML
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        self.preprocessing_steps.append("✅ Data prepared for machine learning")
        return features_df, target
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps performed
        
        Returns:
            Dict: Summary of preprocessing
        """
        return {
            'steps_performed': self.preprocessing_steps,
            'final_shape': self.processed_data.shape if self.processed_data is not None else None,
            'columns': list(self.processed_data.columns) if self.processed_data is not None else None
        }

# Convenience function for quick preprocessing
def preprocess_tcs_data(file_path: str, add_indicators: bool = True) -> pd.DataFrame:
    """
    Quick preprocessing function for TCS data
    
    Args:
        file_path (str): Path to CSV file
        add_indicators (bool): Whether to add technical indicators
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    preprocessor = TCSDataPreprocessor()
    df = preprocessor.load_and_clean_data(file_path)
    
    if add_indicators:
        df = preprocessor.add_technical_indicators(df)
    
    return df