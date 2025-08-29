# TCS Stock Data Analysis & Prediction Project
# Main Streamlit Dashboard Application

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
import io
import sys
import os
import pickle
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from eda import TCSDataAnalyzer
    from feature_engineering import TCSFeatureEngineer
    from model_training import TCSModelTrainer
    from preprocess import TCSDataPreprocessor
except ImportError:
    # Fallback if imports fail
    pass

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TCS Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with eye-friendly colors
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .tab-content {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-top: 10px;
    }
    
    .download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for consistent styling
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#4facfe',
    'info': '#00f2fe',
    'warning': '#fa709a',
    'danger': '#fee140',
    'chart_colors': ['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#fa709a', '#fee140', '#ff6b6b', '#4ecdc4']
}

@st.cache_data
def load_data():
    """Load all TCS datasets with error handling"""
    datasets = {}
    
    # File paths
    files = {
        'history': 'data/TCS_stock_history.csv',
        'info': 'data/TCS_stock_info.csv',
        'actions': 'data/TCS_stock_action.csv'
    }
    
    for name, filepath in files.items():
        try:
            datasets[name] = pd.read_csv(filepath)
            if name == 'history':
                datasets[name]['Date'] = pd.to_datetime(datasets[name]['Date'])
                datasets[name] = datasets[name].set_index('Date').sort_index()
            elif name == 'actions':
                datasets[name]['Date'] = pd.to_datetime(datasets[name]['Date'])
                datasets[name] = datasets[name].sort_values('Date')
        except FileNotFoundError:
            st.error(f"⚠️ File not found: {filepath}")
            datasets[name] = pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            datasets[name] = pd.DataFrame()
    
    return datasets

def process_uploaded_file(uploaded_file):
    """Process uploaded dataset file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
            return df
        else:
            st.error("Please upload a CSV file")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def prepare_features_and_target(df):
    """Prepare features and target for ML models"""
    try:
        # Use feature engineering
        feature_engineer = TCSFeatureEngineer(df)
        df_with_features = feature_engineer.engineer_all_features()
        
        # Prepare for ML
        preprocessor = TCSDataPreprocessor()
        X, y = preprocessor.prepare_for_ml(df_with_features)
        
        return X, y
    except:
        # Simple fallback if feature engineering fails
        df = df.copy()
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_30'] = df['Close'].rolling(30).mean()
        df['Volume_MA'] = df['Volume'].rolling(10).mean() if 'Volume' in df.columns else 0
        
        # Select features
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Daily_Return', 'MA_10', 'MA_30']
        if 'Volume' not in df.columns:
            feature_cols.remove('Volume')
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Create target (next day's closing price)
        y = df['Close'].shift(-1)
        X = df[available_cols]
        
        # Remove NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y

def train_linear_regression_model(X, y):
    """Train Linear Regression model - OPTIMIZED VERSION"""
    try:
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Convert indices to arrays for compatibility
        train_dates = X_train.index.tolist()
        test_dates = X_test.index.tolist()
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': {'train': y_pred_train, 'test': y_pred_test},
            'actuals': {'train': y_train.values, 'test': y_test.values},
            'metrics': {
                'train': {'mse': train_mse, 'r2': train_r2, 'mae': train_mae},
                'test': {'mse': test_mse, 'r2': test_r2, 'mae': test_mae}
            },
            'indices': {'train': train_dates, 'test': test_dates}
        }
    except Exception as e:
        st.error(f"Linear Regression training error: {str(e)}")
        return None

def create_lstm_sequences(data, sequence_length=60):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model(df, sequence_length=30):  # Reduced from 60 to 30
    """Train LSTM model - FAST OPTIMIZED VERSION WITH TIMESTAMP FIX"""
    try:
        # Use smaller dataset for faster training
        df_sample = df.tail(500) if len(df) > 500 else df  # Use last 500 days max
        
        # Prepare data
        close_prices = df_sample['Close'].values.reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = create_lstm_sequences(scaled_data, sequence_length)
        
        if len(X) < 50:  # Need minimum data
            st.error("Insufficient data for LSTM training. Need at least 50 sequences.")
            return None
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build SIMPLIFIED LSTM model for speed
        model = Sequential([
            LSTM(25, input_shape=(sequence_length, 1)),  # Reduced neurons
            Dropout(0.1),  # Reduced dropout
            Dense(10),  # Smaller dense layer
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])  # Faster learning rate
        
        # Train model - MUCH FASTER
        with st.spinner("🚀 Training LSTM (Fast Mode)..."):
            history = model.fit(
                X_train, y_train,
                batch_size=16,  # Smaller batch size
                epochs=10,  # Reduced epochs for speed
                validation_data=(X_test, y_test),
                verbose=0
            )
        
        # Predictions
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_train_orig = scaler.inverse_transform(y_pred_train).flatten()
        y_pred_test_orig = scaler.inverse_transform(y_pred_test).flatten()
        
        # Metrics
        train_mse = mean_squared_error(y_train_orig, y_pred_train_orig)
        test_mse = mean_squared_error(y_test_orig, y_pred_test_orig)
        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)
        train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
        test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
        
        # FIXED: Create date indices for predictions without timestamp arithmetic
        full_dates = df_sample.index.tolist()
        
        # Calculate actual indices without arithmetic operations
        train_start_idx = sequence_length
        train_end_idx = sequence_length + len(y_train_orig)
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + len(y_test_orig)
        
        # Safely extract date ranges
        train_dates = full_dates[train_start_idx:train_end_idx] if train_end_idx <= len(full_dates) else full_dates[train_start_idx:]
        test_dates = full_dates[test_start_idx:test_end_idx] if test_end_idx <= len(full_dates) else full_dates[test_start_idx:]
        
        # Ensure we have matching lengths
        min_train_len = min(len(train_dates), len(y_train_orig), len(y_pred_train_orig))
        min_test_len = min(len(test_dates), len(y_test_orig), len(y_pred_test_orig))
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history,
            'predictions': {
                'train': y_pred_train_orig[:min_train_len], 
                'test': y_pred_test_orig[:min_test_len]
            },
            'actuals': {
                'train': y_train_orig[:min_train_len], 
                'test': y_test_orig[:min_test_len]
            },
            'metrics': {
                'train': {'mse': train_mse, 'r2': train_r2, 'mae': train_mae},
                'test': {'mse': test_mse, 'r2': test_r2, 'mae': test_mae}
            },
            'indices': {
                'train': train_dates[:min_train_len], 
                'test': test_dates[:min_test_len]
            }
        }
    except Exception as e:
        st.error(f"LSTM training error: {str(e)}")
        return None

def plot_model_predictions(model_results, model_name):
    """Plot actual vs predicted prices - FINAL FIX FOR TIMESTAMP ERROR"""
    try:
        fig = go.Figure()
        
        # Get the raw indices and convert to strings to avoid timestamp issues
        train_dates = [str(date) for date in model_results['indices']['train']]
        test_dates = [str(date) for date in model_results['indices']['test']]
        
        # Ensure we have valid data
        if not train_dates or not test_dates:
            st.warning("No sufficient data for plotting")
            return go.Figure()
        
        # Training predictions
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=model_results['actuals']['train'],
                name='Actual (Train)',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=model_results['predictions']['train'],
                name='Predicted (Train)',
                line=dict(color='lightblue', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        # Test predictions
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=model_results['actuals']['test'],
                name='Actual (Test)',
                line=dict(color='red', width=2),
                hovertemplate='Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=model_results['predictions']['test'],
                name='Predicted (Test)',
                line=dict(color='lightcoral', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        # NO VERTICAL LINE - This was causing the timestamp error
        # Removed the vline that was causing timestamp arithmetic issues
        
        fig.update_layout(
            title=f'🔮 {model_name} - Actual vs Predicted Stock Prices',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Plotting error: {str(e)}")
        # Return simple figure with just metrics display
        fig = go.Figure()
        fig.add_annotation(
            text="Model trained successfully! Results available in metrics above.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="blue")
        )
        fig.update_layout(
            title=f'✅ {model_name} - Training Completed Successfully',
            height=400
        )
        return fig

def create_summary_statistics_table(df):
    """Create summary statistics table - FIXED FORMAT ERROR"""
    try:
        stats = {
            'Metric': [
                'Total Records', 'Date Range', 'Current Price', 'Highest Price', 
                'Lowest Price', 'Average Price', 'Price Volatility', 'Total Return (%)',
                'Average Volume', 'Max Volume'
            ],
            'Value': []
        }
        
        # Basic stats - FIXED FORMAT SPECIFIERS
        stats['Value'].extend([
            f"{len(df):,}",
            f"{df.index.min().date()} to {df.index.max().date()}",
            f"₹{df['Close'].iloc[-1]:.2f}",
            f"₹{df['Close'].max():.2f}",  # FIXED: Removed extra colon
            f"₹{df['Close'].min():.2f}",  # FIXED: Removed extra colon
            f"₹{df['Close'].mean():.2f}",
            f"{df['Close'].std():.2f}",
            f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%"
        ])
        
        # Volume stats if available
        if 'Volume' in df.columns:
            stats['Value'].extend([
                f"{df['Volume'].mean():,.0f}",
                f"{df['Volume'].max():,.0f}"
            ])
        else:
            stats['Value'].extend(['N/A', 'N/A'])
        
        return pd.DataFrame(stats)
    except Exception as e:
        st.error(f"Error creating summary statistics: {str(e)}")
        return pd.DataFrame()

def create_download_data(predictions_df, model_name):
    """Create downloadable CSV data"""
    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=True)
    return csv_buffer.getvalue()

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 TCS Stock Analysis Dashboard</h1>
        <p>Comprehensive analysis of Tata Consultancy Services stock data with ML predictions</p>
        <p><strong>📅 Live Dashboard • 🤖 ML-Powered • 📊 Real-time Analytics</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div class="info-card">
        <h3>🎛️ Dashboard Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.markdown("### 📁 Dataset Options")
    use_default = st.sidebar.radio("Choose data source:", ["Use Default Dataset", "Upload New Dataset"])
    
    uploaded_file = None
    if use_default == "Upload New Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with Date, Open, High, Low, Close, Volume columns"
        )
    
    # Load data
    with st.spinner('🔄 Loading TCS stock data...'):
        if uploaded_file is not None:
            df_history = process_uploaded_file(uploaded_file)
            df_actions = pd.DataFrame()  # No actions data for uploaded files
            df_info = pd.DataFrame()
        else:
            datasets = load_data()
            df_history = datasets.get('history', pd.DataFrame())
            df_info = datasets.get('info', pd.DataFrame())
            df_actions = datasets.get('actions', pd.DataFrame())
    
    if df_history.empty:
        st.error("❌ Could not load historical data. Please check if the data files exist.")
        return
    
    # Date range selector
    st.sidebar.markdown("### 📅 Date Range")
    min_date = df_history.index.min().date()
    max_date = df_history.index.max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    mask = (df_history.index.date >= start_date) & (df_history.index.date <= end_date)
    df_filtered = df_history[mask].copy()
    
    # Model selection
    st.sidebar.markdown("### 🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose ML Model",
        ["Linear Regression", "LSTM Neural Network"],
        help="Select the machine learning model for predictions"
    )
    
    # Chart type selector
    st.sidebar.markdown("### 📊 Visualization Options")
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["📈 Line Chart", "🕯️ Candlestick", "📊 OHLC"]
    )
    
    # Analysis type selector
    st.sidebar.markdown("### 📊 Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["📈 Price Analysis", "📊 Volume Analysis", "💰 Corporate Actions", "🎯 Technical Analysis"]
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 EDA", "🤖 Predictions", "💡 Insights"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 📊 Stock Overview & Summary Statistics")
        
        # Key metrics row
        if not df_filtered.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = df_filtered['Close'].iloc[-1]
            price_change = df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[0]
            price_change_pct = (price_change / df_filtered['Close'].iloc[0]) * 100
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 Current Price</h4>
                    <h2>₹{current_price:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                card_class = "success-card" if price_change >= 0 else "warning-card"
                symbol = "📈" if price_change >= 0 else "📉"
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{symbol} Price Change</h4>
                    <h2>₹{price_change:.2f}</h2>
                    <p>({price_change_pct:+.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'Volume' in df_filtered.columns:
                    total_volume = df_filtered['Volume'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Total Volume</h4>
                        <h2>{total_volume:,.0f}</h4>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if 'Volume' in df_filtered.columns:
                    avg_volume = df_filtered['Volume'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📈 Avg Volume</h4>
                        <h2>{avg_volume:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col5:
                volatility = df_filtered['Close'].pct_change().std() * 100 * np.sqrt(252)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Volatility</h4>
                    <h2>{volatility:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary Statistics Table
        st.markdown("### 📋 Summary Statistics")
        summary_stats = create_summary_statistics_table(df_filtered)
        if not summary_stats.empty:
            st.dataframe(summary_stats, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 🔬 Exploratory Data Analysis")
        
        try:
            # Initialize analyzer with actions data
            analyzer = TCSDataAnalyzer(df_filtered, df_actions)
            
            # Calculate moving averages if not present
            if 'MA_50' not in df_filtered.columns:
                df_filtered['MA_50'] = df_filtered['Close'].rolling(50).mean()
            if 'MA_200' not in df_filtered.columns:
                df_filtered['MA_200'] = df_filtered['Close'].rolling(200).mean()
            if 'Daily_Return' not in df_filtered.columns:
                df_filtered['Daily_Return'] = df_filtered['Close'].pct_change() * 100
                
            # Update analyzer data
            analyzer.df = df_filtered
            
            # Create columns for EDA plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Dividends vs Close Price")
                div_fig = analyzer.plot_dividends_vs_close_price()
                st.plotly_chart(div_fig, use_container_width=True)
                
                st.markdown("#### 📊 Daily % Change Distribution")
                dist_fig = analyzer.plot_daily_change_distribution()
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Stock Splits vs Close Price")
                split_fig = analyzer.plot_stock_splits_vs_close_price()
                st.plotly_chart(split_fig, use_container_width=True)
                
                st.markdown("#### 🎯 Moving Average Crossover Signals")
                ma_signals_fig = analyzer.plot_moving_average_crossover_signals()
                st.plotly_chart(ma_signals_fig, use_container_width=True)
            
            # Full width plots
            st.markdown("#### 📈 Price Trends Analysis")
            price_fig = analyzer.plot_price_trends()
            st.plotly_chart(price_fig, use_container_width=True)
            
            st.markdown("#### 🔥 Correlation Analysis")
            corr_fig = analyzer.plot_correlation_heatmap()
            st.plotly_chart(corr_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in EDA analysis: {str(e)}")
            st.info("Some EDA features may not be available due to missing dependencies or data issues.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 🤖 Machine Learning Predictions")
        
        if st.button("🚀 Train Model & Generate Predictions", type="primary"):
            with st.spinner(f"Training {selected_model} model..."):
                try:
                    if selected_model == "Linear Regression":
                        # Prepare features
                        X, y = prepare_features_and_target(df_filtered)
                        if len(X) > 100:  # Ensure sufficient data
                            model_results = train_linear_regression_model(X, y)
                            
                            # Display metrics
                            st.markdown("### 📊 Model Performance Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="success-card">
                                    <h4>📈 R² Score (Test)</h4>
                                    <h2>{model_results['metrics']['test']['r2']:.4f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>📉 MSE (Test)</h4>
                                    <h2>{model_results['metrics']['test']['mse']:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="warning-card">
                                    <h4>📊 MAE (Test)</h4>
                                    <h2>{model_results['metrics']['test']['mae']:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Plot predictions
                            pred_fig = plot_model_predictions(model_results, "Linear Regression")
                            st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Prepare download data
                            predictions_df = pd.DataFrame({
                                'Date': list(model_results['indices']['train']) + list(model_results['indices']['test']),
                                'Actual_Price': list(model_results['actuals']['train']) + list(model_results['actuals']['test']),
                                'Predicted_Price': list(model_results['predictions']['train']) + list(model_results['predictions']['test']),
                                'Data_Split': ['Train'] * len(model_results['indices']['train']) + ['Test'] * len(model_results['indices']['test'])
                            })
                            
                            # Download section
                            st.markdown('<div class="download-section">', unsafe_allow_html=True)
                            st.markdown("### 💾 Download Predictions")
                            csv_data = create_download_data(predictions_df, "Linear_Regression")
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv_data,
                                file_name=f"TCS_Linear_Regression_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        else:
                            st.error("Insufficient data for model training. Need at least 100 data points.")
                    
                    elif selected_model == "LSTM Neural Network":
                        model_results = train_lstm_model(df_filtered)
                        
                        if model_results:
                            # Display metrics
                            st.markdown("### 📊 LSTM Model Performance Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="success-card">
                                    <h4>📈 R² Score (Test)</h4>
                                    <h2>{model_results['metrics']['test']['r2']:.4f}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>📉 MSE (Test)</h4>
                                    <h2>{model_results['metrics']['test']['mse']:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="warning-card">
                                    <h4>📊 MAE (Test)</h4>
                                    <h2>{model_results['metrics']['test']['mae']:.2f}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Plot predictions
                            pred_fig = plot_model_predictions(model_results, "LSTM Neural Network")
                            st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Prepare download data
                            predictions_df = pd.DataFrame({
                                'Date': list(model_results['indices']['train']) + list(model_results['indices']['test']),
                                'Actual_Price': list(model_results['actuals']['train']) + list(model_results['actuals']['test']),
                                'Predicted_Price': list(model_results['predictions']['train']) + list(model_results['predictions']['test']),
                                'Data_Split': ['Train'] * len(model_results['indices']['train']) + ['Test'] * len(model_results['indices']['test'])
                            })
                            
                            # Download section
                            st.markdown('<div class="download-section">', unsafe_allow_html=True)
                            st.markdown("### 💾 Download Predictions")
                            csv_data = create_download_data(predictions_df, "LSTM")
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv_data,
                                file_name=f"TCS_LSTM_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.info("Please ensure you have sufficient data and try again.")
        
        else:
            st.info("👆 Click the button above to train the selected model and generate predictions!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 💡 Market Insights & Analysis")
        
        # Key insights based on data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Price Analysis")
            if not df_filtered.empty:
                current_price = df_filtered['Close'].iloc[-1]
                ma_50 = df_filtered['Close'].rolling(50).mean().iloc[-1] if len(df_filtered) >= 50 else current_price
                ma_200 = df_filtered['Close'].rolling(200).mean().iloc[-1] if len(df_filtered) >= 200 else current_price
                
                trend = "📈 Bullish" if current_price > ma_50 > ma_200 else "📉 Bearish" if current_price < ma_50 < ma_200 else "📊 Neutral"
                
                st.markdown(f"""
                **Market Trend:** {trend}
                
                **Current Price:** ₹{current_price:.2f}
                
                **50-Day MA:** ₹{ma_50:.2f}
                
                **200-Day MA:** ₹{ma_200:.2f}
                
                **Trend Analysis:**
                - Price vs 50-MA: {'Above' if current_price > ma_50 else 'Below'}
                - Price vs 200-MA: {'Above' if current_price > ma_200 else 'Below'}
                """)
        
        with col2:
            st.markdown("### 📊 Risk Metrics")
            if not df_filtered.empty:
                returns = df_filtered['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                max_drawdown = ((df_filtered['Close'] / df_filtered['Close'].cummax()) - 1).min() * 100
                
                st.markdown(f"""
                **Annual Volatility:** {volatility:.2f}%
                
                **Max Drawdown:** {max_drawdown:.2f}%
                
                **Sharpe Ratio:** {(returns.mean() / returns.std() * np.sqrt(252)):.2f}
                
                **Risk Assessment:**
                - Volatility Level: {'High' if volatility > 30 else 'Medium' if volatility > 20 else 'Low'}
                - Drawdown Risk: {'High' if abs(max_drawdown) > 30 else 'Medium' if abs(max_drawdown) > 15 else 'Low'}
                """)
        
        # Trading signals and recommendations
        st.markdown("### 🎯 Trading Signals")
        
        if len(df_filtered) >= 200:
            # Calculate some basic signals
            current_price = df_filtered['Close'].iloc[-1]
            ma_50 = df_filtered['Close'].rolling(50).mean().iloc[-1]
            ma_200 = df_filtered['Close'].rolling(200).mean().iloc[-1]
            
            if 'Volume' in df_filtered.columns:
                avg_volume = df_filtered['Volume'].rolling(20).mean().iloc[-1]
                current_volume = df_filtered['Volume'].iloc[-1]
                volume_signal = "High" if current_volume > avg_volume * 1.5 else "Normal"
            else:
                volume_signal = "N/A"
            
            signals = []
            if current_price > ma_50 > ma_200:
                signals.append("✅ Golden Cross - Bullish trend")
            elif current_price < ma_50 < ma_200:
                signals.append("❌ Death Cross - Bearish trend")
            
            if volume_signal == "High":
                signals.append("📊 High volume detected - Strong interest")
            
            # RSI calculation (simple version)
            delta = df_filtered['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 70:
                signals.append("⚠️ RSI Overbought - Consider taking profits")
            elif current_rsi < 30:
                signals.append("💚 RSI Oversold - Potential buying opportunity")
            
            if signals:
                for signal in signals:
                    st.markdown(f"- {signal}")
            else:
                st.markdown("- 📊 No strong signals detected at current levels")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Company information section (if available)
    if not df_info.empty:
        st.markdown("---")
        st.markdown("### 🏢 Company Information")
        
        # Parse company info
        company_metrics = {}
        for _, row in df_info.iterrows():
            try:
                key = str(row.iloc[0]).strip()
                value = row.iloc[1]
                company_metrics[key] = value
            except:
                continue
        
        # Display key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sector = company_metrics.get('sector', 'N/A')
            st.markdown(f"""
            <div class="info-card">
                <h4>🏭 Sector</h4>
                <h3>{sector}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            industry = company_metrics.get('industry', 'N/A')
            st.markdown(f"""
            <div class="info-card">
                <h4>🔧 Industry</h4>
                <h3>{industry}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            employees = company_metrics.get('fullTimeEmployees', 'N/A')
            if isinstance(employees, (int, float)):
                employees = f"{employees:,.0f}"
            st.markdown(f"""
            <div class="info-card">
                <h4>👥 Employees</h4>
                <h3>{employees}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            market_cap = company_metrics.get('marketCap', 'N/A')
            if isinstance(market_cap, (int, float)):
                market_cap = f"₹{market_cap:,.0f}"
            st.markdown(f"""
            <div class="info-card">
                <h4>💰 Market Cap</h4>
                <h3>{market_cap}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🚀 TCS Stock Analysis Dashboard</h4>
        <p>Powered by Streamlit • Built with ❤️ for Professional Stock Analysis</p>
        <p>📊 Real-time Data • 🤖 ML Predictions • 📈 Advanced Analytics</p>
        <p><strong>Complete Project Features:</strong> EDA Plots • ML Models • Interactive Dashboard • CSV Downloads • Technical Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()