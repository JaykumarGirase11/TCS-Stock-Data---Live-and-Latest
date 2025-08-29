# TCS Stock Data Analysis & Prediction Project
# Visualization Module for Streamlit Dashboard

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class TCSVisualizer:
    """
    Advanced visualization class for TCS stock data dashboard
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#4facfe',
            'danger': '#00f2fe',
            'warning': '#fa709a',
            'info': '#fee140',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'chart_colors': ['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#fa709a', '#fee140', '#ff6b6b', '#4ecdc4']
        }
    
    def create_interactive_candlestick(self, df: pd.DataFrame, days: int = 100) -> go.Figure:
        """
        Create interactive candlestick chart with volume
        
        Args:
            df (pd.DataFrame): Stock data
            days (int): Number of recent days to show
            
        Returns:
            go.Figure: Interactive candlestick chart
        """
        recent_data = df.tail(days)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('🕯️ TCS Stock Price', '📊 Trading Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name='TCS Stock',
                increasing_line_color=self.color_palette['success'],
                decreasing_line_color=self.color_palette['danger']
            ),
            row=1, col=1
        )
        
        # Volume bars
        if 'Volume' in recent_data.columns:
            colors = [self.color_palette['success'] if recent_data['Close'].iloc[i] >= recent_data['Open'].iloc[i] 
                     else self.color_palette['danger'] for i in range(len(recent_data))]
            
            fig.add_trace(
                go.Bar(
                    x=recent_data.index,
                    y=recent_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add moving averages if available
        if 'MA_20' in recent_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['MA_20'],
                    name='MA 20',
                    line=dict(color=self.color_palette['warning'], width=2)
                ),
                row=1, col=1
            )
        
        if 'MA_50' in recent_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['MA_50'],
                    name='MA 50',
                    line=dict(color=self.color_palette['info'], width=2)
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title=f'🕯️ TCS Stock Candlestick Chart (Last {days} Days)',
            template='plotly_white',
            height=700,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_technical_indicators_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive technical indicators dashboard
        
        Args:
            df (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            go.Figure: Technical analysis dashboard
        """
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                '📈 Price with Bollinger Bands', '📊 RSI (14)',
                '🎯 MACD', '⚡ Stochastic Oscillator',
                '📏 ATR (Average True Range)', '🌊 Williams %R',
                '📈 Price Rate of Change', '📊 Volume Analysis'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Price with Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Close Price',
                line=dict(color=self.color_palette['primary'], width=2)
            ),
            row=1, col=1
        )
        
        if 'BB_upper_20' in df.columns and 'BB_lower_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_upper_20'],
                    name='BB Upper',
                    line=dict(color=self.color_palette['danger'], width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_lower_20'],
                    name='BB Lower',
                    line=dict(color=self.color_palette['success'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
                    name='RSI (14)',
                    line=dict(color=self.color_palette['secondary'], width=2)
                ),
                row=1, col=2
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2, opacity=0.5)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2, opacity=0.3)
        
        # 3. MACD
        if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_signal'],
                    name='Signal',
                    line=dict(color=self.color_palette['danger'], width=2)
                ),
                row=2, col=1
            )
            
            # MACD Histogram
            colors = [self.color_palette['success'] if x > 0 else self.color_palette['danger'] 
                     for x in df['MACD_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # 4. Stochastic Oscillator
        if 'Stoch_K_14' in df.columns and 'Stoch_D_3' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Stoch_K_14'],
                    name='%K',
                    line=dict(color=self.color_palette['warning'], width=2)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Stoch_D_3'],
                    name='%D',
                    line=dict(color=self.color_palette['info'], width=2)
                ),
                row=2, col=2
            )
            
            # Stochastic reference lines
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=2, opacity=0.5)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=2, opacity=0.5)
        
        # 5. ATR
        if 'ATR_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ATR_14'],
                    name='ATR (14)',
                    line=dict(color=self.color_palette['secondary'], width=2),
                    fill='tonexty'
                ),
                row=3, col=1
            )
        
        # 6. Williams %R
        if 'Williams_R_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Williams_R_14'],
                    name='Williams %R',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=3, col=2
            )
            
            # Williams %R reference lines
            fig.add_hline(y=-20, line_dash="dash", line_color="red", row=3, col=2, opacity=0.5)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", row=3, col=2, opacity=0.5)
        
        # 7. Price Rate of Change
        if 'ROC_10' in df.columns:
            colors = [self.color_palette['success'] if x > 0 else self.color_palette['danger'] 
                     for x in df['ROC_10']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['ROC_10'],
                    name='ROC (10)',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=4, col=1
            )
        
        # 8. Volume Analysis
        if 'Volume' in df.columns and 'Volume_MA_20' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=self.color_palette['secondary'],
                    opacity=0.6
                ),
                row=4, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Volume_MA_20'],
                    name='Volume MA',
                    line=dict(color=self.color_palette['danger'], width=2)
                ),
                row=4, col=2
            )
        
        fig.update_layout(
            title='🎯 TCS Technical Indicators Dashboard',
            height=1200,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_dashboard(self, model_results: Dict[str, Dict]) -> go.Figure:
        """
        Create comprehensive model performance dashboard
        
        Args:
            model_results (Dict): Results from multiple models
            
        Returns:
            go.Figure: Performance dashboard
        """
        # Prepare metrics data
        metrics_data = []
        for model_name, results in model_results.items():
            if 'metrics' in results and 'test' in results['metrics']:
                metrics = results['metrics']['test']
                metrics_data.append({
                    'Model': model_name,
                    'R²': metrics.get('r2', 0),
                    'RMSE': metrics.get('rmse', 0),
                    'MAE': metrics.get('mae', 0),
                    'MAPE': metrics.get('mape', 0) if 'mape' in metrics else 0
                })
        
        if not metrics_data:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No model performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📈 R² Score (Higher = Better)', '📊 RMSE (Lower = Better)', 
                          '📉 MAE (Lower = Better)', '⚡ MAPE (Lower = Better)'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = metrics_df['Model'].tolist()
        colors = self.color_palette['chart_colors'][:len(models)]
        
        # R² Score
        fig.add_trace(
            go.Bar(
                x=models,
                y=metrics_df['R²'],
                name='R²',
                marker_color=colors,
                text=[f'{x:.4f}' for x in metrics_df['R²']],
                textposition='auto',
                hovertemplate='Model: %{x}<br>R²: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(
                x=models,
                y=metrics_df['RMSE'],
                name='RMSE',
                marker_color=colors,
                text=[f'{x:.2f}' for x in metrics_df['RMSE']],
                textposition='auto',
                hovertemplate='Model: %{x}<br>RMSE: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(
                x=models,
                y=metrics_df['MAE'],
                name='MAE',
                marker_color=colors,
                text=[f'{x:.2f}' for x in metrics_df['MAE']],
                textposition='auto',
                hovertemplate='Model: %{x}<br>MAE: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # MAPE
        if metrics_df['MAPE'].sum() > 0:  # Only show if MAPE data exists
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=metrics_df['MAPE'],
                    name='MAPE',
                    marker_color=colors,
                    text=[f'{x:.2f}%' for x in metrics_df['MAPE']],
                    textposition='auto',
                    hovertemplate='Model: %{x}<br>MAPE: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='🏆 Model Performance Comparison Dashboard',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_prediction_analysis_dashboard(self, actual: np.ndarray, predicted: np.ndarray, 
                                           dates: pd.DatetimeIndex, model_name: str) -> go.Figure:
        """
        Create detailed prediction analysis dashboard
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            dates (pd.DatetimeIndex): Date index
            model_name (str): Name of the model
            
        Returns:
            go.Figure: Prediction analysis dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'🎯 {model_name} - Actual vs Predicted',
                '📊 Prediction Accuracy Distribution',
                '📈 Residuals Over Time',
                '⚡ Residuals Distribution'
            ),
            specs=[[{"secondary_y": False}, {"type": "histogram"}],
                   [{"secondary_y": False}, {"type": "histogram"}]]
        )
        
        # Calculate residuals and errors
        residuals = actual - predicted
        abs_error = np.abs(residuals)
        percentage_error = (residuals / actual) * 100
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                name='Actual',
                line=dict(color=self.color_palette['primary'], width=3),
                hovertemplate='Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predicted,
                name='Predicted',
                line=dict(color=self.color_palette['danger'], width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Prediction Accuracy Distribution
        accuracy = 100 - np.abs(percentage_error)
        fig.add_trace(
            go.Histogram(
                x=accuracy,
                nbinsx=30,
                name='Accuracy %',
                marker_color=self.color_palette['success'],
                opacity=0.7,
                hovertemplate='Accuracy: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Residuals Over Time
        colors = [self.color_palette['success'] if x > 0 else self.color_palette['danger'] for x in residuals]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=residuals,
                name='Residuals',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Residual: ₹%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1, opacity=0.5)
        
        # 4. Residuals Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residuals Dist',
                marker_color=self.color_palette['warning'],
                opacity=0.7,
                histnorm='probability density',
                hovertemplate='Residual: ₹%{x:.2f}<br>Density: %{y:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add normal distribution curve
        residuals_std = np.std(residuals)
        residuals_mean = np.mean(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = (1/(residuals_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - residuals_mean) / residuals_std) ** 2)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_curve,
                mode='lines',
                name='Normal Curve',
                line=dict(color=self.color_palette['primary'], width=2, dash='dot')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'🔍 {model_name} - Detailed Prediction Analysis',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_trading_signals_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Create trading signals and strategy dashboard
        
        Args:
            df (pd.DataFrame): Stock data with signals
            
        Returns:
            go.Figure: Trading signals dashboard
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '🎯 Price with Trading Signals',
                '📊 Signal Strength Indicators',
                '💰 Strategy Performance'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Price with Trading Signals
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Close Price',
                line=dict(color=self.color_palette['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'MA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA_50'],
                    name='MA 50',
                    line=dict(color=self.color_palette['warning'], width=1.5)
                ),
                row=1, col=1
            )
        
        if 'MA_200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA_200'],
                    name='MA 200',
                    line=dict(color=self.color_palette['danger'], width=1.5)
                ),
                row=1, col=1
            )
        
        # Generate simple buy/sell signals based on MA crossover
        if 'MA_50' in df.columns and 'MA_200' in df.columns:
            df_signals = df.copy()
            df_signals['MA_Signal'] = 0
            df_signals.loc[df_signals['MA_50'] > df_signals['MA_200'], 'MA_Signal'] = 1
            df_signals.loc[df_signals['MA_50'] <= df_signals['MA_200'], 'MA_Signal'] = -1
            
            # Find crossover points
            df_signals['Signal_Change'] = df_signals['MA_Signal'].diff()
            buy_signals = df_signals[df_signals['Signal_Change'] == 2]
            sell_signals = df_signals[df_signals['Signal_Change'] == -2]
            
            # Add buy signals
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Close'],
                        mode='markers',
                        name='Buy Signals',
                        marker=dict(
                            size=12,
                            color=self.color_palette['success'],
                            symbol='triangle-up',
                            line=dict(width=2, color='darkgreen')
                        ),
                        hovertemplate='BUY: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Add sell signals
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['Close'],
                        mode='markers',
                        name='Sell Signals',
                        marker=dict(
                            size=12,
                            color=self.color_palette['danger'],
                            symbol='triangle-down',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate='SELL: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. Signal Strength Indicators
        if 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
                    name='RSI (14)',
                    line=dict(color=self.color_palette['secondary'], width=2)
                ),
                row=2, col=1
            )
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        
        # 3. Strategy Performance (Cumulative Returns)
        if 'Daily_Return' in df.columns:
            # Calculate cumulative returns
            df['Cumulative_Return'] = (1 + df['Daily_Return'] / 100).cumprod()
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Cumulative_Return'],
                    name='Cumulative Returns',
                    line=dict(color=self.color_palette['success'], width=2),
                    fill='tonexty'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='🎯 TCS Trading Signals & Strategy Dashboard',
            height=900,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_summary_cards_html(self, stats: Dict) -> str:
        """
        Create HTML for summary cards with enhanced styling
        
        Args:
            stats (Dict): Summary statistics
            
        Returns:
            str: HTML string for cards
        """
        html = ""
        
        if 'price_stats' in stats:
            price_stats = stats['price_stats']
            total_return = price_stats.get('total_return_pct', 0)
            return_color = "#4facfe" if total_return >= 0 else "#fa709a"
            return_symbol = "📈" if total_return >= 0 else "📉"
            
            html += f"""
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 1rem; border-radius: 10px; color: white; text-align: center; flex: 1;">
                    <h4 style="margin: 0;">💰 Current Price</h4>
                    <h2 style="margin: 0.5rem 0;">₹{price_stats['current_price']:,.2f}</h2>
                </div>
                <div style="background: linear-gradient(135deg, {return_color} 0%, #fee140 100%); 
                           padding: 1rem; border-radius: 10px; color: white; text-align: center; flex: 1;">
                    <h4 style="margin: 0;">{return_symbol} Total Return</h4>
                    <h2 style="margin: 0.5rem 0;">{total_return:+.2f}%</h2>
                </div>
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           padding: 1rem; border-radius: 10px; color: white; text-align: center; flex: 1;">
                    <h4 style="margin: 0;">📈 Highest Price</h4>
                    <h2 style="margin: 0.5rem 0;">₹{price_stats['highest_price']:,.2f}</h2>
                </div>
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                           padding: 1rem; border-radius: 10px; color: white; text-align: center; flex: 1;">
                    <h4 style="margin: 0;">📉 Lowest Price</h4>
                    <h2 style="margin: 0.5rem 0;">₹{price_stats['lowest_price']:,.2f}</h2>
                </div>
            </div>
            """
        
        return html

# Convenience functions for dashboard
def create_dashboard_charts(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create all dashboard charts
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        Dict[str, go.Figure]: Dictionary of charts
    """
    visualizer = TCSVisualizer()
    
    charts = {
        'candlestick': visualizer.create_interactive_candlestick(df),
        'technical_indicators': visualizer.create_technical_indicators_dashboard(df),
        'trading_signals': visualizer.create_trading_signals_dashboard(df)
    }
    
    return charts

def create_model_charts(model_results: Dict) -> Dict[str, go.Figure]:
    """
    Create model-related charts
    
    Args:
        model_results (Dict): Model results
        
    Returns:
        Dict[str, go.Figure]: Dictionary of model charts
    """
    visualizer = TCSVisualizer()
    
    charts = {
        'performance': visualizer.create_model_performance_dashboard(model_results)
    }
    
    # Add individual model analysis charts
    for model_name, results in model_results.items():
        if 'predictions' in results and 'actuals' in results:
            charts[f'{model_name}_analysis'] = visualizer.create_prediction_analysis_dashboard(
                results['actuals']['test'],
                results['predictions']['test'],
                results['indices']['test'],
                model_name
            )
    
    return charts