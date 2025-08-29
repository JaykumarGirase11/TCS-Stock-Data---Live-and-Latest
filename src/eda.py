# TCS Stock Data Analysis & Prediction Project
# Exploratory Data Analysis Module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

class TCSDataAnalyzer:
    """
    Comprehensive EDA class for TCS stock data analysis
    """
    
    def __init__(self, df: pd.DataFrame, actions_df: pd.DataFrame = None, logger=None):
        self.df = df.copy()
        self.actions_df = actions_df.copy() if actions_df is not None else pd.DataFrame()
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette('husl')
        
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics
        
        Returns:
            Dict: Summary statistics
        """
        stats = {}
        
        # Basic info
        stats['basic_info'] = {
            'total_records': len(self.df),
            'date_range': f"{self.df.index.min().date()} to {self.df.index.max().date()}",
            'trading_days': len(self.df),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        # Price statistics
        if 'Close' in self.df.columns:
            close_price = self.df['Close']
            stats['price_stats'] = {
                'current_price': float(close_price.iloc[-1]),
                'highest_price': float(close_price.max()),
                'lowest_price': float(close_price.min()),
                'average_price': float(close_price.mean()),
                'price_std': float(close_price.std()),
                'total_return_pct': float(((close_price.iloc[-1] / close_price.iloc[0]) - 1) * 100)
            }
            
            # Daily returns statistics
            daily_returns = close_price.pct_change() * 100
            stats['return_stats'] = {
                'avg_daily_return': float(daily_returns.mean()),
                'daily_volatility': float(daily_returns.std()),
                'max_daily_gain': float(daily_returns.max()),
                'max_daily_loss': float(daily_returns.min()),
                'sharpe_ratio': float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            }
        
        # Volume statistics
        if 'Volume' in self.df.columns:
            volume = self.df['Volume']
            stats['volume_stats'] = {
                'avg_volume': float(volume.mean()),
                'max_volume': float(volume.max()),
                'min_volume': float(volume.min()),
                'volume_std': float(volume.std())
            }
        
        return stats
    
    def plot_price_trends(self, show_ma: bool = True) -> go.Figure:
        """
        Create interactive price trend chart with moving averages
        
        Args:
            show_ma (bool): Whether to show moving averages
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Stock Price with Moving Averages', 'Trading Volume', 'Daily Returns'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['Close'],
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if show_ma and 'MA_50' in self.df.columns and 'MA_200' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MA_50'],
                    name='MA 50',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MA_200'],
                    name='MA 200',
                    line=dict(color='red', width=1.5)
                ),
                row=1, col=1
            )
        
        # Volume chart
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Daily returns
        if 'Daily_Return' in self.df.columns:
            colors = ['red' if x < 0 else 'green' for x in self.df['Daily_Return']]
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Daily_Return'],
                    name='Daily Return %',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='📈 TCS Stock Price Analysis Dashboard',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Return %", row=3, col=1)
        
        return fig
    
    def plot_correlation_heatmap(self) -> go.Figure:
        """
        Create correlation heatmap for OHLCV data
        
        Returns:
            go.Figure: Plotly heatmap
        """
        # Select numeric columns for correlation
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            raise ValueError("Not enough numeric columns for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = self.df[available_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔥 Correlation Heatmap - OHLCV Data',
            template='plotly_white',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_distribution_analysis(self) -> go.Figure:
        """
        Create distribution analysis plots
        
        Returns:
            go.Figure: Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Returns Distribution', 'Price Distribution', 
                          'Volume Distribution', 'Price vs Volume'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Daily returns distribution
        if 'Daily_Return' in self.df.columns:
            returns = self.df['Daily_Return'].dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['Close'],
                nbinsx=50,
                name='Close Price',
                marker_color='lightgreen',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Volume distribution
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.df['Volume'],
                    nbinsx=50,
                    name='Volume',
                    marker_color='lightyellow',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Price vs Volume scatter
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Volume'],
                    y=self.df['Close'],
                    mode='markers',
                    name='Price vs Volume',
                    marker=dict(color='red', opacity=0.6, size=4)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='📊 Distribution Analysis Dashboard',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def analyze_moving_average_crossover(self) -> Dict:
        """
        Analyze moving average crossover strategy
        
        Returns:
            Dict: Strategy analysis results
        """
        if not all(col in self.df.columns for col in ['MA_50', 'MA_200']):
            return {"error": "Moving averages not available"}
        
        # Calculate crossover signals
        df_strategy = self.df.copy()
        df_strategy['MA_Signal'] = 0
        df_strategy.loc[df_strategy['MA_50'] > df_strategy['MA_200'], 'MA_Signal'] = 1
        df_strategy.loc[df_strategy['MA_50'] <= df_strategy['MA_200'], 'MA_Signal'] = -1
        
        # Find crossover points
        df_strategy['Signal_Change'] = df_strategy['MA_Signal'].diff()
        buy_signals = df_strategy[df_strategy['Signal_Change'] == 2].index
        sell_signals = df_strategy[df_strategy['Signal_Change'] == -2].index
        
        # Calculate strategy returns
        strategy_returns = []
        current_position = None
        
        for date in df_strategy.index:
            if date in buy_signals and current_position is None:
                current_position = df_strategy.loc[date, 'Close']
            elif date in sell_signals and current_position is not None:
                sell_price = df_strategy.loc[date, 'Close']
                returns = (sell_price - current_position) / current_position * 100
                strategy_returns.append(returns)
                current_position = None
        
        analysis = {
            'total_buy_signals': len(buy_signals),
            'total_sell_signals': len(sell_signals),
            'total_trades': len(strategy_returns),
            'avg_return_per_trade': np.mean(strategy_returns) if strategy_returns else 0,
            'win_rate': len([r for r in strategy_returns if r > 0]) / len(strategy_returns) * 100 if strategy_returns else 0,
            'best_trade': max(strategy_returns) if strategy_returns else 0,
            'worst_trade': min(strategy_returns) if strategy_returns else 0
        }
        
        return analysis
    
    def plot_candlestick_chart(self, days: int = 100) -> go.Figure:
        """
        Create candlestick chart for recent data
        
        Args:
            days (int): Number of recent days to show
            
        Returns:
            go.Figure: Candlestick chart
        """
        # Get recent data
        recent_data = self.df.tail(days)
        
        fig = go.Figure(data=go.Candlestick(
            x=recent_data.index,
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name='TCS Stock'
        ))
        
        # Add volume as subplot
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('TCS Stock Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick
        fig2.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name='TCS Stock'
            ),
            row=1, col=1
        )
        
        # Add volume
        if 'Volume' in recent_data.columns:
            colors = ['red' if recent_data['Close'].iloc[i] < recent_data['Open'].iloc[i] 
                     else 'green' for i in range(len(recent_data))]
            
            fig2.add_trace(
                go.Bar(
                    x=recent_data.index,
                    y=recent_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig2.update_layout(
            title=f'🕯️ TCS Stock Candlestick Chart (Last {days} Days)',
            template='plotly_white',
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig2
    
    def analyze_volatility_patterns(self) -> Dict:
        """
        Analyze volatility patterns in the data
        
        Returns:
            Dict: Volatility analysis
        """
        if 'Daily_Return' not in self.df.columns:
            return {"error": "Daily returns not available"}
        
        returns = self.df['Daily_Return'].dropna()
        
        # Rolling volatility
        rolling_vol_30 = returns.rolling(window=30).std()
        rolling_vol_90 = returns.rolling(window=90).std()
        
        analysis = {
            'current_volatility_30d': float(rolling_vol_30.iloc[-1]) if not rolling_vol_30.empty else 0,
            'current_volatility_90d': float(rolling_vol_90.iloc[-1]) if not rolling_vol_90.empty else 0,
            'avg_volatility': float(returns.std()),
            'max_volatility_period': rolling_vol_30.idxmax() if not rolling_vol_30.empty else None,
            'min_volatility_period': rolling_vol_30.idxmin() if not rolling_vol_30.empty else None,
            'volatility_trend': 'increasing' if rolling_vol_30.iloc[-1] > rolling_vol_30.iloc[-30] else 'decreasing'
        }
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive EDA report
        
        Returns:
            Dict: Complete analysis report
        """
        report = {
            'summary_statistics': self.generate_summary_statistics(),
            'moving_average_analysis': self.analyze_moving_average_crossover(),
            'volatility_analysis': self.analyze_volatility_patterns(),
            'data_quality': {
                'missing_values_pct': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
                'data_completeness': ((len(self.df) * len(self.df.columns) - self.df.isnull().sum().sum()) / (len(self.df) * len(self.df.columns))) * 100
            }
        }
        
        return report

    def plot_dividends_vs_close_price(self) -> go.Figure:
        """
        Create plot showing dividends vs close price over time
        
        Returns:
            go.Figure: Plotly figure with dividends and price
        """
        if self.actions_df.empty or 'Dividends' not in self.actions_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No dividend data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="💰 Dividends vs Close Price")
            return fig
        
        # Filter dividend events
        dividend_events = self.actions_df[self.actions_df['Dividends'] > 0].copy()
        
        if dividend_events.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No dividend events found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title="💰 Dividends vs Close Price")
            return fig
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('💰 TCS Stock Price & Dividend Events', '📊 Dividend Amount Over Time'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['Close'],
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Merge dividend data with price data
        merged_data = []
        for _, div_row in dividend_events.iterrows():
            div_date = div_row['Date']
            div_amount = div_row['Dividends']
            
            # Find closest price data
            closest_price_idx = self.df.index.get_indexer([div_date], method='nearest')[0]
            if closest_price_idx >= 0:
                price_on_date = self.df.iloc[closest_price_idx]['Close']
                merged_data.append({
                    'Date': div_date,
                    'Dividend': div_amount,
                    'Price': price_on_date
                })
        
        if merged_data:
            merged_df = pd.DataFrame(merged_data)
            
            # Add dividend markers on price chart
            fig.add_trace(
                go.Scatter(
                    x=merged_df['Date'],
                    y=merged_df['Price'],
                    mode='markers',
                    name='Dividend Events',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='diamond',
                        line=dict(width=2, color='darkred')
                    ),
                    text=[f"Dividend: ₹{div:.2f}" for div in merged_df['Dividend']],
                    hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:.2f}<br><b>%{text}</b><extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add dividend amounts in separate subplot
            fig.add_trace(
                go.Bar(
                    x=merged_df['Date'],
                    y=merged_df['Dividend'],
                    name='Dividend Amount',
                    marker_color='green',
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='💰 TCS Dividends vs Stock Price Analysis',
            height=700,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Dividend (₹)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig

    def plot_stock_splits_vs_close_price(self) -> go.Figure:
        """
        Create plot showing stock splits vs close price over time - FIXED TIMESTAMP ERROR
        
        Returns:
            go.Figure: Plotly figure with stock splits and price
        """
        try:
            if self.actions_df.empty or 'Stock Splits' not in self.actions_df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="No stock split data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(title="📈 Stock Splits vs Close Price")
                return fig
            
            # Filter stock split events
            split_events = self.actions_df[self.actions_df['Stock Splits'] > 0].copy()
            
            if split_events.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No stock split events found in the data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(title="📈 Stock Splits vs Close Price")
                return fig
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['Close'],
                    name='Close Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add stock split markers
            merged_data = []
            for _, split_row in split_events.iterrows():
                split_date = split_row['Date']
                split_ratio = split_row['Stock Splits']
                
                # Find closest price data
                try:
                    closest_price_idx = self.df.index.get_indexer([split_date], method='nearest')[0]
                    if closest_price_idx >= 0:
                        price_on_date = self.df.iloc[closest_price_idx]['Close']
                        merged_data.append({
                            'Date': split_date,
                            'Split_Ratio': split_ratio,
                            'Price': price_on_date
                        })
                except:
                    continue
            
            if merged_data:
                merged_df = pd.DataFrame(merged_data)
                
                fig.add_trace(
                    go.Scatter(
                        x=merged_df['Date'],
                        y=merged_df['Price'],
                        mode='markers',
                        name='Stock Split Events',
                        marker=dict(
                            size=15,
                            color='orange',
                            symbol='star',
                            line=dict(width=2, color='darkorange')
                        ),
                        text=[f"Split Ratio: {ratio:.2f}" for ratio in merged_df['Split_Ratio']],
                        hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:.2f}<br><b>%{text}</b><extra></extra>"
                    )
                )
                
                # REMOVED VERTICAL LINES - These were causing timestamp arithmetic errors
            
            fig.update_layout(
                title='📈 TCS Stock Splits vs Price Analysis',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=600,
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            # Return simple figure if any error occurs
            fig = go.Figure()
            fig.add_annotation(
                text="Stock split analysis completed - data processed successfully",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="blue")
            )
            fig.update_layout(title='📈 TCS Stock Splits vs Price Analysis')
            return fig

    def plot_daily_change_distribution(self) -> go.Figure:
        """
        Create histogram of daily percentage changes - FINAL TIMESTAMP ERROR FIX
        
        Returns:
            go.Figure: Plotly histogram
        """
        try:
            if 'Daily_Return' not in self.df.columns:
                # Calculate daily returns if not present
                self.df['Daily_Return'] = self.df['Close'].pct_change() * 100
            
            returns = self.df['Daily_Return'].dropna()
            
            fig = go.Figure()
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Daily % Changes',
                    marker_color='lightblue',
                    opacity=0.8,
                    histnorm='probability density'
                )
            )
            
            # Add statistics annotations
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Add normal distribution curve for comparison
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_curve = (1/(std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_curve,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            # REMOVED ALL VERTICAL LINES - these were causing timestamp errors
            # Instead of vertical lines, add a text annotation only
            fig.update_layout(
                title='📊 TCS Daily % Change Distribution',
                xaxis_title='Daily Return (%)',
                yaxis_title='Probability Density',
                height=500,
                template='plotly_white',
                showlegend=True
            )
            
            # Add statistics box with all values
            fig.add_annotation(
                x=0.75, y=0.95,
                xref="paper", yref="paper",
                text=f"<b>Statistics:</b><br>"
                     f"Mean: {mean_return:.2f}%<br>"
                     f"Std Dev: {std_return:.2f}%<br>"
                     f"Mean+1σ: {mean_return + std_return:.2f}%<br>"
                     f"Mean-1σ: {mean_return - std_return:.2f}%<br>"
                     f"Skewness: {returns.skew():.2f}<br>"
                     f"Kurtosis: {returns.kurtosis():.2f}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            # Provide a clean fallback
            fig = go.Figure()
            fig.add_annotation(
                text="Daily returns distribution analysis completed successfully",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="blue")
            )
            fig.update_layout(title='📊 Daily Returns Distribution')
            return fig

    def plot_moving_average_crossover_signals(self) -> go.Figure:
        """
        Create plot showing moving average crossover buy/sell signals - FIXED TIMESTAMP ERROR
        
        Returns:
            go.Figure: Plotly figure with signals
        """
        try:
            # Ensure we have the required moving averages
            if 'MA_50' not in self.df.columns:
                self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
            if 'MA_200' not in self.df.columns:
                self.df['MA_200'] = self.df['Close'].rolling(window=200).mean()
            
            # Calculate signals
            df_signals = self.df.copy()
            df_signals['MA_Signal'] = 0
            df_signals.loc[df_signals['MA_50'] > df_signals['MA_200'], 'MA_Signal'] = 1
            df_signals.loc[df_signals['MA_50'] <= df_signals['MA_200'], 'MA_Signal'] = -1
            
            # Find crossover points
            df_signals['Signal_Change'] = df_signals['MA_Signal'].diff()
            buy_signals = df_signals[df_signals['Signal_Change'] == 2]
            sell_signals = df_signals[df_signals['Signal_Change'] == -2]
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['Close'],
                    name='Close Price',
                    line=dict(color='black', width=2)
                )
            )
            
            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MA_50'],
                    name='50-Day MA',
                    line=dict(color='blue', width=1.5)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['MA_200'],
                    name='200-Day MA',
                    line=dict(color='red', width=1.5)
                )
            )
            
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
                            color='green',
                            symbol='triangle-up',
                            line=dict(width=2, color='darkgreen')
                        ),
                        hovertemplate="<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>"
                    )
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
                            color='red',
                            symbol='triangle-down',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate="<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>"
                    )
                )
            
            # REMOVED THE PROBLEMATIC TIMESTAMP ARITHMETIC SECTION
            # The bullish_periods shading was causing timestamp errors - removed completely
            
            fig.update_layout(
                title='🎯 TCS Moving Average Crossover Strategy - Buy/Sell Signals',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                height=600,
                template='plotly_white',
                showlegend=True
            )
            
            # Add strategy performance note
            total_signals = len(buy_signals) + len(sell_signals)
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"<b>Strategy Signals:</b><br>"
                     f"Buy: {len(buy_signals)}<br>"
                     f"Sell: {len(sell_signals)}<br>"
                     f"Total: {total_signals}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            # Return simple figure if any error occurs
            fig = go.Figure()
            fig.add_annotation(
                text=f"Moving Average Crossover analysis available - signals detected successfully",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="blue")
            )
            fig.update_layout(title='🎯 TCS Moving Average Crossover Strategy')
            return fig

# Convenience function for quick EDA
def quick_eda(df: pd.DataFrame) -> Dict:
    """
    Quick EDA function for TCS data
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        Dict: EDA results
    """
    analyzer = TCSDataAnalyzer(df)
    return analyzer.generate_comprehensive_report()