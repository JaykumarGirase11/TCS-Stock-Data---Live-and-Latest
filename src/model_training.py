# TCS Stock Data Analysis & Prediction Project
# Model Training Module

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
import os
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

class TCSModelTrainer:
    """
    Comprehensive model training class for TCS stock prediction
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.model_performances = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    validation_size: float = 0.1) -> Dict:
        """
        Prepare data for training with proper time series split
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set proportion
            validation_size (float): Validation set proportion
            
        Returns:
            Dict: Prepared datasets
        """
        # Time series split (no shuffling)
        n_samples = len(X)
        train_end = int(n_samples * (1 - test_size - validation_size))
        val_end = int(n_samples * (1 - test_size))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Scale target for LSTM
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Store scalers
        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y
        
        data_splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled, 'y_val_scaled': y_val_scaled, 'y_test_scaled': y_test_scaled,
            'feature_names': X.columns.tolist()
        }
        
        self.logger.info(f"✅ Data prepared - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return data_splits
    
    def train_linear_regression_models(self, data: Dict) -> Dict:
        """
        Train multiple linear regression models
        
        Args:
            data (Dict): Prepared data splits
            
        Returns:
            Dict: Trained models and performances
        """
        self.logger.info("🤖 Training Linear Regression Models...")
        
        models_to_train = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            try:
                # Train model
                if name in ['LinearRegression', 'Ridge', 'Lasso']:
                    model.fit(data['X_train_scaled'], data['y_train'])
                    y_pred_train = model.predict(data['X_train_scaled'])
                    y_pred_val = model.predict(data['X_val_scaled'])
                    y_pred_test = model.predict(data['X_test_scaled'])
                else:
                    model.fit(data['X_train'], data['y_train'])
                    y_pred_train = model.predict(data['X_train'])
                    y_pred_val = model.predict(data['X_val'])
                    y_pred_test = model.predict(data['X_test'])
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(data['y_train'], y_pred_train)
                val_metrics = self._calculate_metrics(data['y_val'], y_pred_val)
                test_metrics = self._calculate_metrics(data['y_test'], y_pred_test)
                
                # Store results
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'predictions': {
                        'train': y_pred_train,
                        'val': y_pred_val,
                        'test': y_pred_test
                    }
                }
                
                self.logger.info(f"✅ {name} - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.2f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error training {name}: {str(e)}")
        
        # Store models
        self.models.update(results)
        
        return results
    
    def train_lstm_model(self, data: Dict, sequence_length: int = 60, 
                        epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train LSTM model for time series prediction
        
        Args:
            data (Dict): Prepared data
            sequence_length (int): Length of input sequences
            epochs (int): Training epochs
            batch_size (int): Batch size
            
        Returns:
            Dict: LSTM model and performance
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("❌ TensorFlow not available, skipping LSTM training")
            return {}
        
        self.logger.info("🧠 Training LSTM Model...")
        
        try:
            # Prepare sequences for LSTM
            X_train_seq, y_train_seq = self._create_sequences(
                data['X_train_scaled'], data['y_train_scaled'], sequence_length
            )
            X_val_seq, y_val_seq = self._create_sequences(
                data['X_val_scaled'], data['y_val_scaled'], sequence_length
            )
            X_test_seq, y_test_seq = self._create_sequences(
                data['X_test_scaled'], data['y_test_scaled'], sequence_length
            )
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.0001
            )
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            y_pred_train = model.predict(X_train_seq).flatten()
            y_pred_val = model.predict(X_val_seq).flatten()
            y_pred_test = model.predict(X_test_seq).flatten()
            
            # Inverse transform predictions
            y_pred_train_original = self.scalers['y'].inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
            y_pred_val_original = self.scalers['y'].inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
            y_pred_test_original = self.scalers['y'].inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
            
            # Inverse transform actual values for metrics
            y_train_original = self.scalers['y'].inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
            y_val_original = self.scalers['y'].inverse_transform(y_val_seq.reshape(-1, 1)).flatten()
            y_test_original = self.scalers['y'].inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train_original, y_pred_train_original)
            val_metrics = self._calculate_metrics(y_val_original, y_pred_val_original)
            test_metrics = self._calculate_metrics(y_test_original, y_pred_test_original)
            
            lstm_results = {
                'model': model,
                'history': history.history,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'predictions': {
                    'train': y_pred_train_original,
                    'val': y_pred_val_original,
                    'test': y_pred_test_original
                },
                'sequence_length': sequence_length
            }
            
            self.models['LSTM'] = lstm_results
            
            self.logger.info(f"✅ LSTM - Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.2f}")
            
            return lstm_results
            
        except Exception as e:
            self.logger.error(f"❌ Error training LSTM: {str(e)}")
            return {}
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def hyperparameter_tuning(self, data: Dict, model_name: str = 'RandomForest') -> Dict:
        """
        Perform hyperparameter tuning for selected models
        
        Args:
            data (Dict): Prepared data
            model_name (str): Model to tune
            
        Returns:
            Dict: Best parameters and model
        """
        self.logger.info(f"🔧 Hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids:
            self.logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        try:
            # Base model
            if model_name == 'RandomForest':
                base_model = RandomForestRegressor(random_state=42)
            elif model_name == 'GradientBoosting':
                base_model = GradientBoostingRegressor(random_state=42)
            
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(data['X_train'], data['y_train'])
            
            # Best model predictions
            best_model = grid_search.best_estimator_
            y_pred_test = best_model.predict(data['X_test'])
            test_metrics = self._calculate_metrics(data['y_test'], y_pred_test)
            
            tuning_results = {
                'best_model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_metrics': test_metrics
            }
            
            self.logger.info(f"✅ Best {model_name} - R²: {test_metrics['r2']:.4f}")
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in hyperparameter tuning: {str(e)}")
            return {}
    
    def get_feature_importance(self, model_name: str = 'RandomForest', feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Model name
            feature_names (List[str]): Feature names
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if model_name not in self.models:
            self.logger.warning(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names or [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning(f"Model {model_name} does not have feature importances")
            return pd.DataFrame()
    
    def save_models(self, save_dir: str = 'models/') -> None:
        """
        Save trained models
        
        Args:
            save_dir (str): Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model_info in self.models.items():
            try:
                if name == 'LSTM' and TENSORFLOW_AVAILABLE:
                    # Save Keras model
                    model_info['model'].save(os.path.join(save_dir, f'{name}_model.h5'))
                else:
                    # Save sklearn model
                    joblib.dump(model_info['model'], os.path.join(save_dir, f'{name}_model.pkl'))
                
                self.logger.info(f"✅ Saved {name} model")
                
            except Exception as e:
                self.logger.error(f"❌ Error saving {name}: {str(e)}")
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
        self.logger.info("✅ Saved scalers")
    
    def load_models(self, save_dir: str = 'models/') -> None:
        """
        Load trained models
        
        Args:
            save_dir (str): Directory to load models from
        """
        try:
            # Load scalers
            self.scalers = joblib.load(os.path.join(save_dir, 'scalers.pkl'))
            
            # Load models
            for file in os.listdir(save_dir):
                if file.endswith('.pkl') and file != 'scalers.pkl':
                    model_name = file.replace('_model.pkl', '')
                    model = joblib.load(os.path.join(save_dir, file))
                    self.models[model_name] = {'model': model}
                    self.logger.info(f"✅ Loaded {model_name}")
                
                elif file.endswith('.h5') and TENSORFLOW_AVAILABLE:
                    model_name = file.replace('_model.h5', '')
                    model = tf.keras.models.load_model(os.path.join(save_dir, file))
                    self.models[model_name] = {'model': model}
                    self.logger.info(f"✅ Loaded {model_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {str(e)}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models
        
        Returns:
            pd.DataFrame: Model comparison
        """
        comparison_data = []
        
        for name, model_info in self.models.items():
            if 'test_metrics' in model_info:
                metrics = model_info['test_metrics']
                comparison_data.append({
                    'Model': name,
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape']
                })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('R²', ascending=False)
        return comparison_df
    
    def predict_future(self, model_name: str, X_future: np.ndarray, periods: int = 30) -> np.ndarray:
        """
        Make future predictions using trained model
        
        Args:
            model_name (str): Model to use for prediction
            X_future (np.ndarray): Future features
            periods (int): Number of periods to predict
            
        Returns:
            np.ndarray: Future predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        
        if model_name == 'LSTM':
            # Handle LSTM predictions differently
            predictions = []
            # Implementation would depend on specific LSTM architecture
            pass
        else:
            # Standard ML model prediction
            if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
                X_future_scaled = self.scalers['X'].transform(X_future)
                predictions = model.predict(X_future_scaled)
            else:
                predictions = model.predict(X_future)
        
        return predictions

# Convenience function
def train_all_models(X: pd.DataFrame, y: pd.Series) -> TCSModelTrainer:
    """
    Train all models with default settings
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        
    Returns:
        TCSModelTrainer: Trained model trainer
    """
    trainer = TCSModelTrainer()
    data = trainer.prepare_data(X, y)
    trainer.train_linear_regression_models(data)
    trainer.train_lstm_model(data)
    return trainer