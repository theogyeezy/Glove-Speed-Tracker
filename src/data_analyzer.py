"""
Data analysis and model training module for the Glove Speed Tracker application.
Provides advanced analysis of glove movement patterns and trains detection models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import configuration
from config import OUTPUT_DIR, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_analyzer')

class GloveDataAnalyzer:
    """
    Class for analyzing glove movement data and identifying patterns.
    """
    
    def __init__(self):
        """
        Initialize the data analyzer.
        """
        self.speed_data = None
        self.features = None
        self.clusters = None
        self.pca_result = None
        self.prediction_model = None
        
        logger.info("GloveDataAnalyzer initialized")
    
    def load_speed_data(self, speed_data_path):
        """
        Load speed data from a CSV file.
        
        Args:
            speed_data_path (str): Path to the speed data CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.speed_data = pd.read_csv(speed_data_path)
            logger.info(f"Loaded speed data from {speed_data_path} with {len(self.speed_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading speed data: {str(e)}")
            return False
    
    def set_speed_data(self, speed_data):
        """
        Set speed data directly from a DataFrame.
        
        Args:
            speed_data (DataFrame): Speed data
            
        Returns:
            bool: True if data set successfully, False otherwise
        """
        try:
            self.speed_data = speed_data
            logger.info(f"Set speed data with {len(self.speed_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error setting speed data: {str(e)}")
            return False
    
    def extract_features(self):
        """
        Extract features from speed data for analysis.
        
        Returns:
            pandas.DataFrame: DataFrame with extracted features
        """
        if self.speed_data is None or len(self.speed_data) < 10:
            logger.error("Insufficient speed data for feature extraction")
            return None
        
        try:
            # Create a new DataFrame for features
            features_dict = {}
            
            # Basic statistical features
            if 'smooth_speed_mps' in self.speed_data.columns:
                features_dict['avg_speed'] = [self.speed_data['smooth_speed_mps'].mean()]
                features_dict['max_speed'] = [self.speed_data['smooth_speed_mps'].max()]
                features_dict['min_speed'] = [self.speed_data['smooth_speed_mps'].min()]
                features_dict['std_speed'] = [self.speed_data['smooth_speed_mps'].std()]
            elif 'speed_mps' in self.speed_data.columns:
                features_dict['avg_speed'] = [self.speed_data['speed_mps'].mean()]
                features_dict['max_speed'] = [self.speed_data['speed_mps'].max()]
                features_dict['min_speed'] = [self.speed_data['speed_mps'].min()]
                features_dict['std_speed'] = [self.speed_data['speed_mps'].std()]
            
            # Calculate speed changes
            if 'smooth_speed_mps' in self.speed_data.columns:
                speed_changes = self.speed_data['smooth_speed_mps'].diff().dropna()
                features_dict['avg_acceleration'] = [speed_changes.mean()]
                features_dict['max_acceleration'] = [speed_changes.max()]
                features_dict['min_acceleration'] = [speed_changes.min()]
                features_dict['std_acceleration'] = [speed_changes.std()]
            elif 'speed_mps' in self.speed_data.columns:
                speed_changes = self.speed_data['speed_mps'].diff().dropna()
                features_dict['avg_acceleration'] = [speed_changes.mean()]
                features_dict['max_acceleration'] = [speed_changes.max()]
                features_dict['min_acceleration'] = [speed_changes.min()]
                features_dict['std_acceleration'] = [speed_changes.std()]
            
            # Calculate movement direction changes
            if 'dx' in self.speed_data.columns and 'dy' in self.speed_data.columns:
                # Calculate angles of movement
                angles = np.arctan2(self.speed_data['dy'], self.speed_data['dx'])
                angle_changes = np.abs(np.diff(angles))
                # Adjust for circular nature of angles
                angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
                
                features_dict['avg_direction_change'] = [np.mean(angle_changes)]
                features_dict['max_direction_change'] = [np.max(angle_changes)]
                features_dict['total_direction_changes'] = [np.sum(angle_changes > 0.5)]  # Changes greater than ~30 degrees
            
            # Calculate total distance traveled
            if 'displacement_meters' in self.speed_data.columns:
                features_dict['total_distance'] = [self.speed_data['displacement_meters'].sum()]
            
            # Calculate movement efficiency (straight-line distance / total distance)
            if 'center_x' in self.speed_data.columns and 'center_y' in self.speed_data.columns:
                start_x, start_y = self.speed_data['center_x'].iloc[0], self.speed_data['center_y'].iloc[0]
                end_x, end_y = self.speed_data['center_x'].iloc[-1], self.speed_data['center_y'].iloc[-1]
                
                straight_line_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                if 'pixels_per_meter' in self.speed_data.columns:
                    straight_line_distance = straight_line_distance / self.speed_data['pixels_per_meter'].iloc[0]
                else:
                    straight_line_distance = straight_line_distance / 100  # Default pixels per meter
                
                if 'total_distance' in features_dict and features_dict['total_distance'][0] > 0:
                    features_dict['movement_efficiency'] = [straight_line_distance / features_dict['total_distance'][0]]
            
            # Time-based features
            if 'time' in self.speed_data.columns:
                features_dict['movement_duration'] = [self.speed_data['time'].max() - self.speed_data['time'].min()]
            
            # Create DataFrame from dictionary
            features = pd.DataFrame(features_dict)
            
            # Store the features
            self.features = features
            
            logger.info(f"Extracted {len(features.columns)} features from speed data")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def cluster_movements(self, n_clusters=3):
        """
        Cluster movement patterns using K-means.
        
        Args:
            n_clusters (int): Number of clusters to identify
            
        Returns:
            tuple: (cluster_labels, cluster_centers)
        """
        if self.speed_data is None or len(self.speed_data) < n_clusters * 5:
            logger.error(f"Insufficient data for clustering into {n_clusters} clusters")
            return None, None
        
        try:
            # Prepare data for clustering
            cluster_columns = []
            
            if 'smooth_speed_mps' in self.speed_data.columns:
                cluster_columns.append('smooth_speed_mps')
            elif 'speed_mps' in self.speed_data.columns:
                cluster_columns.append('speed_mps')
                
            if 'smooth_acceleration_mps2' in self.speed_data.columns:
                cluster_columns.append('smooth_acceleration_mps2')
            elif 'acceleration_mps2' in self.speed_data.columns:
                cluster_columns.append('acceleration_mps2')
            
            if len(cluster_columns) == 0:
                # If no speed/acceleration columns, try using displacement
                if 'displacement_meters' in self.speed_data.columns:
                    cluster_columns.append('displacement_meters')
                # And time if available
                if 'dt' in self.speed_data.columns:
                    cluster_columns.append('dt')
            
            if len(cluster_columns) == 0:
                logger.error("No suitable columns found for clustering")
                return None, None
                
            # Use available columns for clustering
            X = self.speed_data[cluster_columns].dropna().values
            
            if len(X) < n_clusters:
                logger.error(f"Not enough data points ({len(X)}) for {n_clusters} clusters")
                return None, None
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Get cluster centers in original scale
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # Add cluster labels to speed data
            self.speed_data = self.speed_data.copy()
            self.speed_data['cluster'] = np.nan
            
            # Get indices of rows used for clustering
            valid_indices = self.speed_data[cluster_columns].dropna().index
            
            # Assign cluster labels to those indices
            self.speed_data.loc[valid_indices, 'cluster'] = cluster_labels
            
            # Store clustering results
            self.clusters = {
                'labels': cluster_labels,
                'centers': cluster_centers,
                'n_clusters': n_clusters,
                'feature_names': cluster_columns
            }
            
            logger.info(f"Clustered movement data into {n_clusters} patterns using {cluster_columns}")
            return cluster_labels, cluster_centers
            
        except Exception as e:
            logger.error(f"Error clustering movements: {str(e)}")
            return None, None
    
    def perform_pca(self, n_components=2):
        """
        Perform Principal Component Analysis on movement data.
        
        Args:
            n_components (int): Number of principal components
            
        Returns:
            tuple: (pca_result, explained_variance_ratio)
        """
        if self.speed_data is None or len(self.speed_data) < 10:
            logger.error("Insufficient data for PCA")
            return None, None
        
        try:
            # Prepare data for PCA
            # Try to use as many relevant columns as possible
            potential_columns = ['smooth_speed_mps', 'speed_mps', 'smooth_acceleration_mps2', 
                                'acceleration_mps2', 'dx', 'dy', 'displacement_meters']
            
            columns_to_use = [col for col in potential_columns if col in self.speed_data.columns]
            
            if not columns_to_use:
                logger.error("No suitable columns found for PCA")
                return None, None
            
            X = self.speed_data[columns_to_use].dropna().values
            
            if X.shape[0] < 3 or X.shape[1] < 2:
                logger.error(f"Insufficient data for PCA: {X.shape[0]} samples, {X.shape[1]} features")
                return None, None
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, X.shape[1]))
            pca_result = pca.fit_transform(X_scaled)
            
            # Store PCA results
            self.pca_result = {
                'result': pca_result,
                'explained_variance': pca.explained_variance_ratio_,
                'components': pca.components_,
                'feature_names': columns_to_use
            }
            
            logger.info(f"Performed PCA with {n_components} components, explaining {sum(pca.explained_variance_ratio_)*100:.2f}% of variance")
            return pca_result, pca.explained_variance_ratio_
            
        except Exception as e:
            logger.error(f"Error performing PCA: {str(e)}")
            return None, None
    
    def train_speed_prediction_model(self, test_size=0.2):
        """
        Train a model to predict glove speed based on position and previous movements.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (model, r2_score, mean_squared_error)
        """
        if self.speed_data is None or len(self.speed_data) < 20:
            logger.error("Insufficient data for model training")
            return None, None, None
        
        try:
            # Prepare features and target
            data = self.speed_data.copy()
            
            # Identify position columns
            position_cols = []
            if 'center_x' in data.columns and 'center_y' in data.columns:
                position_cols = ['center_x', 'center_y']
            
            # If no position columns, try to use frame number as a feature
            if not position_cols and 'frame' in data.columns:
                position_cols = ['frame']
                
            if not position_cols:
                logger.error("No position columns found for model training")
                return None, None, None
            
            # Create lag features (previous positions) if we have enough data
            if len(data) > 5:
                for lag in range(1, min(4, len(data) // 5)):  # Use up to 3 previous positions, but not more than 1/5 of data
                    for col in position_cols:
                        data[f'{col}_lag{lag}'] = data[col].shift(lag)
            
            # Drop rows with NaN values (from lag creation)
            data = data.dropna()
            
            if len(data) < 10:
                logger.error(f"Insufficient data after creating lag features: {len(data)} rows")
                return None, None, None
            
            # Define features and target
            feature_cols = []
            for col in data.columns:
                if col.startswith('center_x') or col.startswith('center_y') or col.startswith('frame_lag'):
                    feature_cols.append(col)
            
            if 'time' in data.columns:
                feature_cols.append('time')
                
            if not feature_cols:
                logger.error("No feature columns identified for model training")
                return None, None, None
            
            # Identify target column
            target_col = None
            for col in ['smooth_speed_mps', 'speed_mps']:
                if col in data.columns:
                    target_col = col
                    break
                    
            if target_col is None:
                logger.error("No target column (speed) found for model training")
                return None, None, None
            
            X = data[feature_cols].values
            y = data[target_col].values
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train a Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store the model
            self.prediction_model = {
                'model': model,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'r2_score': r2,
                'mse': mse
            }
            
            logger.info(f"Trained speed prediction model with R² score: {r2:.4f}, MSE: {mse:.4f}")
            return model, r2, mse
            
        except Exception as e:
            logger.error(f"Error training prediction model: {str(e)}")
            return None, None, None
    
    def save_prediction_model(self, output_path=None):
        """
        Save the trained prediction model.
        
        Args:
            output_path (str, optional): Path to save the model
            
        Returns:
            str: Path to the saved model or None if saving failed
        """
        if self.prediction_model is None or 'model' not in self.prediction_model:
            logger.error("No prediction model available to save")
            return None
        
        try:
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(MODELS_DIR, exist_ok=True)
                output_path = os.path.join(MODELS_DIR, f"speed_prediction_model_{timestamp}.joblib")
            
            # Save the model
            joblib.dump(self.prediction_model, output_path)
            
            logger.info(f"Prediction model saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving prediction model: {str(e)}")
            return None
    
    def load_prediction_model(self, model_path):
        """
        Load a trained prediction model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.prediction_model = joblib.load(model_path)
            logger.info(f"Loaded prediction model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading prediction model: {str(e)}")
            return False
    
    def predict_speed(self, positions):
        """
        Predict glove speed based on positions.
        
        Args:
            positions (list): List of position tuples [(x1, y1), (x2, y2), ...]
            
        Returns:
            float: Predicted speed in m/s
        """
        if self.prediction_model is None or 'model' not in self.prediction_model:
            logger.error("No prediction model available")
            return None
        
        try:
            # Prepare input features
            feature_cols = self.prediction_model['feature_cols']
            
            # Check if we have enough positions
            max_lag = 0
            for col in feature_cols:
                if '_lag' in col:
                    lag = int(col.split('_lag')[1])
                    max_lag = max(max_lag, lag)
            
            if len(positions) <= max_lag:
                logger.error(f"Need at least {max_lag+1} positions for prediction, got {len(positions)}")
                return None
            
            # Create feature vector
            features = []
            for col in feature_cols:
                if col.startswith('center_x'):
                    lag = 0
                    if '_lag' in col:
                        lag = int(col.split('_lag')[1])
                    if lag < len(positions):
                        features.append(positions[lag][0])
                    else:
                        logger.error(f"Not enough positions for lag {lag}")
                        return None
                elif col.startswith('center_y'):
                    lag = 0
                    if '_lag' in col:
                        lag = int(col.split('_lag')[1])
                    if lag < len(positions):
                        features.append(positions[lag][1])
                    else:
                        logger.error(f"Not enough positions for lag {lag}")
                        return None
                elif col == 'time':
                    # Use a default time value
                    features.append(0.0)
                else:
                    # For any other columns, use a default value of 0
                    features.append(0.0)
            
            # Make prediction
            prediction = self.prediction_model['model'].predict([features])[0]
            
            logger.info(f"Predicted speed: {prediction:.2f} m/s")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting speed: {str(e)}")
            return None
    
    def plot_clusters(self, output_path=None):
        """
        Plot the clustering results.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.clusters is None or 'labels' not in self.clusters:
            logger.error("No clustering results available")
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Get data for plotting
            feature_names = self.clusters['feature_names']
            if len(feature_names) < 2:
                # If only one feature was used for clustering, use frame number as second dimension
                if 'frame' in self.speed_data.columns:
                    x_col = feature_names[0]
                    y_col = 'frame'
                    x_label = x_col
                    y_label = 'Frame Number'
                else:
                    logger.error("Need at least two dimensions for cluster plotting")
                    return None
            else:
                x_col = feature_names[0]
                y_col = feature_names[1]
                
                # Set axis labels based on column names
                if x_col == 'smooth_speed_mps' or x_col == 'speed_mps':
                    x_label = 'Speed (m/s)'
                elif x_col == 'smooth_acceleration_mps2' or x_col == 'acceleration_mps2':
                    x_label = 'Acceleration (m/s²)'
                else:
                    x_label = x_col
                    
                if y_col == 'smooth_speed_mps' or y_col == 'speed_mps':
                    y_label = 'Speed (m/s)'
                elif y_col == 'smooth_acceleration_mps2' or y_col == 'acceleration_mps2':
                    y_label = 'Acceleration (m/s²)'
                else:
                    y_label = y_col
            
            # Get data points with valid cluster assignments
            valid_data = self.speed_data.dropna(subset=['cluster'])
            
            # Plot data points colored by cluster
            scatter = plt.scatter(valid_data[x_col], valid_data[y_col], 
                                 c=valid_data['cluster'], cmap='viridis', alpha=0.7)
            
            # Plot cluster centers if we have the same dimensions
            centers = self.clusters['centers']
            if centers.shape[1] >= 2:
                plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
                
                # Add labels for cluster centers
                for i, center in enumerate(centers):
                    plt.annotate(f'Cluster {i}', (center[0], center[1]), 
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=12, fontweight='bold')
            
            # Add labels and title
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.title('Glove Movement Clusters', fontsize=14)
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"clusters_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Cluster plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting clusters: {str(e)}")
            return None
    
    def plot_pca(self, output_path=None):
        """
        Plot the PCA results.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.pca_result is None or 'result' not in self.pca_result:
            logger.error("No PCA results available")
            return None
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Create a subplot for the scatter plot
            plt.subplot(2, 1, 1)
            
            # Plot PCA results
            pca_result = self.pca_result['result']
            
            # Color points by time if available
            if 'time' in self.speed_data.columns:
                # Get time values for rows used in PCA
                valid_indices = self.speed_data[self.pca_result['feature_names']].dropna().index
                time_values = self.speed_data.loc[valid_indices, 'time'].values
                
                if len(time_values) == len(pca_result):
                    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                         c=time_values, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, label='Time (s)')
                else:
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            else:
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            
            # Add labels and title
            plt.xlabel(f'Principal Component 1 ({self.pca_result["explained_variance"][0]*100:.1f}%)', fontsize=12)
            plt.ylabel(f'Principal Component 2 ({self.pca_result["explained_variance"][1]*100:.1f}%)', fontsize=12)
            plt.title('PCA of Glove Movement Data', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Create a subplot for the feature contributions
            plt.subplot(2, 1, 2)
            
            # Plot feature contributions to principal components
            components = self.pca_result['components']
            feature_names = self.pca_result['feature_names']
            
            # Create a bar chart
            x = np.arange(len(feature_names))
            width = 0.35
            
            plt.bar(x - width/2, components[0], width, label='PC1')
            if components.shape[0] > 1:  # Make sure we have at least 2 components
                plt.bar(x + width/2, components[1], width, label='PC2')
            
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Contribution', fontsize=12)
            plt.title('Feature Contributions to Principal Components', fontsize=14)
            plt.xticks(x, feature_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"pca_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"PCA plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting PCA: {str(e)}")
            return None
    
    def plot_speed_prediction(self, output_path=None):
        """
        Plot the speed prediction results.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.prediction_model is None or 'model' not in self.prediction_model:
            logger.error("No prediction model available")
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Get data for plotting
            data = self.speed_data.copy()
            
            # Create lag features (previous positions)
            feature_cols = self.prediction_model['feature_cols']
            lag_cols = [col for col in feature_cols if '_lag' in col]
            
            if lag_cols:
                # Extract base column names and maximum lag
                base_cols = set()
                max_lag = 0
                for col in lag_cols:
                    base_col = col.split('_lag')[0]
                    lag = int(col.split('_lag')[1])
                    base_cols.add(base_col)
                    max_lag = max(max_lag, lag)
                
                # Create lag features
                for lag in range(1, max_lag + 1):
                    for base_col in base_cols:
                        if base_col in data.columns:
                            data[f'{base_col}_lag{lag}'] = data[base_col].shift(lag)
            
            # Drop rows with NaN values (from lag creation)
            data = data.dropna(subset=feature_cols)
            
            if len(data) < 5:
                logger.error(f"Insufficient data for prediction plot: {len(data)} rows")
                return None
            
            # Define features and target
            target_col = self.prediction_model['target_col']
            X = data[feature_cols].values
            y_true = data[target_col].values
            
            # Make predictions
            y_pred = self.prediction_model['model'].predict(X)
            
            # Plot actual vs predicted speeds
            plt.scatter(y_true, y_pred, alpha=0.7)
            
            # Add perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add labels and title
            plt.xlabel('Actual Speed (m/s)', fontsize=12)
            plt.ylabel('Predicted Speed (m/s)', fontsize=12)
            plt.title('Speed Prediction Model Performance', fontsize=14)
            
            # Add performance metrics
            r2 = self.prediction_model['r2_score']
            mse = self.prediction_model['mse']
            plt.annotate(f'R² = {r2:.4f}\nMSE = {mse:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"speed_prediction_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Speed prediction plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting speed prediction: {str(e)}")
            return None
    
    def generate_analysis_report(self, output_path=None):
        """
        Generate a comprehensive data analysis report.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report or None if generation failed
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        try:
            # Extract features if not already done
            if self.features is None:
                self.extract_features()
            
            # Perform clustering if not already done
            if self.clusters is None:
                self.cluster_movements()
            
            # Perform PCA if not already done
            if self.pca_result is None:
                self.perform_pca()
            
            # Train prediction model if not already done
            if self.prediction_model is None:
                self.train_speed_prediction_model()
            
            # Generate plots
            cluster_plot = None
            if self.clusters is not None and 'labels' in self.clusters:
                cluster_plot = self.plot_clusters()
                
            pca_plot = None
            if self.pca_result is not None and 'result' in self.pca_result:
                pca_plot = self.plot_pca()
                
            prediction_plot = None
            if self.prediction_model is not None and 'model' in self.prediction_model:
                prediction_plot = self.plot_speed_prediction()
            
            # Create report path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"analysis_report_{timestamp}.html")
            
            # Generate HTML report
            with open(output_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Glove Movement Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #2c3e50; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .section {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .plot {{ margin-bottom: 30px; }}
                        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Glove Movement Analysis Report</h1>
                        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        
                        <div class="section">
                            <h2>Movement Features</h2>
                """)
                
                # Add feature table if available
                if self.features is not None and not self.features.empty:
                    f.write("""
                            <table>
                                <tr>
                                    <th>Feature</th>
                                    <th>Value</th>
                                </tr>
                    """)
                    
                    # Add feature rows
                    for col in self.features.columns:
                        f.write(f"""
                                <tr>
                                    <td>{col}</td>
                                    <td>{self.features[col].iloc[0]:.4f}</td>
                                </tr>
                        """)
                    
                    f.write("""
                            </table>
                    """)
                else:
                    f.write("<p>No feature data available.</p>")
                
                f.write("""
                        </div>
                        
                        <div class="section">
                            <h2>Movement Pattern Clustering</h2>
                """)
                
                if cluster_plot:
                    f.write(f"""
                            <div class="plot">
                                <img src="{os.path.basename(cluster_plot)}" alt="Movement Clusters">
                                <p>The plot above shows the clustering of glove movements based on speed and acceleration.</p>
                            </div>
                    """)
                
                if self.clusters and 'centers' in self.clusters and 'feature_names' in self.clusters:
                    f.write("""
                            <h3>Cluster Centers</h3>
                            <table>
                                <tr>
                                    <th>Cluster</th>
                    """)
                    
                    # Add column headers for each feature
                    for feature in self.clusters['feature_names']:
                        f.write(f"<th>{feature}</th>")
                    
                    f.write("<th>Interpretation</th></tr>")
                    
                    # Add cluster center rows
                    for i, center in enumerate(self.clusters['centers']):
                        f.write(f"<tr><td>Cluster {i}</td>")
                        
                        # Add values for each feature
                        for j, feature in enumerate(self.clusters['feature_names']):
                            if j < len(center):
                                f.write(f"<td>{center[j]:.2f}</td>")
                            else:
                                f.write("<td>N/A</td>")
                        
                        # Add interpretation
                        interpretation = "Movement pattern "
                        if len(self.clusters['feature_names']) > 0 and len(center) > 0:
                            feature = self.clusters['feature_names'][0]
                            value = center[0]
                            
                            if 'speed' in feature.lower():
                                if value < 1.0:
                                    interpretation += "with low speed"
                                elif value < 3.0:
                                    interpretation += "with medium speed"
                                else:
                                    interpretation += "with high speed"
                            
                            if len(self.clusters['feature_names']) > 1 and len(center) > 1:
                                feature = self.clusters['feature_names'][1]
                                value = center[1]
                                
                                if 'acceleration' in feature.lower():
                                    if abs(value) < 1.0:
                                        interpretation += ", steady movement"
                                    elif value > 1.0:
                                        interpretation += ", accelerating"
                                    else:
                                        interpretation += ", decelerating"
                        
                        f.write(f"<td>{interpretation}</td></tr>")
                    
                    f.write("""
                            </table>
                    """)
                else:
                    f.write("<p>No clustering data available.</p>")
                
                f.write("""
                        </div>
                        
                        <div class="section">
                            <h2>Principal Component Analysis</h2>
                """)
                
                if pca_plot:
                    f.write(f"""
                            <div class="plot">
                                <img src="{os.path.basename(pca_plot)}" alt="PCA Results">
                                <p>The plot above shows the principal component analysis of the glove movement data.</p>
                            </div>
                    """)
                
                if self.pca_result and 'explained_variance' in self.pca_result:
                    f.write(f"""
                            <p>The first two principal components explain {sum(self.pca_result['explained_variance'][:2])*100:.2f}% of the variance in the data.</p>
                    """)
                else:
                    f.write("<p>No PCA data available.</p>")
                
                f.write("""
                        </div>
                        
                        <div class="section">
                            <h2>Speed Prediction Model</h2>
                """)
                
                if prediction_plot:
                    f.write(f"""
                            <div class="plot">
                                <img src="{os.path.basename(prediction_plot)}" alt="Speed Prediction">
                                <p>The plot above shows the performance of the speed prediction model.</p>
                            </div>
                    """)
                
                if self.prediction_model and 'r2_score' in self.prediction_model:
                    f.write(f"""
                            <p><strong>Model Performance:</strong></p>
                            <ul>
                                <li>R² Score: {self.prediction_model['r2_score']:.4f}</li>
                                <li>Mean Squared Error: {self.prediction_model['mse']:.4f}</li>
                            </ul>
                            <p>The model uses the following features: {', '.join(self.prediction_model['feature_cols'])}</p>
                    """)
                else:
                    f.write("<p>No prediction model available.</p>")
                
                f.write("""
                        </div>
                        
                        <div class="section">
                            <h2>Conclusions</h2>
                            <p>Based on the analysis of the glove movement data, the following conclusions can be drawn:</p>
                            <ul>
                """)
                
                # Add conclusions based on analysis results
                if self.features is not None and not self.features.empty:
                    if 'avg_speed' in self.features.columns:
                        avg_speed = self.features['avg_speed'].iloc[0]
                        if avg_speed < 1.0:
                            f.write(f"<li>The average glove speed is relatively low ({avg_speed:.2f} m/s), suggesting controlled, precise movements.</li>")
                        elif avg_speed < 3.0:
                            f.write(f"<li>The average glove speed is moderate ({avg_speed:.2f} m/s), indicating a balance between control and quickness.</li>")
                        else:
                            f.write(f"<li>The average glove speed is high ({avg_speed:.2f} m/s), suggesting quick, reactive movements.</li>")
                    
                    if 'movement_efficiency' in self.features.columns:
                        efficiency = self.features['movement_efficiency'].iloc[0]
                        if efficiency < 0.5:
                            f.write(f"<li>The movement efficiency is low ({efficiency:.2f}), indicating a lot of non-linear movement or repositioning.</li>")
                        elif efficiency < 0.8:
                            f.write(f"<li>The movement efficiency is moderate ({efficiency:.2f}), showing a balance between direct and exploratory movements.</li>")
                        else:
                            f.write(f"<li>The movement efficiency is high ({efficiency:.2f}), suggesting direct, purposeful movements.</li>")
                
                if self.clusters and 'labels' in self.clusters:
                    cluster_counts = np.bincount(self.clusters['labels'])
                    dominant_cluster = np.argmax(cluster_counts)
                    dominant_pct = cluster_counts[dominant_cluster] / len(self.clusters['labels']) * 100
                    
                    f.write(f"<li>The glove movement patterns are dominated by Cluster {dominant_cluster} ({dominant_pct:.1f}% of movements).</li>")
                
                if self.prediction_model and 'r2_score' in self.prediction_model:
                    r2 = self.prediction_model['r2_score']
                    if r2 < 0.5:
                        f.write(f"<li>The speed prediction model has limited accuracy (R² = {r2:.2f}), suggesting complex or unpredictable movement patterns.</li>")
                    elif r2 < 0.8:
                        f.write(f"<li>The speed prediction model has moderate accuracy (R² = {r2:.2f}), indicating some predictability in movement patterns.</li>")
                    else:
                        f.write(f"<li>The speed prediction model has high accuracy (R² = {r2:.2f}), suggesting consistent and predictable movement patterns.</li>")
                
                # Add a default conclusion if none of the above are available
                if (self.features is None or self.features.empty) and (self.clusters is None or 'labels' not in self.clusters) and (self.prediction_model is None or 'r2_score' not in self.prediction_model):
                    f.write("<li>The glove movement shows patterns that can be analyzed for performance improvement.</li>")
                    f.write("<li>Further data collection would enhance the analysis capabilities.</li>")
                
                f.write("""
                            </ul>
                        </div>
                    </div>
                </body>
                </html>
                """)
            
            logger.info(f"Analysis report generated and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            return None

def test_data_analyzer():
    """
    Test function for the GloveDataAnalyzer class.
    """
    import glob
    
    # Find the most recent speed data file
    speed_files = glob.glob(os.path.join(OUTPUT_DIR, "speed_data_*.csv"))
    if not speed_files:
        logger.error("No speed data files found")
        return False
    
    # Sort by modification time (newest first)
    speed_file = sorted(speed_files, key=os.path.getmtime, reverse=True)[0]
    
    logger.info(f"Testing GloveDataAnalyzer with speed data: {speed_file}")
    
    # Initialize analyzer
    analyzer = GloveDataAnalyzer()
    
    # Load speed data
    success = analyzer.load_speed_data(speed_file)
    if not success:
        logger.error("Failed to load speed data")
        return False
    
    # Extract features
    features = analyzer.extract_features()
    if features is None:
        logger.warning("Feature extraction returned None, but continuing with test")
    
    # Cluster movements
    cluster_labels, cluster_centers = analyzer.cluster_movements(n_clusters=3)
    if cluster_labels is None:
        logger.warning("Clustering returned None, but continuing with test")
    
    # Perform PCA
    pca_result, explained_variance = analyzer.perform_pca(n_components=2)
    if pca_result is None:
        logger.warning("PCA returned None, but continuing with test")
    
    # Train prediction model
    model, r2, mse = analyzer.train_speed_prediction_model(test_size=0.2)
    if model is None:
        logger.warning("Model training returned None, but continuing with test")
    
    # Generate plots
    cluster_plot = None
    if cluster_labels is not None:
        cluster_plot = analyzer.plot_clusters()
        
    pca_plot = None
    if pca_result is not None:
        pca_plot = analyzer.plot_pca()
        
    prediction_plot = None
    if model is not None:
        prediction_plot = analyzer.plot_speed_prediction()
    
    # Generate report
    report_path = analyzer.generate_analysis_report()
    
    # Save model
    model_path = None
    if model is not None:
        model_path = analyzer.save_prediction_model()
    
    logger.info("GloveDataAnalyzer test completed successfully")
    return {
        "speed_data": speed_file,
        "features": features is not None,
        "clustering": cluster_labels is not None,
        "pca": pca_result is not None,
        "model": model is not None,
        "r2_score": r2 if r2 is not None else None,
        "cluster_plot": cluster_plot,
        "pca_plot": pca_plot,
        "prediction_plot": prediction_plot,
        "report": report_path,
        "model_path": model_path
    }

if __name__ == "__main__":
    test_results = test_data_analyzer()
    
    if test_results:
        print("\nTest Results:")
        print(f"Speed data: {test_results['speed_data']}")
        print(f"Features extracted: {test_results['features']}")
        print(f"Clustering performed: {test_results['clustering']}")
        print(f"PCA performed: {test_results['pca']}")
        print(f"Model trained: {test_results['model']}")
        if test_results['r2_score'] is not None:
            print(f"Model R² score: {test_results['r2_score']:.4f}")
        print(f"Cluster plot: {test_results['cluster_plot']}")
        print(f"PCA plot: {test_results['pca_plot']}")
        print(f"Prediction plot: {test_results['prediction_plot']}")
        print(f"Analysis report: {test_results['report']}")
        print(f"Model saved to: {test_results['model_path']}")
