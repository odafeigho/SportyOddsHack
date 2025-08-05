# SportyOdds Hack - 3-way Soccer Result Predictor
# Author: Odafeigho

import importlib
import subprocess
import sys
import time
import math
import warnings
import os
import pandas as pd
from datetime import datetime

# Disable PyTensor C compilation to suppress g++ warnings
os.environ["PYTENSOR_FLAGS"] = "cxx="

# Define install_dependencies
def install_dependencies():
    """
    Check for required modules and install them if missing.
    """
    required_modules = ['numpy', 'pandas', 'pymc', 'scikit-learn', 'arviz', 'openpyxl', 'tqdm']
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"{module} is already installed.")
        except ImportError:
            print(f"Installing {module}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"{module} installed successfully.")
            except subprocess.CalledProcessError:
                print(f"Failed to install {module}. Please install it manually.")
                sys.exit(1)

# Call install_dependencies to ensure all required modules are installed
install_dependencies()

# Import non-standard libraries
import numpy as np
import pymc as pm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Manual data input module
def manual_data_input():
    """
    Prompt user to manually input match data and return a DataFrame.
    Returns:
        pd.DataFrame: DataFrame with user-entered match data.
    """
    print("\nManual Data Input Mode")
    print("Enter match data. Type 'done' when finished.")
    required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 
                    'home_odds', 'draw_odds', 'away_odds', 'home_injuries', 'away_injuries', 
                    'home_suspensions', 'away_suspensions', 'head_to_head_wins_home', 'head_to_head_wins_away']
    data = []
    
    while True:
        row = {}
        try:
            # Prompt for match data
            user_input = input("Enter match data or 'done' to finish: ").strip().lower()
            if user_input == 'done':
                if not data:
                    print("Error: No data entered. At least one match is required.")
                    sys.exit(1)
                break
            
            # Date
            date_str = input("Enter match date (YYYY-MM-DD): ").strip()
            try:
                row['date'] = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print("Error: Invalid date format. Use YYYY-MM-DD.")
                continue
            
            # Team names
            row['home_team'] = input("Enter home team name: ").strip()
            row['away_team'] = input("Enter away team name: ").strip()
            if not row['home_team'] or not row['away_team']:
                print("Error: Team names cannot be empty.")
                continue
            
            # Numeric inputs
            for col in ['home_goals', 'away_goals', 'home_odds', 'draw_odds', 'away_odds', 
                        'home_injuries', 'away_injuries', 'home_suspensions', 'away_suspensions', 
                        'head_to_head_wins_home', 'head_to_head_wins_away']:
                while True:
                    try:
                        value = float(input(f"Enter {col.replace('_', ' ')} (numeric, non-negative): ").strip())
                        if value < 0:
                            print(f"Error: {col.replace('_', ' ')} cannot be negative.")
                            continue
                        row[col] = value
                        break
                    except ValueError:
                        print(f"Error: {col.replace('_', ' ')} must be a valid number.")
            
            data.append(row)
        
        except KeyboardInterrupt:
            print("\nInput interrupted. Exiting.")
            sys.exit(1)
    
    df = pd.DataFrame(data, columns=required_cols)
    print("Manual input completed. DataFrame created with", len(df), "matches.")
    return df

# Load and preprocess dataset
def load_and_preprocess_data(file_path, n_games=20):
    """
    Load dataset from CSV/Excel or manual input and engineer features.
    Args:
        file_path (str): Path to CSV or Excel file with match data.
        n_games (int): Number of previous games to consider (10 or 20).
    Returns:
        pd.DataFrame: Processed dataset with features and target.
    """
    required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 
                    'home_odds', 'draw_odds', 'away_odds', 'home_injuries', 'away_injuries', 
                    'home_suspensions', 'away_suspensions', 'head_to_head_wins_home', 'head_to_head_wins_away']
    
    # Try loading from file
    df = None
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, parse_dates=['date'])
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path, parse_dates=['date'])
            else:
                print(f"Error: Unsupported file format for {file_path}. Use CSV or Excel.")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except pd.errors.ParserError:
            print(f"Error: Unable to parse {file_path}. Ensure it is a valid CSV or Excel file.")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    # If file loading fails or no file provided, prompt for manual input
    if df is None:
        print("Falling back to manual data input.")
        df = manual_data_input()
    
    # Validate required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Drop rows with NaN in critical columns
    critical_cols = ['home_goals', 'away_goals', 'home_odds', 'draw_odds', 'away_odds']
    df = df.dropna(subset=critical_cols)
    
    # Ensure odds and numeric columns are valid
    for col in ['home_goals', 'away_goals', 'home_odds', 'draw_odds', 'away_odds', 
                'home_injuries', 'away_injuries', 'home_suspensions', 'away_suspensions', 
                'head_to_head_wins_home', 'head_to_head_wins_away']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['home_odds'] > 0) & (df['draw_odds'] > 0) & (df['away_odds'] > 0)]
    
    # Check if dataset is empty after cleaning
    if df.empty:
        print("Error: Dataset is empty after cleaning critical columns. Please check the data.")
        sys.exit(1)
    
    # Create target variable: 0 = Away Win, 1 = Draw, 2 = Home Win
    df['result'] = np.where(df['home_goals'] > df['away_goals'], 2, 
                           np.where(df['home_goals'] == df['away_goals'], 1, 0))
    
    # Feature engineering: Compute average stats for last n_games
    features = []
    for team in set(df['home_team']).union(set(df['away_team'])):
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
        team_matches['is_home'] = team_matches['home_team'] == team
        team_matches['goals_scored'] = np.where(team_matches['is_home'], 
                                               team_matches['home_goals'], team_matches['away_goals'])
        team_matches['goals_conceded'] = np.where(team_matches['is_home'], 
                                                 team_matches['away_goals'], team_matches['home_goals'])
        team_matches['win'] = np.where(team_matches['is_home'], 
                                      team_matches['result'] == 2, team_matches['result'] == 0)
        
        # Compute rolling averages for last n_games
        team_matches[f'avg_goals_scored_{n_games}'] = team_matches['goals_scored'].rolling(n_games, min_periods=1).mean().shift(1).fillna(0)
        team_matches[f'avg_goals_conceded_{n_games}'] = team_matches['goals_conceded'].rolling(n_games, min_periods=1).mean().shift(1).fillna(0)
        team_matches[f'win_rate_{n_games}'] = team_matches['win'].rolling(n_games, min_periods=1).mean().shift(1).fillna(0)
        
        features.append(team_matches[['date', 'home_team', 'away_team', f'avg_goals_scored_{n_games}', 
                                     f'avg_goals_conceded_{n_games}', f'win_rate_{n_games}']])
    
    # Merge team features back to main dataframe
    features_df = pd.concat(features).sort_values('date')
    df = df.merge(features_df, on=['date', 'home_team', 'away_team'], how='left')
    
    # Validate merge
    if df[[f'avg_goals_scored_{n_games}', f'avg_goals_conceded_{n_games}', f'win_rate_{n_games}']].isna().any().any():
        print("Warning: Some matches have missing features after merge. Dropping affected rows.")
        df = df.dropna(subset=[f'avg_goals_scored_{n_games}', f'avg_goals_conceded_{n_games}', f'win_rate_{n_games}'])
    
    # Convert bookmaker odds to implied probabilities
    df['home_prob'] = 1 / df['home_odds']
    df['draw_prob'] = 1 / df['draw_odds']
    df['away_prob'] = 1 / df['away_odds']
    
    # Normalize probabilities to sum to 1, handle invalid odds
    total_prob = df['home_prob'] + df['draw_prob'] + df['away_prob']
    invalid_odds = total_prob == 0
    if invalid_odds.any():
        print("Warning: Invalid bookmaker odds detected. Setting uniform probabilities for affected rows.")
        df.loc[invalid_odds, ['home_prob', 'draw_prob', 'away_prob']] = 1/3
    df.loc[~invalid_odds, 'home_prob'] /= total_prob[~invalid_odds]
    df.loc[~invalid_odds, 'draw_prob'] /= total_prob[~invalid_odds]
    df.loc[~invalid_odds, 'away_prob'] /= total_prob[~invalid_odds]
    
    return df

def generate_polynomial_features(X, degree=1):
    """
    Generate polynomial features from input data.
    Args:
        X (pd.DataFrame): Input features.
        degree (int): Degree of polynomial features (reduced to 1 for speed).
    Returns:
        np.ndarray: Polynomial features.
    """
    if X.shape[1] * (X.shape[1] + 1) / 2 > 1000:  # Warn for large feature sets
        print("Warning: Large number of polynomial features may slow down computation.")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

def bayesian_model(X_poly, y, n_classes=3):
    """
    Build and sample from a Bayesian model with Dirichlet prior for multinomial probabilities.
    Args:
        X_poly (np.ndarray): Polynomial features.
        y (np.ndarray): Target variable (0, 1, 2).
        n_classes (int): Number of outcome classes (3 for Win/Draw/Loss).
    Returns:
        pm.Model: Fitted Bayesian model.
    """
    with pm.Model() as model:
        # Dirichlet prior for class probabilities
        alpha = np.ones(n_classes)  # Uniform prior
        theta = pm.Dirichlet('theta', a=alpha, shape=n_classes)
        
        # Coefficients for polynomial features
        beta = pm.Normal('beta', mu=0, sigma=10, shape=(X_poly.shape[1], n_classes))
        
        # Linear combination of features
        logits = pm.math.dot(X_poly, beta)
        
        # Combine Dirichlet prior with logits via softmax
        p = pm.math.softmax(logits + pm.math.log(theta), axis=1)
        
        # Multinomial likelihood
        y_obs = pm.Categorical('y_obs', p=p, observed=y)
        
        # Sample from posterior with fixed seed for reproducibility
        try:
            trace = pm.sample(500, tune=500, chains=2, cores=1, random_seed=42, 
                             return_inferencedata=False, progressbar=True)
        except Exception as e:
            print(f"Error during sampling: {e}")
            sys.exit(1)
    
    return model, trace

def predict_outcomes(model, trace, X_poly):
    """
    Predict match outcomes using the Bayesian model.
    Args:
        model (pm.Model): Bayesian model.
        trace: Posterior samples.
        X_poly (np.ndarray): Polynomial features for prediction.
    Returns:
        np.ndarray: Predicted class probabilities.
    """
    with model:
        beta = trace['beta'].mean(axis=0)
        theta = trace['theta'].mean(axis=0)
        logits = np.dot(X_poly, beta)
        probs = np.exp(logits + np.log(theta)) / np.sum(np.exp(logits + np.log(theta)), axis=1, keepdims=True)
    return probs

def main(file_path='football_data.csv', n_games=10, skip_animation=False):
    """
    Main function to run the SportyOdds Hack soccer result predictor.
    Args:
        file_path (str): Path to dataset (CSV or Excel), or empty string for manual input.
        n_games (int): Number of previous games to consider (reduced to 10 for speed).
        skip_animation (bool): If True, skip the opening animation.
    """
    if not skip_animation:
        display_opening_animation()
    else:
        print("\nWelcome to SportyOdds Hack by Odafeigho!")
        print("Starting program...\n")
    
    # Prompt for file path if default fails
    if file_path:
        print(f"Attempting to load data from {file_path}")
    else:
        file_path = input("Enter path to CSV or Excel file (or press Enter for manual input): ").strip()
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_data(file_path, n_games)
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        sys.exit(1)
    
    # Check dataset size
    if len(df) > 5000:
        print("Warning: Large dataset detected (>5,000 rows). Subsampling to 5,000 rows for faster computation.")
        df = df.sample(5000, random_state=42)
    
    # Define feature columns
    feature_cols = [f'avg_goals_scored_{n_games}', f'avg_goals_conceded_{n_games}', f'win_rate_{n_games}', 
                    'home_injuries', 'away_injuries', 'home_suspensions', 'away_suspensions', 
                    'head_to_head_wins_home', 'head_to_head_wins_away', 'home_prob', 'draw_prob', 'away_prob']
    
    # Drop rows with NaN in feature columns
    df = df.dropna(subset=feature_cols)
    
    # Check if dataset is empty after cleaning
    if df.empty:
        print("Error: Dataset is empty after cleaning feature columns. Please check the data.")
        sys.exit(1)
    
    X = df[feature_cols]
    y = df['result']
    
    # Scale features
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        sys.exit(1)
    
    # Generate polynomial features
    try:
        X_poly = generate_polynomial_features(X_scaled, degree=1)
    except Exception as e:
        print(f"Error during polynomial feature generation: {e}")
        sys.exit(1)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error during data splitting: {e}")
        sys.exit(1)
    
    # Train Bayesian model
    print("Training Bayesian model (this may take a moment)...")
    try:
        model, trace = bayesian_model(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)
    
    # Predict on test set
    try:
        y_pred_probs = predict_outcomes(model, trace, X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    # Evaluate model
    try:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1-Score: {f1:.2f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    # For new predictions (example)
    # new_data = pd.DataFrame({...})  # Add new match data
    # new_X = scaler.transform(new_data[feature_cols])
    # new_X_poly = generate_polynomial_features(new_X)
    # predictions = predict_outcomes(model, trace, new_X_poly)
    # print("Predicted Probabilities (Away, Draw, Home):", predictions)

# Call main if this is the main module
if __name__ == "__main__":
    # Allow user to skip animation for faster startup
    skip_animation = input("Skip opening animation? (y/n): ").strip().lower() == 'y'
    main(skip_animation=skip_animation)