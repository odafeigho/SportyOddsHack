# SportyOdds Hack - 3-way Soccer Result Predictor
# Author: Odafeigho

import importlib
import subprocess
import sys
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
import time
import math
warnings.filterwarnings('ignore')

# Display opening sinusoidal wave animation
def display_opening_animation():
    """
    Display a text-based sinusoidal wave animation in the console.
    """
    try:
        print("\nWelcome to SportyOdds Hack by Odafeigho!\n")
        height = 10  # Height of the animation window
        width = 50   # Width of the animation window
        frames = 20  # Number of animation frames
        for frame in range(frames):
            # Clear console (works on Unix-like systems and Windows)
            print("\033[H\033[J", end="")  # ANSI clear screen
            for y in range(height, -height, -1):
                row = []
                for x in range(width):
                    # Calculate sine wave with shifting phase
                    wave = int(height * math.sin((x + frame) * 0.2))
                    if y == wave:
                        row.append("*")
                    else:
                        row.append(" ")
                print("".join(row))
            time.sleep(0.1)  # Frame delay
        print("\nStarting SportyOdds Hack...\n")
        time.sleep(1)
    except (OSError, UnicodeEncodeError):
        print("\nWelcome to SportyOdds Hack by Odafeigho!")
        print("Terminal does not support animation. Starting program...\n")
        time.sleep(1)

# Function to check and install dependencies
def install_dependencies():
    """
    Check for required modules and install them if missing.
    """
    required_modules = ['numpy', 'pandas', 'pymc', 'scikit-learn', 'arviz']
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

# Load and preprocess dataset
def load_and_preprocess_data(file_path, n_games=20):
    """
    Load dataset and engineer features for the last n_games, injuries, suspensions, head-to-head, and odds.
    Args:
        file_path (str): Path to CSV file with match data.
        n_games (int): Number of previous games to consider (10 or 20).
    Returns:
        pd.DataFrame: Processed dataset with features and target.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Unable to parse {file_path}. Ensure it is a valid CSV.")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 
                    'home_odds', 'draw_odds', 'away_odds', 'home_injuries', 'away_injuries', 
                    'home_suspensions', 'away_suspensions', 'head_to_head_wins_home', 'head_to_head_wins_away']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
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

# Generate polynomial features
def generate_polynomial_features(X, degree=2):
    """
    Generate polynomial features from input data.
    Args:
        X (pd.DataFrame): Input features.
        degree (int): Degree of polynomial features.
    Returns:
        np.ndarray: Polynomial features.
    """
    if X.shape[1] * (X.shape[1] + 1) / 2 > 1000:  # Warn for large feature sets
        print("Warning: Large number of polynomial features may slow down computation.")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    return poly.fit_transform(X)

# Bayesian model with Dirichlet prior
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
            trace = pm.sample(1000, tune=1000, chains=2, cores=1, random_seed=42, return_inferencedata=False)
        except Exception as e:
            print(f"Error during sampling: {e}")
            sys.exit(1)
    
    return model, trace

# Predict match outcomes
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

# Main function to run the predictor
def main(file_path='football_data.csv', n_games=20):
    """
    Main function to run the SportyOdds Hack soccer result predictor.
    Args:
        file_path (str): Path to dataset.
        n_games (int): Number of previous games to consider.
    """
    # Display opening animation
    display_opening_animation()
    
    # Install dependencies
    install_dependencies()
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_path, n_games)
    
    # Check dataset size
    if len(df) > 10000:
        print("Warning: Large dataset detected (>10,000 rows). Consider subsampling for faster computation.")
    
    # Select features
    feature_cols = [f'avg_goals_scored_{n_games}', f'avg_goals_conceded_{n_games}', f'win_rate_{n_games}', 
                    'home_injuries', 'away_injuries', 'home_suspensions', 'away_suspensions', 
                    'head_to_head_wins_home', 'head_to_head_wins_away', 'home_prob', 'draw_prob', 'away_prob']
    
    # Validate feature columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns: {missing_cols}")
        sys.exit(1)
    
    X = df[feature_cols]
    y = df['result']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Generate polynomial features
    X_poly = generate_polynomial_features(X_scaled, degree=2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Train Bayesian model
    model, trace = bayesian_model(X_train, y_train)
    
    # Predict on test set
    y_pred_probs = predict_outcomes(model, trace, X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # For new predictions (example)
    # new_data = pd.DataFrame({...})  # Add new match data
    # new_X = scaler.transform(new_data[feature_cols])
    # new_X_poly = generate_polynomial_features(new_X)
    # predictions = predict_outcomes(model, trace, new_X_poly)
    # print("Predicted Probabilities (Away, Draw, Home):", predictions)

if __name__ == "__main__":
    main()