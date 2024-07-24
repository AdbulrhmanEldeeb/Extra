import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, HuberRegressor, PassiveAggressiveRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

def evaluate_regressors(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Evaluates multiple regression models on a given dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the performance metrics (MAPE and RMSE) of each regression model.
    """
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Define the regressors
    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),
        "Bayesian Ridge Regression": BayesianRidge(),
        "SVR": SVR(),
        "Linear SVR": LinearSVR(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "AdaBoost Regression": AdaBoostRegressor(),
        "K-Nearest Neighbors Regression": KNeighborsRegressor(),
        "Extra Trees Regression": ExtraTreesRegressor(),
        "SGD Regression": SGDRegressor(),
        "Huber Regression": HuberRegressor(),
        "Passive Aggressive Regression": PassiveAggressiveRegressor(),
        "Theil-Sen Regression": TheilSenRegressor(),
        "RANSAC Regression": RANSACRegressor(),
        "XGBoost Regression": XGBRegressor(),
        "LightGBM Regression": LGBMRegressor(),
        "CatBoost Regression": CatBoostRegressor(verbose=0),
        "MLP": MLPRegressor()
    }
    
    # List to store results
    results = []

    # Evaluate each regressor
    for name, regressor in regressors.items():
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        
        # Calculate performance metrics
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae=mean_absolute_error(y_test,y_pred)
        # Append the results to the list
        results.append({
            "Model": name,
            "MAPE": mape,
            "RMSE": rmse,
            "MAE":mae
        })
    
    # Convert the results list to a DataFrame and return it
    return pd.DataFrame(results)
