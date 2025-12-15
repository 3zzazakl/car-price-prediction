# Car Price Prediction (CarCast) - ML Pipeline

A complete machine learning pipeline for predicting car prices using multiple regression models.

## Ai2 Project

**Project Title:** Car Price Prediction Pipeline
**Objective:** Develop a robust regression model to estimate car prices using vehicle features.
**Methodology:** The project implements a full ML pipeline including data cleaning, feature engineering (Target/One-Hot encoding), and a comparative analysis of 4 algorithms (Random Forest, XGBoost, Gradient Boosting, Ridge).
**Outcome:** A trained model capable of predicting prices with evaluated accuracy metrics (RMSE, R²).

## Overview

This project implements a comprehensive ML workflow that:
- Preprocesses and cleans automotive data
- Engineers features from raw inputs
- Applies multiple encoding strategies for categorical variables
- Trains and compares 4 different regression models
- Provides detailed evaluation metrics and visualizations

## Models Implemented

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of decision trees with bagging |
| **Gradient Boosting** | Sequential ensemble with boosting |
| **XGBoost** | Optimized gradient boosting implementation |
| **Ridge Regression** | Linear regression with L2 regularization |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Dataset

The project requires a CSV file named `car_price_prediction.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| ID | int | Unique identifier |
| Price | int | Target variable (USD) |
| Levy | str | Tax levy ('-' or numeric) |
| Manufacturer | str | Car manufacturer |
| Model | str | Car model |
| Prod. year | int | Production year |
| Category | str | Vehicle category |
| Leather interior | str | Yes/No |
| Fuel type | str | Fuel type |
| Engine volume | str | Engine size (e.g., "2.0 Turbo") |
| Mileage | str | Mileage with units (e.g., "100000 km") |
| Cylinders | float | Number of cylinders |
| Gear box type | str | Transmission type |
| Drive wheels | str | Drive type (FWD/RWD/AWD) |
| Doors | str | Number of doors |
| Wheel | str | Steering wheel position |
| Color | str | Car color |
| Airbags | int | Number of airbags |

## Usage

1. Place your `car_price_prediction.csv` file in the project directory
2. Open `car_price_prediction.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially

```bash
# Or run via command line
jupyter notebook car_price_prediction.ipynb
```

## Project Structure

```
car_price_prediction/
├── car_price_prediction.ipynb    # Main notebook
├── car_price_prediction.csv      # Dataset (not included)
└── README.md                     # This file
```

## Pipeline Steps

### 1. Data Preprocessing
- Clean `Levy` column (convert '-' to 0)
- Extract numeric values from `Engine volume` (remove "Turbo")
- Parse `Mileage` (remove "km" suffix)
- Calculate vehicle `Age` from production year
- Remove duplicates and outliers (IQR method)

### 2. Feature Encoding
- **One-Hot Encoding**: Leather interior, Gear box type, Drive wheels, Wheel
- **Target Encoding**: Fuel type, Model, Airbags, Cylinders, Manufacturer
- **Label Encoding**: Category, Color

### 3. Model Training
- 80/20 train-test split
- StandardScaler for numerical features
- Hyperparameters configurable via `Config` class

### 4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## Configuration

Modify the `Config` class in the notebook to customize:

```python
class Config:
    TARGET = 'Price'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    OUTLIER_REMOVAL = True
    IQR_MULTIPLIER = 1.5
```

## Results

After running the notebook, you'll see:
- Model comparison table sorted by RMSE
- Bar charts comparing RMSE, MAE, and R² across models
- Actual vs. Predicted scatter plot for the best model

## License

This project is open source and available for educational purposes.
