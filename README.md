# Microsoft Stock Prediction Project

## Overview
This project aims to predict Microsoft's stock prices using historical stock market data. The project employs machine learning models such as the **Prophet model** and **Long Short-Term Memory (LSTM) neural networks** to forecast future stock prices. It uses key evaluation metrics like **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Directional Accuracy** to measure the performance of the prediction models.

## Project Structure
- **data/**: Contains the historical stock market data used for training and testing the models.
- **notebooks/**: Jupyter notebooks that walk through the data exploration, model training, and evaluation.
- **models/**: Saved models after training, to be used for future predictions.
- **results/**: Folder where prediction outputs, graphs, and performance metrics are saved.
- **README.md**: Project overview and instructions (this file).

## Data
The dataset used includes the following features:
- `Date`: The date of the stock price entry.
- `Open`: Stock opening price for the day.
- `High`: Highest stock price during the trading day.
- `Low`: Lowest stock price during the trading day.
- `Close`: Stock closing price for the day.
- `Volume`: Number of shares traded.
  
## Visualization
- used matplotlib and seaborn to visualize yearly trends.
  
## ADF test
- Stationarity in time series is a fundamental concept that refers to a time series where the statistical properties remain constant over time. The Augmented Dickey Fuller (ADF) test is a statistical test used to check if a time series is stationary.

## Feature Engineering
Several feature engineering techniques were applied to improve model accuracy, including:
- Lag features to capture past stock prices as predictors for future values.
- Moving averages and rolling statistics to smooth out noise in the data.
- Trend and seasonality decomposition to analyze cyclical patterns in stock prices.

## Models Used
### 1. **Prophet Model**
- Prophet is a time series forecasting model developed by Facebook. It is particularly useful for time series with daily observations and multiple seasonalities.
  
### 2. **LSTM (Long Short-Term Memory)**
- LSTM is a type of Recurrent Neural Network (RNN) well-suited for sequential data such as stock prices. It captures long-term dependencies in data and is effective for time-series prediction tasks.

### 3. ARIMA (AutoRegressive Integrated Moving Average)
- A traditional statistical model used for time series forecasting. ARIMA is effective for short-term stock price predictions, especially when dealing with linear trends and seasonality.

### 4. XGBoost
- A machine learning model known for its speed and performance in handling structured data. XGBoost is used here to capture non-linear patterns in the stock prices.

## Installation
### Requirements:
- Python 3.x
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - keras
  - tensorflow
  - fbprophet
  - scikit-learn

## Usage
- **Data Preparation**: Preprocess the stock data, including filling missing values, scaling, and splitting into training and test sets.
- **Model Training**: Train both Prophet and LSTM models on historical stock data.
- **Prediction**: Use the trained models to predict future stock prices.
- **Evaluation**: Evaluate the models using MAE, MSE, and Directional Accuracy.

## Evaluation Metrics
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in a set of predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **Directional Accuracy**: Measures how often the predicted price change direction (up or down) matches the actual price movement.

## Future Work
- Improve model accuracy by fine-tuning hyperparameters.
- Experiment with other models like ARIMA and GRU for comparison.
- Incorporate external data (e.g., news sentiment) to enhance prediction accuracy.

## Contributing
If you'd like to contribute to this project, please create a pull request with a detailed explanation of your changes.

## License
This project is licensed under the MIT License.

