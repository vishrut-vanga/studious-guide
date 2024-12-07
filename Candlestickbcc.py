import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score




from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/bitcoin.csv')

# Reverse rows to have oldest first for RSI calculation
df_rev_rows = df.iloc[::-1].reset_index(drop=True)

# Filter data for the years 2013 to 2020
df_rev_rows['Date'] = pd.to_datetime(df_rev_rows['Date'])

# Function to calculate RSI
def calculateRSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    data['RSI'] = RSI
    return data




# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * num_std_dev)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * num_std_dev)
    return data

# Apply Bollinger Bands calculation
df_rev_rows = calculate_bollinger_bands(df_rev_rows)
df_rev_rows.dropna(subset=['Bollinger_High', 'Bollinger_Low'], inplace=True)



# Apply RSI calculation
df_rev_rows = calculateRSI(df_rev_rows)

# Remove NaN values from RSI column
df_rev_rows.dropna(subset=['RSI'], inplace=True)




def mean_reversion_strategy(data, oversold_threshold=30, overbought_threshold=70):
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['RSI'] = data['RSI']
    signals['Bollinger_High'] = data['Bollinger_High']
    signals['Bollinger_Low'] = data['Bollinger_Low']
    signals['Buy_Signal'] = np.where((signals['RSI'] < oversold_threshold) | (signals['Price'] < signals['Bollinger_Low']), 1, 0)
    signals['Sell_Signal'] = np.where((signals['RSI'] > overbought_threshold) | (signals['Price'] > signals['Bollinger_High']), -1, 0)
    return signals

# Apply mean reversion strategy
signals = mean_reversion_strategy(df_rev_rows)


# Create candlestick chart
fig = go.Figure()

fig.add_trace(go.Candlestick(x=df_rev_rows['Date'],
                             open=df_rev_rows['Open'],
                             high=df_rev_rows['High'],
                             low=df_rev_rows['Low'],
                             close=df_rev_rows['Close'],
                             name='Candlestick'))

# Update layout for candlestick chart
fig.update_layout(title='Bitcoin Candlestick Chart with RSI Signals (2013-2020)',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Show buy signals
fig_signals = go.Figure()

# Buy signals
buy_signals = signals[signals['Buy_Signal'] == 1]
fig_signals.add_trace(go.Scatter(x=df_rev_rows.loc[buy_signals.index, 'Date'],
                                 y=buy_signals['Price'],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up',
                                             size=10,
                                             color='green',
                                             line=dict(width=1, color='darkgreen')),
                                 name='Buy Signal'))

# Sell signals
sell_signals = signals[signals['Sell_Signal'] == -1]
fig_signals.add_trace(go.Scatter(x=df_rev_rows.loc[sell_signals.index, 'Date'],
                                 y=sell_signals['Price'],
                                 mode='markers',
                                 marker=dict(symbol='triangle-down',
                                             size=10,
                                             color='red',
                                             line=dict(width=1, color='darkred')),
                                 name='Sell Signal'))

# Update layout for signals chart
fig_signals.update_layout(title='Buy/Sell Signals (2013-2020)',
                          xaxis_title='Date',
                          yaxis_title='Price')

# Show signals plot
fig_signals.show()

# Reshape X for model training
X = signals[['RSI', 'Bollinger_High', 'Bollinger_Low']].values
y = signals['Buy_Signal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train GradientBoostingRegressor model
gbr = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_gbr = grid_search.best_estimator_

# Make predictions
y_gbr_train_pred = best_gbr.predict(X_train)
y_gbr_test_pred = best_gbr.predict(X_test)

# Evaluate model performance
gbr_train_mse = mean_squared_error(y_train, y_gbr_train_pred)
gbr_train_r2 = r2_score(y_train, y_gbr_train_pred)

gbr_test_mse = mean_squared_error(y_test, y_gbr_test_pred)
gbr_test_r2 = r2_score(y_test, y_gbr_test_pred)

print(f'Training MSE: {gbr_train_mse:.2f}, Training R^2: {gbr_train_r2:.2f}')
print(f'Testing MSE: {gbr_test_mse:.2f}, Testing R^2: {gbr_test_r2:.2f}')

# Initialize the model
gbr = GradientBoostingRegressor(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of cross-validation scores
mean_cv_score = -cv_scores.mean()
std_cv_score = cv_scores.std()

print(f'Mean Cross-Validation MSE: {mean_cv_score:.2f}, Std: {std_cv_score:.2f}')

