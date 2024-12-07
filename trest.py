import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
from sklearn.model_selection import cross_val_score

# Example functions for calculating indicators (assuming you have these defined)
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

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * num_std_dev)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * num_std_dev)
    return data

def calculate_sma(data, window=20):
    data['SMA_{}'.format(window)] = data['Close'].rolling(window=window).mean()
    return data

def calculate_macd(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def calculate_ema(data, window=50):
    data['EMA_50'] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift())
    data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift())
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=window, min_periods=1).mean()
    data.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR'], axis=1, inplace=True)
    return data

def calculate_stochastic_oscillator(data, window=14):
    data['Lowest_Low'] = data['Low'].rolling(window=window).min()
    data['Highest_High'] = data['High'].rolling(window=window).max()
    data['%K'] = (data['Close'] - data['Lowest_Low']) * 100 / (data['Highest_High'] - data['Lowest_Low'])
    data['%D'] = data['%K'].rolling(window=3).mean()
    data.drop(['Lowest_Low', 'Highest_High'], axis=1, inplace=True)
    return data

# Load and prepare data
df = pd.read_csv('https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/ethereum.csv')  # Replace with your data source
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Reverse the rows
df = df.iloc[::-1].copy()

# Calculate indicators
df = calculate_bollinger_bands(df)
df = calculateRSI(df)
df = calculate_sma(df)
df = calculate_macd(df)
df = calculate_ema(df, window=50)   # 50-period EMA
df = calculate_atr(df)
df = calculate_stochastic_oscillator(df)

# Verify columns
expected_columns = ['RSI', 'Bollinger_High', 'Bollinger_Low', 'EMA_50', 'ATR', '%K', '%D']
for col in expected_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from the DataFrame.")

# Drop rows with NaN values created by rolling windows
df.dropna(subset=expected_columns, inplace=True)

# Define mean reversion strategy
def mean_reversion_strategy(data, oversold_threshold=30, overbought_threshold=70):
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['EMA_50'] = data['EMA_50']
    signals['ATR'] = data['ATR']
    signals['%K'] = data['%K']
    signals['%D'] = data['%D']
    signals['RSI'] = data['RSI']
    signals['Bollinger_Low'] = data['Bollinger_Low']
    signals['Bollinger_High'] = data['Bollinger_High']
    
    # Define Buy/Sell conditions based on RSI, EMA, ATR, and Stochastic Oscillator
    signals['Buy_Signal'] = np.where(
        (signals['RSI'] < oversold_threshold) | 
        (signals['%K'] < oversold_threshold) |
        (signals['Price'] < signals['EMA_50']) | 
        (signals['ATR'] < signals['ATR'].rolling(14).mean()) | (signals['Bollinger_High'] > 70.0), 1, 0)
    
    signals['Sell_Signal'] = np.where(
        (signals['RSI'] > overbought_threshold) | 
        (signals['%K'] > overbought_threshold) | 
        (signals['Price'] > signals['EMA_50']) | 
        (signals['ATR'] > signals['ATR'].rolling(14).mean()) | (signals['Bollinger_Low'] < 30.0), -1, 0)
    
    return signals

# Apply mean reversion strategy
signals = mean_reversion_strategy(df)

# Create candlestick chart
fig = go.Figure()

fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlestick'))

# Update layout for candlestick chart
fig.update_layout(title='Ethereum Candlestick Chart with RSI Signals',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Show buy signals
fig_signals = go.Figure()

# Reshape X for model training
X = signals[['RSI', 'Bollinger_High', 'Bollinger_Low', 'EMA_50', 'ATR', '%K', '%D']].values
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

# Backtesting function based on signals
def backtest_strategy(data, signals, initial_balance=100000):
    balance = initial_balance
    btc_balance = 0
    trade_log = []

    for i in range(1, len(data)):
        if signals['Buy_Signal'].iloc[i] == 1 and balance > 0:
            # Buy signal
            btc_balance = balance / data['Close'].iloc[i]
            balance = 0
            trade_log.append(('Buy', data.index[i], data['Close'].iloc[i], btc_balance))
        elif signals['Sell_Signal'].iloc[i] == -1 and btc_balance > 0:
            # Sell signal
            balance = btc_balance * data['Close'].iloc[i]
            btc_balance = 0
            trade_log.append(('Sell', data.index[i], data['Close'].iloc[i], balance))

    # Calculate final balance
    final_balance = balance + (btc_balance * data['Close'].iloc[-1])
    return trade_log, final_balance

# Example usage
trade_log, final_balance = backtest_strategy(df, signals, initial_balance=100000)
print(f'Final Balance: {final_balance:.2f}')
for trade in trade_log:
    print(trade)

def evaluate_performance(trade_log, initial_balance=100000):
    balance_over_time = [initial_balance]
    
    for trade in trade_log:
        if trade[0] == 'Buy':
            balance_over_time.append(balance_over_time[-1])
        elif trade[0] == 'Sell':
            balance_over_time.append(trade[3])
    
    balance_over_time = np.array(balance_over_time)
    total_return = (balance_over_time[-1] / initial_balance) - 1
    
    # Calculate returns
    returns = np.diff(balance_over_time) / balance_over_time[:-1]
    
    # Handle case where returns array is empty
    if len(returns) == 0:
        sharpe_ratio = np.nan
    else:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else np.nan

    # Calculate max drawdown
    drawdowns = np.maximum.accumulate(balance_over_time) - balance_over_time
    max_drawdown = np.max(drawdowns) / np.maximum.accumulate(balance_over_time)[-1]

    return total_return, sharpe_ratio, max_drawdown

# Evaluate performance
total_return, sharpe_ratio, max_drawdown = evaluate_performance(trade_log)
print(f'Total Return: {total_return:.2%}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Max Drawdown: {max_drawdown:.2%}')

