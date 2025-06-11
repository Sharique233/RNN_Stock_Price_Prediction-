#!/usr/bin/env python
# coding: utf-8

# ### Topic: RNN_Stock_Price_Prediction 
#   #### -by Sharique_Seraj

# ### Problem Statement: Predicting stock prices is a challenging task due to the volatile and nonlinear nature of financial markets. However, the sequential patterns present in historical price data make Recurrent Neural Networks (RNNs) a suitable modeling approach. In this assignment, we aim to forecast the closing stock prices of four major technology companies: Amazon (AMZN), Google (GOOGL), IBM, and Microsoft (MSFT) using historical stock market data.

# ### Objective: The objective of this assignment is to try and predict the stock prices using historical data from four companies IBM (IBM), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT).
# ### The goal is to train RNN-based models on historical price data and leverage their ability to capture temporal dependencies for predicting future prices. This can potentially support better investment decisions and financial insights.
# ### We use four different companies because they belong to the same sector: Technology. Using data from all four companies may improve the performance of the model. This way, we can capture the broader market sentiment.

# ### Business Value: Data related to stock markets lends itself well to modeling using RNNs due to its sequential nature. We can keep track of opening prices, closing prices, highest prices, and so on for a long period of time as these values are generated every working day. The patterns observed in this data can then be used to predict the future direction in which stock prices are expected to move. Analyzing this data can be interesting in itself, but it also has a financial incentive as accurate predictions can lead to massive profits.

# In[16]:


# Stock Price Prediction Using RNNs

# 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


# In[17]:


# 2. Define Data Loading and Aggregation Method
file_list = [
    'AMZN_stocks_data.csv',
    'GOOGL_stocks_data.csv',
    'IBM_stocks_data.csv',
    'MSFT_stocks_data.csv'
]
def load_and_aggregate(file_list):
    df_list = []
    for file in file_list:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df_list.append(df)
    master_df = pd.concat(df_list, axis=0).sort_values('Date').reset_index(drop=True)
    return master_df

if __name__ == "__main__":
    files = ['AMZN_stocks_data.csv', 'GOOGL_stocks_data.csv', 'IBM_stocks_data.csv', 'MSFT_stocks_data.csv']
    stocks = ['AMZN', 'GOOGL', 'IBM', 'MSFT']
    
df = load_and_aggregate(file_list)  
df


# In[33]:


# 3. Data Exploration & EDA
def explore_data(df):
    print(df.info())
    print(df.describe())
    print("Missing Values:\n", df.isnull().sum())
    
explore_data(df)


# In[34]:


sns.histplot(df['Volume'], kde=True)
plt.title('Frequency Distribution of Volumes')
plt.show()


# In[30]:


plt.figure(figsize=(15,5))
for stock in df['Name'].unique():
       temp = df[df['Name'] == stock]
       plt.plot(temp['Date'], temp['Volume'], label=stock)
plt.title('Stock Volume Variation Over Time')
plt.legend()
plt.show()


# In[35]:


close_df = df.pivot(index='Date', columns='Name', values='Close')
sns.heatmap(close_df.corr(), annot=True)
plt.title('Close Price Correlation Between Stocks')
plt.show()
explore_data(df)


# In[58]:


files = ['AMZN_stocks_data.csv', 'GOOGL_stocks_data.csv', 'IBM_stocks_data.csv', 'MSFT_stocks_data.csv']
stocks = ['AMZN', 'GOOGL', 'IBM', 'MSFT']

# Load and explore data
df = load_and_aggregate(files)
explore_data(df)

# Parameters
window_size = 30
stride = 1
test_ratio = 0.2

# Create windows and scale data
X, y = create_windows(df, stocks, window_size, stride)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test)

# Build, train, and evaluate models
models = {
    "Simple RNN": create_rnn,
    "LSTM": create_lstm,
    "GRU": create_gru
}



# In[60]:


# Prepare data
X, y = create_windows(df, stocks, window_size, stride)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test)

# Model configurations
models = {
    "Simple RNN": create_rnn,
    "LSTM": create_lstm,
    "GRU": create_gru
}


# In[43]:


# 4. Create Time Windows and Scaling Helper Functions
window_size = 30
window_stride = 1
test_ratio = 0.2

X, y = create_windows(df, stocks, window_size, window_stride)

def create_windows(df,stocks, window_size, window_stride):
    data = df.pivot(index='Date', columns='Name', values='Close')
    data = data[stocks].dropna()

    X, y = [], []
    for i in range(0, len(data) - window_size, window_stride):
        window = data.iloc[i:i+window_size].values
        target = data.iloc[i+window_size].values
        X.append(window)
        y.append(target)
    print(f"Total sequences generated: {len(X)}")
    return np.array(X), np.array(y)
create_windows(df,stocks, window_size, window_stride)


# In[68]:


# Set window parameters
window_size = 30
stride = 1
test_ratio = 0.2

# Prepare data
X, y = create_windows(df, stocks, window_size, stride)
# First, split into train and temp (which will later be split into val and test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)

# Then split temp into training and validation sets (e.g., 80% train, 20% val of temp)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test)

def scale_data(X_train, X_test, y_train, y_test):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    nsamples, nx, ny = X_train.shape

    X_train_scaled = X_scaler.fit_transform(X_train.reshape((nsamples, nx*ny))).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape((X_test.shape[0], nx*ny))).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler
scale_data(X_train, X_test, y_train, y_test)


# In[69]:


# 5. Model Creation Functions
input_shape = X_train_scaled.shape[1:]
output_dim = y_train_scaled.shape[1]


def create_rnn(input_shape, output_dim, units=64):
    model = Sequential()
    model.add(SimpleRNN(units=units, input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm(input_shape, output_dim, units=64):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru(input_shape, output_dim, units=64):
    model = Sequential()
    model.add(GRU(units=units, input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

create_rnn(input_shape, output_dim, units=64)
create_lstm(input_shape, output_dim, units=64)
create_gru(input_shape, output_dim, units=64)


# In[70]:


def train_model(model, X_train, y_train, X_val, y_val):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es])
    return history
train_model(model, X_train, y_train, X_val, y_val)


# In[73]:


def evaluate_and_plot(model, X_test, y_test, y_scaler,stocks, model_name="Model"):
    preds = model.predict(X_test)
    preds_rescaled = y_scaler.inverse_transform(preds)
    actuals_rescaled = y_scaler.inverse_transform(y_test)

    for i, stock in enumerate(stocks):
        plt.figure(figsize=(12, 4))
        plt.plot(preds_rescaled[:, i], label=f'{stock} Predicted')
        plt.plot(actuals_rescaled[:, i], label=f'{stock} Actual')
        plt.title(f'{model_name} - {stock} Stock Price Prediction')
        plt.legend()
        plt.show()

        rmse = np.sqrt(mean_squared_error(actuals_rescaled[:, i], preds_rescaled[:, i]))
        mae = mean_absolute_error(actuals_rescaled[:, i], preds_rescaled[:, i])
        print(f"{model_name} - {stock} RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
evaluate_and_plot(model, X_test, y_test, y_scaler,stocks, model_name="Model")


# In[79]:


#6. Main Pipeline Execution

if __name__ == "__main__":
    files = ['AMZN_stocks_data.csv', 'GOOGL_stocks_data.csv', 'IBM_stocks_data.csv', 'MSFT_stocks_data.csv']
    stocks = ['AMZN', 'GOOGL', 'IBM', 'MSFT']
    df = load_and_aggregate(files)

    explore_data(df)

    window_size = 30
    stride = 1
    test_ratio = 0.2

    X, y = create_windows(df, stocks, window_size, stride)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Check for neural network compatibility
    assert len(X.shape) == 3 and len(y.shape) == 2, "Data shapes not compatible with RNN input."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test)

    models = {
        "Simple RNN": create_rnn,
        "LSTM": create_lstm,
        "GRU": create_gru
    }

    for name, builder in models.items():
        print(f"\nTraining {name} model...")
        model = builder(X_train_scaled.shape[1:], y_train_scaled.shape[1])
        train_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        evaluate_and_plot(model, X_test_scaled, y_test_scaled, y_scaler, stocks, model_name=name)


# In[78]:


# Final Conclusion
print("\nConclusion: All three models—RNN, LSTM, and GRU—were trained and evaluated.")
print("LSTM and GRU generally perform better due to their ability to capture long-term dependencies in sequences.")
print("This notebook can be extended further by adding hyperparameter tuning or integrating external financial indicators.")


# In[ ]:




