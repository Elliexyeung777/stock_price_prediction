
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import yfinance as yf
from pandas_datareader import data as pdr

def get_stock_data(symbol, start_date, end_date):
    yf.pdr_override()
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

def preprocess_data(data, feature_columns, target_column, n_steps):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[feature_columns])
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        y.append(data[target_column].iloc[i])
    
    return np.array(X), np.array(y), scaler

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=125, input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    return history

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print(f"The root mean squared error is {rmse:.2f}.")

def plot_loss(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history.history['loss'], label='loss')
    plt.legend(loc='best')
    plt.show()

