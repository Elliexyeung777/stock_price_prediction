import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

class StockPredictionEngine:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def preprocess_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))        
        X = []
        y = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    def train_model(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        return rmse

