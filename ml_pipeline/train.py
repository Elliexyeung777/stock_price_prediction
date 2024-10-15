# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
# Set plotting style
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
yf.pdr_override()

# Load data
dataset = pdr.get_data_yahoo('AAPL', start='2020-01-01', end=datetime.now())

# Define helper functions
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))

def plot_loss(history):
    plt.figure(figsize = (15,10))
    plt.plot(history.history['loss'], label='loss')
    plt.legend(loc='best')
    plt.show()

# Prepare data
dataset['RSI'] = ta.rsi(dataset.Close, length=15)
dataset['EMAF'] = ta.ema(dataset.Close, length=20)
dataset['EMAM'] = ta.ema(dataset.Close, length=100)
dataset['EMAS'] = ta.ema(dataset.Close, length=150)

dataset['Target'] = dataset['Open'] - dataset['Open'].shift(1)
dataset.dropna(inplace=True)

feat_columns = ['Open', 'High', 'Low', 'Close', 'RSI']
label_col = ['Target']

# Split training and test sets
split_date = dataset.index[int(len(dataset) * 0.8)]
training_set = dataset.loc[:split_date, feat_columns + label_col]
test_set = dataset.loc[split_date:, feat_columns + label_col]

X_train = training_set[feat_columns]
y_train = training_set[label_col]
X_test = test_set[feat_columns]
y_test = test_set[label_col]

# Normalize data
sc = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = sc.fit_transform(X_train.values)
X_test_scaled = sc.transform(X_test.values)

# Build model (2D input)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(feat_columns),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Plot loss curve
plot_loss(history)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate model
return_rmse(y_test, predictions)

# Create output folder if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Save loss curve
plt.figure(figsize=(15, 10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output/loss_curve.png')
plt.close()

# Save prediction results
plt.figure(figsize=(15, 10))
plt.plot(y_test.index, y_test.values, color='red', label='Actual Price')
plt.plot(y_test.index, predictions, color='blue', label='Predicted Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('output/prediction_results.png')
plt.close()

print('Charts have been saved in the output folder.')





