from logging import PercentStyle
import math
from datetime import date
from os import name, sep
import sys  # for arguments
import argparse
from typing import Sequence
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, optimizers, losses, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pd.options.mode.chained_assignment = None


def fix_dataset(s, current_df):
    current_df = current_df.to_frame()
    new_col = []
    index = current_df.index
    number_of_rows = len(index)

    for idx in range(0, number_of_rows):
        text = idx
        new_col.append(text)

    current_df.rename(columns={s: 'Prices'}, inplace=True)
    current_df['Dates'] = new_col

    # print("Number of rows and columns:", current_df.shape)
    df = current_df['Prices']

    data = current_df[['Dates', 'Prices']]

    return data


def create_data(n, set_count, time_series, set_scaled):
    X = []
    y = []
    for i in range(n, set_count*time_series):
        X.append(set_scaled[i-n:i, 0])
        y.append(set_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y


def scale_data(sc,set_to_scale):
    set_to_scale = sc.fit_transform(set_to_scale)
    return set_to_scale


def create_data(n, set_count, time_series, set_scaled):
    X = []
    y = []
    for i in range(n, set_count*time_series):
        X.append(set_scaled[i-n:i, 0])
        y.append(set_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y


def create_model_multiple(unitsNum, activationFunction, optimizerFunction, lossFunction, train_set_x, train_set_y):
    model = keras.Sequential()

    # Add first layer
    # Note: return sequences is True because we want to feed its results to the next layer
    model.add(layers.LSTM(units=unitsNum, return_sequences=True,
                          input_shape=(train_set_x.shape[1], 1)))
    model.add(layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum, return_sequences=True))
    model.add(layers.Dropout(0.2))

    # # Adding a third LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum, return_sequences=True))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(units=unitsNum, return_sequences=True))
    model.add(layers.Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum))
    model.add(layers.Dropout(0.2))

    # Adding the output layer
    model.add(layers.Dense(units=1, activation=activationFunction))

    # Compiling the RNN
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # compiling using different optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def fit_model(currentModel, epochsNum, batchNum, train_set_x, train_set_y):
    currentModel.fit(train_set_x, train_set_y,
                     epochs=epochsNum, batch_size=batchNum)


def create_model_single(unitsNum, activationFunction, optimizerFunction, lossFunction, train_set_x, train_set_y):
    model = keras.Sequential()

    # Add first layer
    # Note: return sequences is True because we want to feed its results to the next layer
    model.add(layers.LSTM(units=unitsNum, return_sequences=True,
                          input_shape=(train_set_x.shape[1], 1)))
    model.add(layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum, return_sequences=True))
    model.add(layers.Dropout(0.2))

    # # Adding a third LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum, return_sequences=True))
    model.add(layers.Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(layers.LSTM(units=unitsNum))
    model.add(layers.Dropout(0.2))

    # Adding the output layer
    model.add(layers.Dense(units=1, activation=activationFunction))

    # Compiling the RNN
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # compiling using different optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def predict_and_inverse(scaler, curModel, test_set):
    predicted_stock_price = curModel.predict(test_set)
    # print(predicted_stock_price)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    predictions = predicted_stock_price
    return predictions


def plot_predicted_and_valid(current_df, training_count, predictions, name):
    prices = current_df
    train = prices[:training_count]
    valid = prices[training_count:]
    valid['Predictions'] = predictions
    print(predictions[5])
    plt.figure(figsize=(14, 8))

    plt.plot(train['Prices'])

    plt.title(f"Predicted values for %s stock" % name)

    plt.plot(valid[['Prices', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    mse = mean_squared_error(valid['Prices'],valid['Predictions'])
    rmse = math.sqrt(mse)
    print("Root mean square error is: " ,rmse)


def calculate(r,names,stock_values_df,train_set_count,numN,sc,test_set_count, model):
  print("Going to predict for " , names[r] , " stock in position ", r)
  current_df = fix_dataset(r, stock_values_df.iloc[r])
  
  dataset_train = current_df.iloc[:train_set_count, 1:2]
  dataset_test = current_df.iloc[train_set_count:, 1:2]
  dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
  
  inputs = dataset_total[len(dataset_total) - len(dataset_test) - numN:].values
  sc.fit_transform(inputs)
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)
  
  X_test = []
  for i in range(numN, test_set_count + numN):
    X_test.append(inputs[i-numN:i, 0])
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  # print(X_test.shape)

  predictions = predict_and_inverse(sc, model, X_test)

  name = names[r]
  plot_predicted_and_valid(current_df, train_set_count, predictions,name)


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def detect_fix_dataset(s,current_df):
  current_df = current_df.to_frame()
  new_col = []
  index = current_df.index
  number_of_rows = len(index)

  for idx in range(0,number_of_rows):
    text = idx
    new_col.append(text)

  current_df.rename(columns= {s: 'Close'}, inplace = True)
  current_df['Date'] = new_col

  # print("Number of rows and columns:", current_df.shape)
  df = current_df['Close']

  data = current_df[ ['Date', 'Close'] ]
 
  return data