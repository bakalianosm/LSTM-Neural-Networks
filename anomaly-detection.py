from logging import PercentStyle
import math
from datetime import date
from os import name, sep
import sys # for arguments
import argparse
from typing import Sequence
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, optimizers, losses, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sqlalchemy import true
from utilities import *
import seaborn as sns
import random
# import plotly.graph_objs as go
import plotly.graph_objects as go

import pickle

pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser(description='Process arguments for the forecast')
parser.add_argument('-d', required=True,  help='Path for the input file')
parser.add_argument('-n', required=True, help='Number for selected time series')
parser.add_argument('-mae', required=True, help='Error value as double')
args = parser.parse_args()

path = args.d 
n_time_series = int(args.n)
error_val = np.double(args.mae)

# How many values to see behing for predicting the next (its like window size)
numN = 40
print("Path is " + path + " and num n is " + args.n)

df = pd.read_csv(path, sep='\t', header=None)

row,col = df.shape
print("rows are " , row, " and columns are ", col )

columns_number = col
# Separate names and values
names = df.iloc[0:,0]
stock_values_df = df.iloc[0:,1:]



train_set_count = math.ceil((0.9) * columns_number)
test_set_count = columns_number - train_set_count -1

print("Train set has " , train_set_count, " items")
print("Test set has " , test_set_count, " items")



# comb_training_set, comb_test_set = np.array([]), np.array([])
# comb_training_set, comb_test_set = comb_training_set.reshape(-1,1) ,comb_test_set.reshape(-1,1)

# appended = pd.DataFrame()
# for i in range(0, 300):
#   current_df = detect_fix_dataset(i,stock_values_df.iloc[i])
#   # print(current_df)
#   training_set = current_df.iloc[:train_set_count]
#   appended = pd.concat([appended, training_set], ignore_index=True)
  
# train = appended



scaler = MinMaxScaler()
# scaler = scaler.fit(train[['Close']])
# train['Close'] = scaler.transform(train[['Close']])


TIME_STEPS = 50
# reshape to [samples, time_steps, n_features]
# X_train, y_train = create_dataset(
#   train[['Close']],
#   train.Close,
#   TIME_STEPS
# )


# print(X_train)
# model = keras.Sequential()
# model.add(layers.LSTM(
#     units=64,
#     input_shape=(X_train.shape[1], X_train.shape[2])
# ))
# model.add(layers.Dropout(rate=0.3))

# model.add(layers.RepeatVector(n=X_train.shape[1]))

# model.add(layers.LSTM(units=64, return_sequences=True))
# model.add(layers.Dropout(rate=0.3))
# model.add(
#   layers.TimeDistributed(
#     layers.Dense(units=X_train.shape[2])
#   )
# )
# model.compile(loss='mae', optimizer='adam')
# history = model.fit(
#     X_train, y_train,
#     epochs=32,
#     batch_size=128,
#     validation_split=0.1,
#     shuffle=False
# )
# model
# model.save("models/model-detect-multiple-32-epochs-300-data-for-train")



model = keras.models.load_model('models/detect_model')
print("Model type = ", type(model))
train_size = train_set_count


threshold = error_val
print("row is ", row-1)

for i in range (0, n_time_series):
  
  r = random.randint(300,row-1)

  print("Working for item " , r , " of the dataset")
  current_df = detect_fix_dataset(r,stock_values_df.iloc[r])
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=current_df['Date'], y=current_df['Close'], name='Close price'))
  fig.update_layout(showlegend=True, title='Stock price per day')
  fig.show()
  test =  current_df.iloc[train_size:len(current_df)]
  test['Close'] = scaler.fit_transform(test[['Close']])

  X_test, y_test = create_dataset(
  test[['Close']],
  test.Close,
  TIME_STEPS
  )
  X_test_pred = model.predict(X_test)

  test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

  # plt.hist(test_mae_loss, bins=50)
  # plt.xlabel('Test MAE loss')
  # plt.ylabel('Number of samples')
  
  test_score_df = pd.DataFrame(test[TIME_STEPS:])
  test_score_df['loss'] = test_mae_loss
  test_score_df['threshold'] = threshold
  test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
  test_score_df['Close'] = test[TIME_STEPS:]['Close']


  


  # fig = go.Figure()
  # fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['loss'], name='Test loss'))
  # fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['threshold'], name='Threshold'))
  # fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
  # fig.show()
  

  anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
  anomalies.shape
  if(anomalies['Close'].empty == True):
      print("No anomalies for this threshold")
      continue
  anomalies['Close'] = scaler.inverse_transform(anomalies[['Close']])
  test_score_df['Close'] =  scaler.inverse_transform(test_score_df[['Close']])

  name = names[r]
  import seaborn as sns


  # fig = go.Figure()
  # fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['Close'], name='Spot Price'))
  # fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'],mode='markers' ,name='Anomalies'))
  # fig.update_layout(showlegend=True, title='Anomalies')
  # fig.show()


  plt.figure(figsize=(16,8))
  plt.title(f"Detecting anomalies for %s stock" % name)

  plt.plot(test_score_df.index, test_score_df['Close'], color = 'blue', label = 'spot price')
  sns.scatterplot(anomalies.index, anomalies['Close'], color = sns.color_palette()[3], label = 'anomalies')

  plt.legend()
  plt.show()

