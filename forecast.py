from audioop import minmax
from logging import PercentStyle
import math
from datetime import date
from os import name, sep
from ssl import CertificateError
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
from keras.models import model_from_json
import pickle
from utilities import *
import random


pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser(description='Process arguments for the forecast')
parser.add_argument('-d', required=True,  help='Path for the input file')
parser.add_argument('-n', required=True, help='Number for selected time series')
args = parser.parse_args()

path = args.d 
n_time_series = int(args.n)

# look back
numN = 30
# How many values to see behing for predicting the next (its like window size)
# numN = 20
print("Path is " + path + " and num n is " + args.n)

df = pd.read_csv(path, sep='\t', header=None)

row,col = df.shape
print("rows are " , row, " and columns are ", col )

columns_number = col
# Separate names and values
names = df.iloc[0:,0]
stock_values_df = df.iloc[0:,1:]


train_set_count = math.ceil((0.8) * columns_number)
test_set_count = columns_number - train_set_count -1

print("Train set has " , train_set_count, " items")
print("Test set has " , test_set_count, " items")

sc = MinMaxScaler(feature_range = (0, 1))

           
# ------------------------------------------- # MAKE THE MODEL(S) ------------------------------------------- #

                    # MAKING MODEL ---> 80% OF ALL TIME SERIES, 32 EPOCHS, 256 BATCHES
# comb_training_set, comb_test_set = np.array([]), np.array([])
# comb_training_set, comb_test_set = comb_training_set.reshape(-1,1) ,comb_test_set.reshape(-1,1)

# for i in range(0, row-1):
#   current_df = fix_dataset(i,stock_values_df.iloc[i])
#   training_set = current_df.iloc[:train_set_count, 0:1].values
#   test_set = current_df.iloc[train_set_count:, 0:1].values 
  
#   comb_training_set = np.concatenate((comb_training_set,training_set),axis=0 )
#   comb_test_set = np.concatenate((comb_test_set,test_set),axis=0 )


# print(comb_training_set.shape)
# print(comb_test_set.shape)

# training_set_scaled = scale_data(sc,comb_training_set)

# X_train = []
# y_train = []

# X_train, y_train = create_data(numN, train_set_count,100, training_set_scaled)

# print(X_train.shape)
# print(y_train.shape)

# multi_model = create_model_multiple( 80, 'linear', 'adam', 'mae', X_train, y_train)
# multi_model.summary()
# fit_model(multi_model, 32, 256, X_train, y_train)

# save it
#multi_model.save("models/model-forecast-multiple-32-epochs-300-time-series-256-batch")

# ------------------------------------------- # OR LOAD THE PRETRAINED MODEL ------------------------------------------- #
 

                       #MODEL THAT I MADE USING 300 TIME SERIES
multi_model = keras.models.load_model('models/forecast_model')


import random
for i in range(0, n_time_series):
  r = random.randint(0, row-1)
 
  calculate(r,names,stock_values_df,train_set_count,numN,sc,test_set_count, multi_model)

  print("Fitting each time series in itself ")

  # making the model for one time series training
  current_df = fix_dataset(r, stock_values_df.iloc[r])
  training_set = current_df.iloc[:train_set_count, 1:2].values
  test_set = current_df.iloc[train_set_count:, 1:2].values
  training_set_scaled = scale_data(sc,training_set)

  X_train = []
  y_train = []

  X_train, y_train = create_data(numN, train_set_count,1, training_set_scaled)
  print("For 1 time series x train shape",X_train.shape)
  print("For 1 time series y train shape" ,y_train.shape)

  model_single = create_model_single( 80, 'linear', 'adam', 'mae', X_train, y_train)
  fit_model(model_single, 32, 65, X_train, y_train)
  calculate(r,names,stock_values_df,train_set_count,numN,sc,test_set_count, model_single)

