import pandas as pd
import keras

RUN_NAME = "run 2 wit 16 nodes"

# Loading Data from CSV file
training_data_df = pd.read_csv('sales_data_training.csv')
testing_data_df = pd.read_csv('sales_data_test.csv')

# Data Scaling 
# Because Neural Nets work best when data is within small range (like 0 to 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler( feature_range= (0,1) )

# scaling training and test data
training_scaled = scaler.fit_transform( training_data_df )
testing_scaled = scaler.transform( testing_data_df ) 

print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# Creating pandas data frame objects from scaled data
training_scaled_df = pd.DataFrame( training_scaled, columns = training_data_df.columns.values)
testing_scaled_df = pd.DataFrame( testing_scaled, columns = testing_data_df.columns.values)

X = training_scaled_df.drop('total_earnings', axis=1).values
Y = training_scaled_df[['total_earnings']].values

# Define model
from keras.models import Sequential
from keras.layers import *

model = Sequential()

# Add first Layer
model.add(Dense(16 , input_dim=9 , activation='relu', name="layer1"))    # Dense Layer is fully connected 
                                                          # 32 is number of neurons in this layer
                                                          # For the first Layer we need to specify the input Dimension (= number of features in our data set)
model.add(Dense(64, activation='relu', name="layer2"))
model.add(Dense(32, activation='relu', name="layer3"))
model.add(Dense(1, activation='linear', name="outputlayer")) # OUTPUT LAYER # activation = linear, because output is just 1 value : total earnings

model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# Create a TensorBoard Logger 
logger = keras.callbacks.TensorBoard(
                                    log_dir = "logs/{}".format(RUN_NAME),
                                    write_graph=True,
                                    histogram_freq=5)

# Prepareing Test data
X_test = testing_scaled_df.drop('total_earnings', axis=1).values
Y_test = testing_scaled_df[['total_earnings']].values


# Training the model
model.fit( X, Y, epochs=100, shuffle=True, verbose=2 , callbacks=[logger] , validation_data=(X_test,Y_test))


# Tesing model
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)

# Making predictions on new data set 
X = pd.read_csv("make_predictions.csv").values # this data is already scaled

prediction = model.predict( X )

prediction = prediction  - scaler.min_[8]
prediction = prediction / scaler.scale_[8]


model.save("trained_model.h5")

from keras.models import load_model
model = load_model("trained_model.h5")





























