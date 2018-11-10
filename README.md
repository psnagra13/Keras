# Keras 

## Creating Model 

model = keras.models.Sequential() # For ANN
model.add(keras.layers.Dense())   # Adding Layers to the model

## Compiling Model 
model.compile(loss = 'mean_squared_error' , optimizer = 'adam')

## Training Phase
model.fit (training_data , expected_output)

## Testing Phase
error = model.evaluate( testing_data , expected_output)

## Saving Model 
model.save ("trained_model.h5")

## Loading Model from file
model = keras.models.load_model('trained_model.h5)

## Making Predictions
predictions = model.predict(new_data)


# Keras Sequential Model

model = keras.models.Sequential()

model.add(Dense(16, input_dim=9)) # Dense Layer is fully connected 

                                       # 16 is number of neurons in this layer

                                       # For the first Layer we need to specify the input Dimension (= number of features in our data set)

model.add( Dense( 64 ) )

model.add( Dense( 32))

model.compile( optimizer= 'adam' m loss='mse')


# Types of layers
Dense

keras.layers.convolutional.Conv2D()

keras.layers.recurrent.LSTM()

# Why we need scaling
Neural Nets are trained well when all coloumns are scaled to same range.

# Opening Tensorboard
tensorboard --logdir=logs   # logs is directory where logs are saved



