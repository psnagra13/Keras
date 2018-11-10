import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load Keras Resnet50 model
# It is pretrained against the ImageNet Database
model = resnet50.ResNet50()

# Load Image and resize to 224x224 pixels
img = image.load_img("bay.jpg", target_size = (224,224))

# Convert the image to a numpy array
x= image.img_to_array(img)

# Add a fouth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)


# scale the input image to the range used in trained network
x = resnet50.preprocess_input(x)

# Predict
predictions = model.predict(x)

predictedClasses = resnet50.decode_predictions(predictions , top =9)

for imagenet_id, name, likelihood in predictedClasses[0]:
    print (" - {}: {:2f} likelihood".format(name, likelihood))