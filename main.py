import os
import tensorflow
from tensorflow import keras
import numpy
import matplotlib.image as img
from PIL import Image

# Opening the file containing all the image directory and their training labels
zero_indexed_file = open("zero-indexed-files.txt",'r')
training_image_directory = []
training_image_label = []
label_type = ["glass","paper","cardboard","plastic","metal","trash"]

# Reading the file to sort the directories and their labels into arrays
for i in zero_indexed_file:
    i = i.split(" ")
    training_image_directory.append(i[0])
    training_image_label.append(int(i[1]))    

# Setting the exact directory for each image using their labels
for i in range(0,len(training_image_label)):
    match training_image_label[i]:
        case 0:
            training_image_directory[i] = "Garbage classification\\glass\\" + training_image_directory[i]
        case 1:
            training_image_directory[i] = "Garbage classification\\paper\\" + training_image_directory[i]
        case 2:
            training_image_directory[i] = "Garbage classification\\cardboard\\" + training_image_directory[i]
        case 3:
            training_image_directory[i] = "Garbage classification\\plastic\\" + training_image_directory[i]
        case 4:
            training_image_directory[i] = "Garbage classification\\metal\\" + training_image_directory[i]
        case 5:
            training_image_directory[i] = "Garbage classification\\trash\\" + training_image_directory[i]

# Code to resize data images (only needed to be run once)
'''
for i in training_image_directory:
    image = Image.open(i)
    image = image.resize((28,37))
    image = image.save("Resize\\" + i)
'''

# Code to use the resized images comment out if original dataset for training is preferred (Will increase training time)
# Training without resizing greatly increases testing accuracy with farily similar training accuracy
os.chdir("Resize\\")

# Making an array of each image processed as an array
training_image_array = []
for i in training_image_directory:
    training_image_array.append(img.imread(i,"r"))

# Converting the images and labels into a numpy array
training_set = numpy.array(training_image_array)
training_set_labels = numpy.array(training_image_label)
shape = training_set.shape

# Making the neural network
model = keras.Sequential([
  keras.layers.Rescaling(1./255, input_shape=(shape[1], shape[2], shape[3])),
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(6)
])

# Compiling the model to be trained
model.compile(optimizer = "adam", loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

# Making checkpoints for trained model
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
os.chdir("..\\")

# Command to train the neural network
model.fit(training_set, training_set_labels, epochs= 25,callbacks=cp_callback,verbose=1)

# Run this function on a folder directory to classify images in it
def recognise(folder_directory):
    prediction_directories = []
    prediction_images = []
    testing_set = []
    for i in os.listdir(folder_directory):
        prediction_directories.append(i)

    for i in range(0, len(prediction_directories)):
        try:
            image = Image.open(folder_directory + prediction_directories[i])
            image = image.resize((28,37))
            prediction_directories[i] = "cache\\" + prediction_directories[i]
            image = image.save(prediction_directories[i])
        except:
            pass

    for i in prediction_directories:
        try:
            prediction_images.append(img.imread(i))
        except:
            pass

    testing_set = numpy.array(prediction_images)

    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(testing_set)

    for i in os.listdir("cache\\"):
        try:
            os.remove("cache\\" + i)
        except:
            pass
    return label_type[numpy.argmax(predictions)]

print(recognise("prediction\\"))

