# Project Trubbish

This is a machine learning program implemented in Python using TensorFlow to classify images of trash into six broad categories. The aim of this program is to assist with recycling efforts by accurately categorizing different types of trash.

## Prerequisites
- Python
- TensorFlow
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Installation
1. Make sure you have Python installed. You can download it from the official Python website: [python.org](https://www.python.org/downloads/).
2. Install the required Python packages by running the following command:
   ```
   pip install tensorflow numpy matplotlib Pillow
   ```

## Usage
1. Download or clone the project repository to your local machine.
2. Place the images you want to classify into the appropriate folders based on their labels. The labels are as follows:
   - Glass: `Garbage classification/glass/`
   - Paper: `Garbage classification/paper/`
   - Cardboard: `Garbage classification/cardboard/`
   - Plastic: `Garbage classification/plastic/`
   - Metal: `Garbage classification/metal/`
   - Trash: `Garbage classification/trash/`

3. Open the `zero-indexed-files.txt` file and ensure that each image directory is correctly listed with its corresponding label index (0 to 5).
4. Run the Python script using the following command:
   ```
   python trash_classification.py
   ```

   This will train the neural network using the provided image dataset. The trained model will be saved in the `training_1` directory.

5. After training, you can use the `recognise()` function to classify images in a given folder directory. Simply provide the folder path as an argument to the function call. For example:
   ```python
   result = recognise("prediction/")
   print(result)
   ```

   This function will return the predicted label for the given image.

## File Descriptions
- `zero-indexed-files.txt`: A text file containing image directories and their corresponding labels. Update this file with the correct image paths and labels before running the program.
- `trash_classification.py`: The main Python script that implements the machine learning program using TensorFlow. It trains the model and provides the `recognise()` function for image classification.

## Model Architecture
The neural network model used for trash classification consists of the following layers:
1. Input Rescaling: Normalizes pixel values to the range [0, 1].
2. Convolutional Layers: Extracts features from the input images using convolutional filters.
3. Max Pooling Layers: Reduces the spatial dimensions of the feature maps.
4. Flatten Layer: Flattens the output from the previous layer into a 1D vector.
5. Dense Layers: Fully connected layers that learn high-level representations of the input features.
6. Output Layer: Produces logits (raw predictions) for each class label.

## Training
The model is trained using the Adam optimizer and the Sparse Categorical Crossentropy loss function. The training process involves iterating over the training set for a specified number of epochs. After each epoch, the model's weights are saved using checkpoints to allow for resuming training or using the trained model for predictions later.

## Prediction
The `recognise()` function takes a folder directory as input and classifies the images within that folder. It resizes the images to match the input size expected by the model and then uses the trained model to predict the labels for the images. The function returns the predicted label for the given images.

Please ensure that the folder you provide to the `recognise()` function contains only the images you want to classify.

**Note:** Before running the prediction, make sure you have trained the model by running the script and that you have images to classify in the specified folder.

## Disclaimer
This machine learning model for trash classification has its limitations and may not achieve perfect accuracy. It's always recommended to validate the predictions and use human judgment for critical decisions.

## License
This project is licensed under the [MIT License](LICENSE).