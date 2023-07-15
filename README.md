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

   With the sample data, the training accuracy achieved was greater than 99%, and the testing accuracy on unseen data was greater than 85%.

5. After training, you can use the `recognise()` function to classify images in a given folder directory. Simply provide the folder path as an argument to the function call. For example:
   ```python
   result = recognise("prediction/")
   print(result)
   ```

   This function will return the predicted label for the given image.

## File Descriptions
- `zero-indexed-files.txt`: A text file containing image directories and their corresponding labels. Update this file with the correct image paths and labels before running the program.
- `main.py`: The main Python script that implements the machine learning program using TensorFlow. It trains the model and provides the `recognise()` function for image classification.

## Model Architecture
The neural network model used for trash classification consists of the following layers:
1. Input Rescaling: Normalizes pixel values to the range [0, 1].
2. Convolutional Layers: Extracts features from the input images using convolutional filters.
3. Max Pooling Layers: Reduces the spatial dimensions of the feature maps.
4. Flatten Layer: Flattens the output from the previous layer into a 1D vector.
5. Dense Layers: Fully connected layers that learn high-level representations of the input features.
6. Output Layer: Produces logits (raw predictions) for each class label.

The neural network was sourced from [Keras Classification](https://www.tensorflow.org/tutorials/images/classification)

## Training
The model is trained using the Adam optimizer and the Sparse Categorical Crossentropy loss function. The training process involves iterating over the training set for a specified number of epochs. After each epoch, the model's weights are saved using checkpoints to allow for resuming training or using the trained model for predictions later.

With the provided sample data, the model achieved a training accuracy of over 99% and a testing accuracy of over 85% on unseen data.

## Prediction
The `recognise()` function takes a folder directory as input and classifies the images within that folder. It resizes the images to match the input size expected by the model and then uses the trained model to predict the labels for the images. The function returns the predicted label for the given images.

Please ensure that the folder you provide to the `recognise()` function contains only the images you want to classify.

**Note:** Before running the prediction, make sure you have trained the model by running the script and that you have images to classify in the specified folder.

## Disclaimer
This machine learning model for trash classification has its limitations and may not achieve perfect accuracy. It's always recommended to validate the predictions and use human judgment for critical decisions.

## Contact
Email ayushsharma14@gmail.com if there is some issue with running the program or to reporting bugs

## Troubleshooting

# Permission Denied
To troubleshoot the "permission denied" error, you can try running the program with administrative rights. Here's how you can do it:

Right-click on the Python script (.py file) that you want to run.
From the context menu, select "Run as administrator". This will execute the script with administrative privileges.
Running the program with administrative rights may help resolve any permission-related issues that are causing the "permission denied" error.

If you're using an integrated development environment (IDE) like PyCharm or Visual Studio Code, you can launch the IDE with administrative rights, and then run the program within the IDE.

Note that running a program with administrative rights should be done with caution, as it grants the program elevated privileges on your system.

# Training the model without resizing the images
Running the Program without Image Resizing
By default, the provided program resizes the input images to match the input size expected by the model. This resizing helps to speed up the training process and allows the program to handle images of different sizes consistently.

However, if you prefer to run the program without resizing the images, it is possible to do so. Keep in mind that not resizing the images can significantly increase the processing time and memory requirements, especially if your images have large dimensions.

To run the program without resizing the images, you can modify the code in the main.py script. Look for the section where the images are loaded and resized, and comment out or remove the lines of code responsible for resizing the images. Here's an example:

```
# Load and resize images
# image = Image.open(image_path)
# image = image.resize((img_width, img_height))
```
By removing or commenting out these lines, the program will load the images without resizing them. However, keep in mind that you will need to ensure that all your images have the same dimensions as expected by the model.

It's worth noting that running the program without resizing the images can result in improved accuracy during testing, as the model will be trained and tested on the original image sizes. However, it may come at the cost of increased computation time and memory usage, particularly if your dataset contains large images.

Before opting for this approach, consider the computational resources available to you and evaluate the trade-off between accuracy and processing time.

#
## License
This project is licensed under the [MIT License](LICENSE).
