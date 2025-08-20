
# Chest X-Ray Pneumonia Detection

## Project Overview
This project focuses on building a deep learning model to classify chest X-ray images as either normal or indicative of pneumonia. The goal is to develop an automated system that can assist in the detection of pneumonia from X-ray images.

## Dataset
The dataset used for this project is the **Chest X-Ray Images (Pneumonia)** dataset, available on Kaggle. It contains a large collection of chest X-ray images categorized into two classes: 'NORMAL' and 'PNEUMONIA'.

## Model Architecture
The model utilizes a transfer learning approach based on the **VGG16** convolutional neural network. The pre-trained VGG16 model, trained on the ImageNet dataset, is used as a feature extractor. A global average pooling layer and two dense layers (with ReLU activation and a sigmoid output layer for binary classification) are added on top of the VGG16 base to adapt it for the chest X-ray classification task. The VGG16 base layers are kept frozen during initial training.

## Code Structure
This project is implemented in a Jupyter notebook format. Here's a breakdown of the key steps and their corresponding code cells:

*   **Data Loading and Unzipping:** Download and extract the dataset from Kaggle.
*   **Setup Dataset Paths and Parameters:** Define variables for dataset directories, image dimensions (height and width), and batch size.
*   **Create TensorFlow Datasets:** Load the images from the directories into TensorFlow `Dataset` objects for training, validation, and testing, automatically inferring class labels from folder names.
*   **Visualize Sample Images:** Display a few example images from the training dataset along with their class labels to get a visual understanding of the data.
*   **Prepare Data for Model:** Apply caching and prefetching to the datasets to optimize data loading and improve training performance.
*   **Define Model Architecture:**
    *   Load the pre-trained VGG16 model as the base, excluding its top classification layers.
    *   Freeze the layers of the VGG16 base model so their weights are not updated during initial training.
    *   Define the input shape for the model.
    *   Add a preprocessing layer suitable for the VGG16 model.
    *   Connect the VGG16 base model to the input, passing the input through the preprocessing layer first.
    *   Add a Global Average Pooling 2D layer to reduce the spatial dimensions of the feature maps from the base model.
    *   Add a dense layer with ReLU activation.
    *   Add a final dense layer with a sigmoid activation for binary classification (Pneumonia vs. Normal).
    *   Create the final Keras Model by specifying the input and output layers.
*   **Compile the Model:** Configure the model for training by specifying the optimizer (Adam), the loss function (binary crossentropy for binary classification), and the evaluation metric (accuracy).
*   **Train the Model:** Train the compiled model using the training dataset and validate its performance on the validation dataset over a specified number of epochs.
*   **Visualize Training History:** Plot the training and validation accuracy and loss over epochs to evaluate the model's learning progress and identify potential overfitting.
*   **Evaluate the Model:** Assess the trained model's performance on the unseen test dataset to get an estimate of its generalization capability.
*   **Save the Model:** Save the trained model in the Keras format for future use or deployment.
*   **Image Prediction Function:** Define a Python function that takes an image file path and the trained model as input, preprocesses the image, and returns the model's prediction (probability of pneumonia).
*   **Perform Prediction on Uploaded Image:** Use the defined function to make a prediction on an image uploaded by the user and print the predicted class based on a threshold of 0.5.
