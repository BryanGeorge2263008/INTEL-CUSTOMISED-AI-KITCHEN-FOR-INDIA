# Intel Customised AI Kitchen for India

This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow and Keras, designed to recognize various vegetables and suggest recipes based on the identified items. The model is trained on the Fruits 360 dataset, which has been augmented for better performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project aims to build a robust vegetable recognition system using a pre-trained MobileNetV2 model. The recognized vegetables can then be used to suggest recipes, providing a practical application for meal planning and dietary management.

## Dataset

The dataset used for training and testing the model is the Fruits 360 dataset, which includes a wide variety of fruits and vegetables. The dataset is split into training and testing sets, with further splitting of the training set for validation.

- Training Directory: /content/fruits/fruits-360_dataset/fruits-360/Training
- Testing Directory: /content/fruits/fruits-360_dataset/fruits-360/Test

## Model Architecture

The model is based on MobileNetV2, a lightweight deep learning model optimized for mobile and embedded vision applications. The architecture includes:

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- GlobalAveragePooling2D Layer
- Dense Layer: 512 units, ReLU activation
- Dropout Layer: 50% dropout rate
- Output Layer: Softmax activation, number of classes based on the dataset

## Data Augmentation

Data augmentation is applied to the training data to enhance the model's generalization ability. The augmentation techniques include:

- Rescaling
- Shear Transformation
- Zoom Transformation
- Horizontal Flip
- Rotation
- Width and Height Shift
- Fill Mode: Nearest

## Training the Model

The model is trained using the augmented dataset with the following configurations:

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks: EarlyStopping, ReduceLROnPlateau

### Training Script

python
# Training the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)


## Evaluation

The model is evaluated on the test dataset to determine its accuracy and performance.

### Evaluation Script

python
# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)


## Usage

To use the model, follow these steps:

1. Clone the repository:
    bash
    git clone https://github.com/BryanGeorge2263008/INTEL-CUSTOMISED-AI-KITCHEN-FOR-INDIA.git
    
2. Install the required dependencies (see below).
3. Prepare your dataset in the specified directory structure.
4. Run the training script to train the model.
5. Use the trained model to predict vegetables and suggest recipes.

## Dependencies

- TensorFlow
- Keras
- Python 3.x
- numpy
- matplotlib

Install the dependencies using pip:

bash
pip install tensorflow numpy matplotlib


## Results

The model achieved an accuracy of 91% on the test dataset. Detailed training and validation metrics are available in the training logs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.
