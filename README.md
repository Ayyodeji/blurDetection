# blurDetection
```markdown
# Multi-Classification Blur Detection Project

This repository contains a blur detection project that focuses on classifying images into sharp, defocused, and motion-blurred categories. The project utilizes a dataset from Kaggle called "blur-dataset" to train a convolutional neural network (CNN) for accurate blur detection.

## Dataset

The dataset used in this project is the ["blur-dataset"](https://www.kaggle.com/kwentar/blur-dataset) from Kaggle. It consists of three types of images: sharp images, defocused images, and motion-blurred images. The goal is to build a model that can distinguish between these categories based on features extracted from the images' edges.

## Project Structure

The project is organized as follows:

- `script/`: Contains Python script for data exploration, feature extraction, model training, and evaluation.
- `models/`: Contains saved model checkpoints.

## How to Use

1. Clone the repository:

   ```sh
   git clone https://github.com/ayyodeji/blurDetection.git
   ```
## Feature Engineering and Edge Detection

One of the critical steps in this project is feature engineering, which involves extracting edge detection features from the images. This process helps the model distinguish between different types of blur.

We utilize three edge detection filters to extract features:
- Laplace Filter: Captures second-order intensity changes.
- Sobel Filter: Detects edges using gradient magnitude.
- Roberts Filter: Highlights edges by approximating gradient magnitude.

These filters are applied to grayscale versions of the images, resulting in feature matrices that capture essential information about image edges and intensity changes.

2. Set up the environment (install required libraries):

   ```sh
   !pip install cv2 numpy os matplotlib keras skimage scikit-learn kaggle
   ```

3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the data preprocessing, feature extraction, model training, and evaluation steps.

4. Train and evaluate the blur detection model using the provided notebooks.

## Results

The trained model achieves the following results on the validation set:

- Accuracy: 72%
- AVerage F1-score: 70.33

The confusion matrix and classification report can be found in the notebooks.

## Future Work
This project serves as a foundation for further exploration:

Experiment with different CNN architectures and hyperparameters to improve model performance.
Investigate advanced image processing techniques for more accurate edge detection.
Explore transfer learning using pre-trained models for better generalization.
