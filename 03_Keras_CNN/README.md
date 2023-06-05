# OCR Guide: Custom OCR with Convolutional Neural Networks Using Keras

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

This sub-folder deals with the custom OCR techniques that apply Convolutional Neural Networks (CNN) using Tensorflow/Keras.

Table of contents:

- [OCR Guide: Custom OCR with Convolutional Neural Networks Using Keras](#ocr-guide-custom-ocr-with-convolutional-neural-networks-using-keras)
  - [1. Introduction](#1-introduction)
  - [2. Notebooks](#2-notebooks)

## 1. Introduction

In this section/module, CNNs are trained with Tensorflow/Keras. The training datasets are:

- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Kaggle A-Z](https://iaexpert.academy/arquivos/alfabeto_A-Z.zip)

Then, the trained networks are used to recognize custom hand-written characters.

## 2. Notebooks

The notebooks are self-explanatory:

- [`01_Custom_OCR_training_the_neural_network.ipynb`](./01_Custom_OCR_training_the_neural_network.ipynb)
- [`02_Custom_OCR_Text_recognition.ipynb`](./02_Custom_OCR_Text_recognition.ipynb)

The first carries out these steps:

- Both number and letter datasets are loaded and concatenated.
- A weight is computed for each symbol, since the dataset is quite imbalanced.
- A CNN is defined (138k params) and trained and evaluated.
- The network is saved to disk.

The second carries out these steps:

- The trained network is loaded.
- A target image is loaded.
- The image is preprocessed to obtain each single character on it in a separate ROI:
  - Image is thresholded
  - Edges of the letter blobs are found
  - Contours of the dilated edges are found
  - Bounding boxes of the disconnedted contours are found
  - For each bounding box, a letter ROI is generated
- We take all ROIs and feed them one by one to our model: the symbol class is predicted.


