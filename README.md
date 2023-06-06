# Guide on Optical Character Recognition (OCR)

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

The repository is organized in theme-related folders and each of them contains (1) a guide in Markdown which explains everything (including setup & Co.) and (2) example code.

- [`01_Tesseract`](./01_Tesseract): main folder with all the necessary basics for OCR with Tesseract.
  - Tesseract: installation, usage
  - Image processing for OCR
  - EAST (detection) + OpenCV (image processing) + Tesseract (recognition) for OCR in natural scenarios
  - OCR in videos: modularization of all learned functions into a video application
- [`02_EasyOCR`](./02_EasyOCR): package which detects very easily text in natural scenes.
- [`03_Keras_CNN`](./03_Keras_CNN): training of a Keras-CNN model to detect handwritten digits.
- [`04_Projects`](./04_Projects):
  - Project 1: specific terms are searched and highlighted in book images.
  - Project 2: processing (alignment, thresholding, etc.) of a receipt to apply OCR.
  - Project 3: license plate detection.

The folder `01` is probably the most important, since the other introduce additional packages and extra examples.

If you are interested in other related guides:

- A compilation of Object Detection and Segmentation Examples: [detection_segmentation_pytorch](https://github.com/mxagar/detection_segmentation_pytorch)
- My notes on PyImageSearch tutorials: [pyimagesearch_tutorials](https://github.com/mxagar/pyimagesearch_tutorials)
- My notes on the Udacity Deep Learning Nanodegree: [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)
- My notes on the Udacity Computer Vision Nanodepree: [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity)


Mikel Sagardia, 2023.  
No guarantees.
