# OCR Guide: EasyOCR

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

This sub-folder deals with the package EasyOCR.

Table of contents:

- [OCR Guide: EasyOCR](#ocr-guide-easyocr)
  - [1. Introduction](#1-introduction)
  - [2. Usage](#2-usage)

## 1. Introduction

Links:

- EasyOCR is an Open Source library maintained by [Jaded AI](https://jaided.ai/)
- [Easy OCR Github](https://github.com/JaidedAI/EasyOCR)

Installation:

```bash
pip install easyocr
```

## 2. Usage

EasyOCR seems very easy to use and it seems to work very well with photos with unstructured or natural text content.

Notebook: [`01_OCR_with_EasyOCR_and_Python.ipynb`](./01_OCR_with_EasyOCR_and_Python.ipynb).

Summary of the contents:

```python
from easyocr import Reader
import cv2
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np
import easyocr

#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

DATA_PATH = "./../material/"

# It is important to have compatible
# OpenCV and EasyOCR versions
print(cv2.__version__) # 4.7.0
print(easyocr.__version__) # 1.7.0

# We need to specify the languages we have in the image
languages_list = ['en', 'pt']
print(languages_list)

# If we have a GPU, we should use it
gpu = False #True

img = cv2.imread(DATA_PATH+'Images/cup.jpg')
original = img.copy()

# We see we need 2 models: detection & recognition
# The inference is very easy: we pass the parameters to the Reader
# and then we call readtext()
# Note: we can get more than one box
reader = Reader(languages_list, gpu)
results = reader.readtext(img)

# The results object is a list of tuples
# Each tuple has:
# - The bounding box coordinates: [lt, rt, br, bl]
# - The text/string
# - The confidence
results
# [([[373, 313], [435, 313], [435, 331], [373, 331]],
#   'BEGIN .',
#   0.6404372122231472)]



```
