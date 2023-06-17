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
  - [3. Notebook Code](#3-notebook-code)

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

## 3. Notebook Code

[`01_OCR_with_EasyOCR_and_Python.ipynb`](./01_OCR_with_EasyOCR_and_Python.ipynb)

```python
# %% [markdown]
# # OCR with EasyOCR
# 
# - Supported by [Jaided AI](https://jaided.ai/)
# - Official repository: https://github.com/JaidedAI/EasyOCR

# %% [markdown]
# # Instalation

# %%
#!pip install easyocr

# %%
#!pip uninstall opencv-python-headless
#!pip install opencv-python-headless==4.1.2.30

# %%
from easyocr import Reader
import cv2
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATA_PATH = "./../material/"

# %%
import easyocr

# %%
# It is important to have compatible
# OpenCV and EasyOCR versions
print(cv2.__version__)
print(easyocr.__version__)

# %% [markdown]
# # Parameters

# %%
# We need to specify the languages we have in the image
languages_list = ['en', 'pt']
print(languages_list)

# %%
# If we have a GPU, we should use it
gpu = False #True

# %%
img = cv2.imread(DATA_PATH+'Images/cup.jpg')
cv2_imshow(img)

# %%
original = img.copy()

# %% [markdown]
# # Text recognition

# %%
# We see we need 2 models: detection & recognition
# The inference is very easy: we pass the parameters to the Reader
# and then we call readtext()
# Note: we can get more than one box
reader = Reader(languages_list, gpu)
results = reader.readtext(img)

# %%
# The results object is a list of tuples
# Each tuple has:
# - The bounding box coordinates: [lt, rt, br, bl]
# - The text/string
# - The confidence
results

# %% [markdown]
# # Writing the results

# %% [markdown]
# ## Text

# %%
# We use the Calibri font to display text on the image.
# We do it so, because Calibri has many symbols
font = DATA_PATH+'Fonts/calibri.ttf'

# %%
def write_text(text, x, y, img, font, color=(50,50,255), font_size=22):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font = font, fill = color)
    img = np.array(img_pil)
    return img

# %% [markdown]
# ## Bouding box
# 
# * **lt** = left top
# * **rt** = rigth top
# * **br** = bottom right
# * **bl** = bottom left 

# %%
def box_coordinates(box):
    # We need to convert the row/col values to ints
    # and they are additionally packed into a tuple
    (lt, rt, br, bl) = box
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    return lt, rt, br, bl

# %%
# First box, bbox coordinates
results[0][0]

# %%
# First box, bbox coordinates - but converted to a tuple of ints
box_coordinates(results[0][0])

# %%
def draw_img(img, lt, br, color=(200,255,0), thickness=2):
    # NOTE: we need only LT and BR for the bbox
    cv2.rectangle(img, lt, br, color, thickness)
    
    return img

# %%
img = original.copy()
for (box, text, probability) in results:
    print(box, text, probability)
    lt, rt, br, bl = box_coordinates(box)
    # NOTE: we need only LT and BR for the bbox
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img, font)

cv2_imshow(img)

# %%
# Another example with a photo
img = cv2.imread(DATA_PATH+'Images/google-cloud.jpg')
reader = Reader(languages_list, gpu)
results = reader.readtext(img)
results

# %%
for (box, text, probability) in results:
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img, font)

cv2_imshow(img)

# %% [markdown]
# # Other languages
# 
# - Documentation: https://www.jaided.ai/easyocr/

# %%
# Supported languages
# https://www.jaided.ai/easyocr/
# German: 'de'
# Spanish: 'es'
# ...
# All languages in the list will be detected
languages_list = ['en','fr']
languages_list

# %%
img = cv2.imread(DATA_PATH+'Images/french.jpg')
reader = Reader(languages_list, gpu)
results = reader.readtext(img)
results

# %%
for (box, text, probability) in results:
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img, font)

cv2_imshow(img)

# %%
# Another image with other languages
languages_list = ['en', 'ch_sim']
languages_list

# %%
font = DATA_PATH+'Fonts/simsun.ttc'

# %%
img = cv2.imread(DATA_PATH+'Images/chinese.jpg')
reader = Reader(languages_list, gpu)
results = reader.readtext(img)
results

# %%
for (box, text, probability) in results:
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img, font)

cv2_imshow(img)

# %% [markdown]
# # Text with background

# %%
def text_background(text, x, y, img, font, font_size=32, color=(200,255,0)):
    # A label/strip with a given color is created on the given coordinates
    # to put the detected text string on there
    background = np.full((img.shape), (0,0,0), dtype=np.uint8)
    text_back = write_text(text, x, y, background, font, font_size=font_size)
    text_back = cv2.dilate(text_back, (np.ones((3,5), np.uint8)))
    fx, fy, fw, fh = cv2.boundingRect(text_back[:,:,2])
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, -1)

    return img

# %%
font = DATA_PATH+'Fonts/calibri.ttf'
languages_list = ['en', 'pt']

# %%
img = cv2.imread(DATA_PATH+'Images/plate-information.jpg')
reader = Reader(languages_list, gpu)
results = reader.readtext(img)
results

# %%
for (box, text, probability) in results:
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br, (200,255,0))
    img = text_background(text, lt[0], lt[1], img, font, 18, (200,255,0))
    img = write_text(text, lt[0], lt[1], img, font, (0,0,0), 18)

cv2_imshow(img)

```