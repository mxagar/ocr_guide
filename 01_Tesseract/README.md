# OCR Guide: Basics and Tesseract

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

This sub-folder deals with OCR in general and the package Tesseract, which is the most common python library for OCR. This module alone should suffice to start working with OCR.

Table of contents:

- [OCR Guide: Basics and Tesseract](#ocr-guide-basics-and-tesseract)
  - [1. Introduction](#1-introduction)
    - [Installation](#installation)
    - [Course Material](#course-material)
  - [2. OCR with Python and Tesseract](#2-ocr-with-python-and-tesseract)
  - [3. Image Pre-Processing](#3-image-pre-processing)
  - [4. EAST for Natural Scenes](#4-east-for-natural-scenes)
  - [5. OCR in Videos](#5-ocr-in-videos)
  - [6. Notebook Code](#6-notebook-code)
    - [`01_OCR_with_Python_and_Tesseract.ipynb`](#01_ocr_with_python_and_tesseractipynb)
    - [`02_OCR_with_Python_Pre_processing.ipynb`](#02_ocr_with_python_pre_processingipynb)
    - [`03_OCR_with_Python_Text_detection_with_EAST.ipynb`](#03_ocr_with_python_text_detection_with_eastipynb)
    - [`04_OCR_in_Videos.ipynb`](#04_ocr_in_videosipynb)

## 1. Introduction

Tesseract is currently owned and maintained by Google: [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract).

It started as a PhD project in the 80's but has evolved since to be the most popular OCR engine; we can use its libraries or binaries to extract text from images.

### Installation

I created a new basic conda environment with [`../conda.yaml`](../conda.yaml) and followed the typical installation steps.

```bash
# Create environment with YAML, incl. packages
conda env create -f conda.yaml
conda activate ocr

# Track any changes and versions you have
pip list --format=freeze > requirements.txt
```

After the environment creation:

    Windows
      1. Install Tesseract OCR EXE
          https://github.com/UB-Mannheim/tesseract/wiki
          Choose the additional languages we want
      2. Set environment variable:
          Path: C:\Program Files\Tesseract-OCR
      3. Test: cmd
          >> tesseract --version
          # tesseract v5.3.1.20230401
      4. Python package: in desired environment
          >> python -m pip install pytesseract
    Mac
      Similar to the Windows installation,
      but steps 1-2 are replaced by
          >> brew install tesseract --all-languages
    
    Linux / WIndows WSL
      Similar to the Windows installation,
      but steps 1-2 are replaced by
          >> sudo apt install tesseract-ocr 

To check the tesseract version and the installed languages:

    tesseract --version
    tesseract --list-langs

If later on we want to add additional language packages, we can install them fetching the data from: [tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata):

    Windows
        1. Download desired package from https://github.com/tesseract-ocr/tessdata
            Example: por.traineddata (portuguese)
        2. Copy file to C:\Program Files\Tesseract-OCR\tessdata
    Linux
        (sudo) apt-get install tesseract-ocr-por # Portuguese

An alternative is to download those packages manually to a local folder and then specify that folder in the `--tessdata-dir` argument.

### Course Material

The material from the course [Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/) can be downloaded from this [Drive Link](https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing).

I have the material locally on the folder [`../material/`](../material/), but not committed to the repository.

## 2. OCR with Python and Tesseract

Links:

- [Colab notebook](https://colab.research.google.com/drive/1SGqZJeatvKqxS09rDPtoMtmgzQ1q9mwW?usp=sharing)
- [Material](https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing)

Contents of the section notebook [`01_OCR_with_Python_and_Tesseract.ipynb`](./lab/01_OCR_with_Python_and_Tesseract.ipynb):

- Plot images
- Download and use specific language packages
- Tesseract configuration parameters: folder of packages, language, page segmentation modes (PSM: text block, one word, etc.), etc.
- Plot detected bboxes on images + text (with specific language symbols): custom functions defined
- Usage of the most common API calls:

    ```python
    # Extract string
    pytesseract.image_to_string(...)

    # Extract orientation and script information
    pytesseract.image_to_osd(...)

    # Extrac text and additional info: type of text, confidence, bbox, etc.
    pytesseract.image_to_data(...)
    ```

Note that:

- Tesseract expectes nice and clear images, either GRAY or RGB, where the text is clearly legible; additionally, we should specify the PSM correctly for detection and recognition.
- If the above is not satisfied, we need to:
  - Apply image processing: filter, thresholding, edges, contours, etc.
  - Use EAST for text detection, then crop ROIs and process them for Tesseract.
- Another option for natural scenes is EasyOCR: this package does the image pre-processing and text detection automatically.

## 3. Image Pre-Processing

Links:

- [Colab notebook](https://colab.research.google.com/drive/13KCAIRvoEwrvnNgTWyW_eITVjEpnQXdO?usp=sharing)
- [Material](https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing)

Contents of the section notebook [`02_OCR_with_Python_Pre_processing.ipynb`](./lab/02_OCR_with_Python_Pre_processing.ipynb):

- Thresholding: Simple/Binary (global), Otsu (global, bi-modal), Adaptive (local, with Gaussian)
- Color inversion: to white background and dark text
- Image resizing
- Morphological operations for noise removal: Erosion, Dilation, Opening (= Erosion + Dilation), Closing (= Dilation + Erosion)
- Filters: Mean, Gaussian, Median, Bilateral

Summary of the API calls to OpenCV:

```python
# Gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
cv2.threshold(...)
cv2.adaptiveThreshold(...)

# Inversion
invert = 255 - gray

# Resizing
increase = cv2.resize(...)

# Morphological operations
erosion = cv2.erode(gray, np.ones((3,3), np.uint8))
dilation = cv2.dilate(gray, np.ones((3,3), np.uint8))

# Filters
average_blur = cv2.blur(gray, (5,5))
gaussian_blur = cv2.GaussianBlur(gray, (5,5), 0)
median_blur = cv2.medianBlur(gray, 3)
bilateral_filter = cv2.bilateralFilter(gray, 15, 55, 45)

# Edges and contours for bounding box detection
edges = cv2.Canny(dilation, 40, 150)

# One way of isolating disconnected blobs consists in computing
# the contours of the blobs (= letters); then, we can find the bounding box of the
# contours. To that end, we find the edges first, then contours, and finally the bboxes.
# Another option is to use cv2.connectedComponentsWithStats() directly
# on the thresholded image.
def find_contours(img):
    # Note that we are getting external contours
    # We need to take into account all possible paramaters
    # of findContours()...
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    # Note we are sorting from left-to-right
    conts = sort_contours(conts, method = 'left-to-right')[0]
    
    return conts

conts = find_contours(dilation.copy())
```

## 4. EAST for Natural Scenes

- [Colab notebook](https://colab.research.google.com/drive/1L_sGCRL6itW_v-Jk3TXNR68UHC39Fbah?usp=sharing)
- [Material](https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing)
- [Paper](https://arxiv.org/pdf/1704.03155v2.pdf)

Contents of the section notebook [`03_OCR_with_Python_Text_detection_with_EAST.ipynb`](./lab/03_OCR_with_Python_Text_detection_with_EAST.ipynb):

EAST = Efficient and Accurate Scene Text Detector (2017).

We need to distinguish 2 concepts:

1. Text detection or localization
2. Character recognition

Tesseract does both, but the text detection/localization works well only on structured/controled scenes. EAST is nowadays the best approach to efficiently detect text on unstructured scene images. When EAST is applied, the workflow is the following:

1. Detect/ocalize text with OpenCV using the EAST network.
2. Take the bounding boxes and apply Tesseract to recognize the string in them.

EAST is a fully convolutional neural network (FCN) which returns a score map and bouding box candidates which might contain text ROIs. Then, those ROIs are collapsed using non-maximum supression (NMS).

![EAST Architecture](./../assets/EAST_architecture.jpg)

The EAST model takes a 320x320 RGB image and returns a 80x80 (1) score/confidence map and a (2) map with the bounding box values. The decoding of the bounding boxes is explained in the paper and in [this Stackoverflow post](https://stackoverflow.com/questions/55583306/decoding-geometry-output-of-east-text-detection).

![EAST Bounding Boxes](./../assets/EAST_bboxes.jpg)

**IMPORTANT**: We need the model, which can be downloaded from different sources:

- [Udemy tutorial](https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing)
- [PyImageSearch tutorial](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/opencv-text-detection/opencv-text-detection.zip)

The notebook has the following contents:

- An image of a natural scene is loaded, resized to 320x320 and converted to blob.
- The weights of the EAST architecture are loaded to the OpenCV DNN module.
- The image is passed to the model; we obtain `scores` and `geometries`: 80x80 maps of confidences and bboxes, respectively.
- BBox geometries are extracted and non-maximum supression is applied.
  - NOTE: it seems that the model handles oriented bounding boxes, but these are simplified to AABBs.
- The ROIs are upscaled and Tesseract is applied on them to get the text.

Note: PyImageSearch has a very similar tutorial:

- [Blog post / tutorial](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/?_ga=2.242979714.1677396935.1685958200-1020982194.1685524223)
- [Source code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/opencv-text-detection/opencv-text-detection.zip)
- [Google Colab](https://colab.research.google.com/drive/1J9R4sUQwFJ8eQRcnOIqWmoxDKNeIOsY4?usp=sharing)

## 5. OCR in Videos

Notebook: [`04_OCR_in_Videos.ipynb`](./lab/04_OCR_in_Videos.ipynb).

In this notebook, nothing really new is shown, but everything is packed to detect and display text in video frames. Therefore, the major interest of the notebook lies on the techniques used to modularise and run everything.

The video was recorded on a car in a bussy city; traffic signals and similar object are detected. The video interface is based on OpenCV: basically, a loop in which frames are taken from a `VideoCapture` object.

Two versions are programmed:

- First, video frames are processed by EAST and then cropped ROIs passed to Tesseract.
- Everything is handled by EasyOCR

## 6. Notebook Code

### [`01_OCR_with_Python_and_Tesseract.ipynb`](./lab/01_OCR_with_Python_and_Tesseract.ipynb)

```python
# %% [markdown]
# # OCR with Python and Tesseract

# %% [markdown]
# This notebook comes from the Udemy course [Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/).
# 
# Table of contents:
# 
# - [Text recognition in images](#Text-recognition-in-images)
# - [Selection of texts](#Selection-of-texts)
# - [Searching specific information](#Searching-specific-information)
# - [Detecting texts in natural scenarios](#Detecting-texts-in-natural-scenarios)

# %% [markdown]
# # Text recognition in images

# %% [markdown]
# ## Installing Tesseract
# 
# - Documentation: https://pypi.org/project/pytesseract/

# %%
# See the notes on the README.md file for more information
# ALSO: Images & Co. need to be downloaded:
# https://drive.google.com/drive/folders/19b4RUoVMZ_lYeHn0lE2ueyJk36cm9rGB?usp=sharing
#!sudo apt install tesseract-ocr
#!pip install pytesseract

# %% [markdown]
# ## Importing the libraries

# %%
import pytesseract
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
import re

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATAPATH = "../../material/"

# %% [markdown]
# ## Reading the image

# %%
img = cv2.imread(DATAPATH+'Images/test01.jpg')
cv2_imshow(img)

# %%
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
# Automatic text extraction: lines are detected, as well as strings
# but it fails when the image/text quality gets more complicated
text = pytesseract.image_to_string(rgb)

# %%
print(text)

# %% [markdown]
# ## Support for other languages

# %%
img = cv2.imread(DATAPATH+'Images/test02-02.jpg')
cv2_imshow(img)

# %%
# Tesseract expects image arrays in RGB format!
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
# The text in the image is not perfectly correct
# because we don't specify the language (Portuguese)
text = pytesseract.image_to_string(rgb)
print(text)

# %%
# Shell/Terminal command to get all installed language packages
!tesseract --list-langs

# %%
# Tesseract version
!tesseract --version

# %%
# See README.md for more information on how to install other lang packages
#!apt-get install tesseract-ocr-por # Portuguese

# %%
!tesseract --list-langs

# %%
# Once we install the Portuguese package
# we specify it and use it.
# Now all symbols are correct
text = pytesseract.image_to_string(rgb, lang='por')
print(text)

# %%
# The recommended way of working with language packages
# is to download them to a local folder and to specify the directory
# in the config argument
!mkdir tessdata

# %%
!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

# %%
# The recommended way of working with language packages
# is to download them to a local folder and to specify the directory
# in the config argument.
# We can pass any argument in the config option, but we don't get
# any errors if we pass something that doesn't work.
config_tesseract = '--tessdata-dir tessdata'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(text)

# %%
!wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true

# %%
!ls tessdata/

# %% [markdown]
# ## Parameters

# %% [markdown]
# ### Page segmentation modes (PSM)

# %%
# We need to choose the page segmentation mode
# before processing an image with tesseract.
# If we don't, tesseract still works, but the results
# can be improved.
# These are the possible PSMs
!tesseract --help-psm

# %%
img = cv2.imread(DATAPATH+'Images/page-book.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
# We pass the parameters to the engine using config_tesseract
# PSM 6: Assume a single uniform block of text
# If we don't pass --psm 6, it still works, but the result is suboptimal
config_tesseract = '--tessdata-dir tessdata --psm 6'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(text)

# %%
config_tesseract = '--tessdata-dir tessdata --psm 6'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(text)

# %%
img = cv2.imread(DATAPATH+'Images/exit.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
# PSM 7: Treat the image as a single text line.
config_tesseract = '--tessdata-dir tessdata --psm 7'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)
print(text)

# %% [markdown]
# ### Page orientation

# %%
from PIL import Image
import matplotlib.pyplot as plt

# %%
# We can detect page-level information: number, orientation, etc.
# All that is known as OSD = Orientation and Script Detection
# To work with OSD, we need to pass a PIL image
# to tesseract
img = Image.open(DATAPATH+'Images/book01.jpg')
plt.imshow(img);

# %%
# OSD = Orientation and Script Detection
# Note that OSD has its own module: osd.traineddata
print(pytesseract.image_to_osd(img))

# %% [markdown]
# # Selection of texts
# 
# 

# %%
from pytesseract import Output

# %%
img = cv2.imread(DATAPATH+'Images/test01.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
# The call image_to_data delivers more information about the detected text
# The image is segmented in blocks/ROIs
# Each ROI is a box/block with probably one word.
# We get a dictionary with lists wherean element in each
# list refers to one word block/box.
# - block_num = each block is a ROI with probably one word
# - conf = prediction confidence (from 0 to 100. -1 means no text was recognized)
# - height = height of detected block of text (bounding box)
# - left = x coordinate where the bounding box starts
# - level = category of the detected block: 1. page, 2. block, 3. paragraph, 4. line, 5. word
# - line_num = line number (starts from 0)
# - page_num = the index of the page where the item was detected
# - text = the recognition result
# - top = y-coordinate where the bounding box starts
# - width = width of the current detected text block
# - word_num = word number (index) within the current block
config_tesseract = '--tessdata-dir tessdata'
result = pytesseract.image_to_data(rgb, config=config_tesseract, lang='eng', output_type=Output.DICT)
result

# %% [markdown]
# - block_num = Current block number. When Tesseract performs the detections, it divides the image into several regions, which can vary according to the PSM parameters and also other criteria of the algorithm. Each block is a region
# 
# - conf = prediction confidence (from 0 to 100. -1 means no text was recognized)
# 
# - height = height of detected block of text (bounding box)
# 
# - left = x coordinate where the bounding box starts
# 
# - level = the level corresponds to the category of the detected block. There are 5 possible values:
#   1. page
#   2. block
#   3. paragraph
#   4. line
#   5. word
# 
# Therefore, if 5 is returned, it means that the detected block is text, if it was 4, it means that a line was detected
# 
# - line_num = line number (starts from 0)
# 
# - page_num = the index of the page where the item was detected
# 
# - text = the recognition result
# 
# - top = y-coordinate where the bounding box starts
# 
# - width = width of the current detected text block
# 
# - word_num = word number (index) within the current block

# %%
result['text'], len(result['text'])

# %%
# Get bounding box and draw a rectange on image
def bouding_box(result, img, i, color = (255,100,0)):
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]
    # image, first point, end point, color, thickness
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return x, y, img

# %%
# Here, we visit each detected block
# and if we have a text witha given confidence
# we draw the bbox and put the text
min_confidence = 40
img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i]) # conf is a string!
    if confidence > min_confidence:
        x, y, img = bouding_box(result, img_copy, i)
        text = result['text'][i]
        # The font must have all the symbols of the text
        # Hershey has only English symbols...
        cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (100,0,255))

cv2_imshow(img_copy)

# %%
# Image with Portuguese symbols
img = cv2.imread(DATAPATH+'Images/test02-02.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
config_tesseract = '--tessdata-dir tessdata'
result = pytesseract.image_to_data(rgb, config=config_tesseract, lang = 'por', output_type = Output.DICT)
result

# %%
from PIL import ImageFont, ImageDraw, Image
# In case we have an image with text in a language with non-English symbols
# we need to draw the text with a font which has the symbols necessary
font = DATAPATH+'Fonts/calibri.ttf'

# %%
# Custom function which replaces cv2.putText()
# and allows for any TIFF font with custom symbols to be used
def write_text(text, x, y, img, font, font_size = 32):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, font = font)
    img = np.array(img_pil)
    return img

# %%
min_confidence = 40
img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i])
    if confidence > min_confidence:
        x, y, img = bouding_box(result, img_copy, i)
        text = result['text'][i]
        #cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255))
        img_copy = write_text(text, x, y, img_copy, font)
cv2_imshow(img_copy)

# %% [markdown]
# # Searching specific information

# %%
import re # regular expressions

# %%
# After extracting the text, we can appla regex to extract specific info
img = cv2.imread(DATAPATH+'Images/table_test.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
result = pytesseract.image_to_data(rgb, config=config_tesseract, lang='por', output_type=Output.DICT)
result

# %%
# https://regexr.com/
# We want to extract date information in the format DD/MM/YYYY
# We define the regex pattern for that
date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

# %%
# This code loads a table with entries
# detects all text blocks
# and finds date blocks which match the pattern above
# Then, boxes are drawn on the image
dates = []
min_confidence = 40
img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i])
    if confidence > min_confidence:
        text = result['text'][i]
        if re.match(date_pattern, text):
            x, y, img = bouding_box(result, img_copy, i, (0,0,255))
            #cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255))
            img_copy = write_text(text, x, y, img_copy, font, 12)
            dates.append(text)
        else:
              x, y, img_copy = bouding_box(result, img_copy, i)
plt.figure(figsize=(15,15))
cv2_imshow(img_copy)

# %%
# Detected dates
dates

# %% [markdown]
# # Detecting texts in natural scenarios

# %%
# We can also detect text in natural scenarios, but
# this is more challenging for the libraries
img = cv2.imread(DATAPATH+'Images/cup.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb)

# %%
result = pytesseract.image_to_data(rgb, lang = 'eng', output_type=Output.DICT)
# We see it doesn't work - in the tutorial it works, but not here!
# However, in the tutorial, many false positives (with high conf) with empty text are returned
# It is very common to get false positives in natural scenarios.
result

# %%
min_confidence = 40
img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i])
    if confidence > min_confidence:
        text = result['text'][i]
        if not text.isspace() and len(text) > 0:
            x, y, img = bouding_box(result, img_copy, i)
            cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255))
cv2_imshow(img_copy)

# %%
result['conf']

# %%
result['text']

```

### [`02_OCR_with_Python_Pre_processing.ipynb`](./lab/02_OCR_with_Python_Pre_processing.ipynb)

```python
# %% [markdown]
# # OCR with Python - pre-processing techiniques

# %% [markdown]
# # Importing the libraries

# %%
import pytesseract
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt
import re

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATAPATH = "../../material/"

# %% [markdown]
# # Grayscale

# %%
img = cv2.imread(DATAPATH+'Images/img-process.jpg')
cv2_imshow(img)

# %%
img.shape

# %%
# Convert to grascale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
gray.shape

# %% [markdown]
# # Thresholding

# %% [markdown]
# ## Simple thresholding

# %%
img = cv2.imread(DATAPATH+'Images/page-book.jpg')
cv2_imshow(img)

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
# Thresholding: Binarization
# Simple thresholding: gray image, min-threshold, max, type
# Types of threshold
# - Simple thresholding: cv2.THRESH_BINARY: (global) we provide the threshold value
# - Otsu method: (global) value detected automatically based on histogram; a bi-modal histogram is assumed, where one peak is the BG and the other the FG, i.e., the information
# - Adaptive thresholding: different thresholds for each local region (not global), and then a Gaussian is applied
# Simple thresholding: image, thres, max-vale, cv2.THRESH_BINARY 
value, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh)

# %%
value

# %%
# If we vary the image, shadows might be detected as information
value, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh)

# %%
value, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh)

# %% [markdown]
# ## Otsu method

# %%
# The Otsu thresholding method detects the thresholding value automatically
# It assumes the image is bi-modal, where one peak is background
# and the other foreground/information.
# Otsu method: min, max, type: cv2.THRESH_BINARY | cv2.THRESH_OTSU
# However, the Otsu method is not always the best - we need to try!
value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2_imshow(otsu)
print(value)

# %%
img = cv2.imread(DATAPATH+'Images/recipe01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
value, thresh = cv2.threshold(gray, 138, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh)

# %%
value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2_imshow(otsu)
print(value)

# %% [markdown]
# ## Adaptive Thresholding

# %%
img = cv2.imread(DATAPATH+'Images/book02.jpg')
cv2_imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2_imshow(otsu)
print(value)

# %%
# Adaptive thresholding is best when we have lighting variations in the image
# Instead of computing a global threshold, local values are computed and then
# the result is consolidated either by computing the mean around each pixel
# or applying a Gaussian filter.
# Adaptive with mean: 
# - blockSize: pixel neighborhood that is used to calculate a threshold value for the pixel
# - C: Constant subtracted from the mean or weighted mean
adaptive_average = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
cv2_imshow(adaptive_average)

# %% [markdown]
# ## Gaussian Adaptive Thresholding

# %%
img = cv2.imread(DATAPATH+'Images/book_adaptative.jpg')
cv2_imshow(img)

# %%
# Adaptive with Gaussian (similar parameters as the mean)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
cv2_imshow(adaptive_gaussian)

# %%
adaptive_average = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
cv2_imshow(adaptive_average)

# %% [markdown]
# # Color inversion

# %%
# Color inversion makes sense when we have white letters with dark BG
# The usual recommendation is to have white BG and black/dark FG or letters
img = cv2.imread(DATAPATH+'Images/img-process.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
img.shape, gray.shape

# %%
gray

# %%
# Inversion: just substract the image to the maximum grayvalue
invert = 255 - gray

# %%
invert

# %%
255 - 68

# %%
cv2_imshow(invert)

# %%
cv2_imshow(thresh)

# %%
invert = 255 - thresh
cv2_imshow(invert)

# %% [markdown]
# # Resizing
# 
# - INTER_NEAREST - a nearest neighbor interpolation. It is widely used because it is the fastest
# - INTER_LINEAR - a bilinear interpolation (it's used by default), generally good for zooming in and out of images
# - INTER_AREA - uses the pixel area ratio. May be a preferred method for image reduction as it provides good results 
# - INTER_CUBIC - bicubic (4x4 neighboring pixels). It has better results
# - INTER_LANCZOS4 - Lanczos interpolation (8x8 neighboring pixels). Among these algorithms, it is the one with the best quality results.

# %%
cv2_imshow(gray)

# %%
gray.shape

# %%
# When resizing, we can increase or decrease the size, and in any direction.
# If the factor is > 1.0, we are increasing the size.
# Usually, the bigger the image, the better for OCR
# If we increase the size, we need to interpolate empty spaces; we have different methods:
# - INTER_NEAREST - a nearest neighbor interpolation. It is widely used because it is the fastest
# - INTER_LINEAR - a bilinear interpolation (it's used by default), generally good for zooming in and out of images
# - INTER_AREA - uses the pixel area ratio. May be a preferred method for image reduction as it provides good results 
# - INTER_CUBIC - bicubic (4x4 neighboring pixels). It has better results
# - INTER_LANCZOS4 - Lanczos interpolation (8x8 neighboring pixels). Among these algorithms, it is the one with the best quality results.
increase = cv2.resize(gray, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
cv2_imshow(increase)

# %%
increase.shape

# %%
decrease = cv2.resize(gray, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)
cv2_imshow(decrease)

# %%
decrease.shape

# %% [markdown]
# # Morphological operations

# %% [markdown]
# ## Erosion

# %%
img = cv2.imread(DATAPATH+'Images/text-opencv.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
np.ones((3,3), np.uint8)

# %%
# Erosion: remove pixels with a kernel
# The size of the kernel depends on the size of the noise/spots
# Small spots (smaller than the kernel) are removed, but everything is thinned
erosion = cv2.erode(gray, np.ones((3,3), np.uint8))
cv2_imshow(erosion)

# %% [markdown]
# ## Dilation

# %%
# Dilation: Add pixels with a kernel
# The size of the kernel depends on the size of the noise/spots
# Small spots are emphasized
dilation = cv2.dilate(gray, np.ones((3,3), np.uint8))
cv2_imshow(dilation)

# %% [markdown]
# ## Opening

# %%
# Opening: Erosion + Dilation: salt/pepper noise is removed while maintaining thickness
erosion = cv2.erode(gray, np.ones((5,5), np.uint8))
opening = cv2.dilate(erosion, np.ones((5,5), np.uint8))
cv2_imshow(gray)
cv2_imshow(erosion)
cv2_imshow(opening)

# %% [markdown]
# ## Closing

# %%
# Closing: Dilation + Erosion:
# Holes are filled/closed while maintaining thickness
img = cv2.imread(DATAPATH+'Images/text-opencv2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
dilation = cv2.dilate(gray, np.ones((5,5), np.uint8))
closing = cv2.erode(dilation, np.ones((5,5), np.uint8))
cv2_imshow(gray)
cv2_imshow(dilation)
cv2_imshow(closing)

# %% [markdown]
# # Noise removal

# %% [markdown]
# ## Average blur

# %%
# We can use filters to reduce noise or emphasize signals:
# - Lowpass filters remove noise but blur the image
# - Highpass filters emphasize the signal, but also the noise
img = cv2.imread(DATAPATH+'Images/test_noise.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
# Agerage/mean filtering
average_blur = cv2.blur(gray, (5,5))
cv2_imshow(average_blur)

# %% [markdown]
# ## Gaussian blur
# 

# %%
# Gaussian filter
gaussian_blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2_imshow(gaussian_blur)

# %% [markdown]
# ## Median blur

# %%
# Median blur
median_blur = cv2.medianBlur(gray, 3)
cv2_imshow(median_blur)

# %% [markdown]
# ## Bilateral filter

# %%
# Bilateral filter: noise is reduced, while preserving the edges
# Look at the documentation for more information on the parameters
bilateral_filter = cv2.bilateralFilter(gray, 15, 55, 45)
cv2_imshow(bilateral_filter)

# %% [markdown]
# # Text detection

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract

# %%
import pytesseract

# %%
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true
#!wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true

# %%
config_tesseract = '--tessdata-dir tessdata'
text = pytesseract.image_to_string(average_blur, lang = 'por', config=config_tesseract)
#text = pytesseract.image_to_string(average_blur, lang = 'por')
print(text)

# %%
# We need to try different processed images to see which one is better for OCR
text = pytesseract.image_to_string(median_blur, lang = 'por', config=config_tesseract)
print(text)

# %% [markdown]
# # Homework

# %%
img = cv2.imread(DATAPATH+'Images/sentence.jpg')
cv2_imshow(img)

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %%
value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2_imshow(thresh)
print(value)

# %%
invert = 255 - thresh
cv2_imshow(invert)

# %%
print(pytesseract.image_to_string(thresh, lang = 'por', config=config_tesseract))

```

### [`03_OCR_with_Python_Text_detection_with_EAST.ipynb`](./lab/03_OCR_with_Python_Text_detection_with_EAST.ipynb)

```python
# %% [markdown]
# # Text detection with EAST
# 
# - Original paper: https://arxiv.org/pdf/1704.03155v2.pdf

# %% [markdown]
# # Importing the libraries

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# PyImageSearch packages for image processing utilities
from imutils.object_detection import non_max_suppression

#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %% [markdown]
# # Connecting to Google Drive

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
#!cp /content/drive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Models/frozen_east_text_detection.pb ./

# %%
#!cp -R /content/drive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Images images/

# %%
DATAPATH = "../../material/"

# %% [markdown]
# # Pre-processing the image

# %%
# EAST is a FCN which detects regions wich might likely contain text.
# It works robustly with natural scenes.
# We obtain a score map and candidate bouding boxes set in the map
# Then, we need to apply non-maximum supression (NMS) to narrow down the bboxes.
detector = DATAPATH+'Models/frozen_east_text_detection.pb'
# We need to resize the image to the input size of the neural network
# EAST works with sizes which are multiples of 320
width, height = 320, 320
image = DATAPATH+'Images/cup.jpg'
# Minimum confidence for our text bboxes.
min_confidence = 0.9

# %%
img = cv2.imread(image)
cv2_imshow(img)

# %%
# Since we resize, the image, we keep a copy of the org
original = img.copy()

# %%
img.shape

# %%
H = img.shape[0]
W = img.shape[1]
print(H, W)

# %%
# Save propostions/ratios to draw/display properly later only
proportion_W = W / float(width)
proportion_H = H / float(height)
print(proportion_W, proportion_H)

# %%
img = cv2.resize(img, (width, height))
H = img.shape[0]
W = img.shape[1]
print(H, W)

# %%
cv2_imshow(img)

# %% [markdown]
# # Loading the neural network

# %% [markdown]
# ![EAST Architecture](../../assets/EAST_architecture.jpg)

# %%
# We create a list with layer names
# that refer to the score map (Sigmoid) and the bounding boxes (concat_3)
layers_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

# %%
# Load the detector network
neural_network = cv2.dnn.readNet(detector)

# %%
# Resized CV2 image (array)
img.shape

# %%
# Convert image array to a BLOB type
blob = cv2.dnn.blobFromImage(img,
                             scalefactor=1.0,
                             size=(W, H),
                             swapRB=True,
                             crop=False)

# %%
blob.shape # batch size

# %%
# Input the BLOB/image
neural_network.setInput(blob)
# Get the output: for that we need to spcify the output layers
scores, geometry = neural_network.forward(layers_names)

# %%
# We get a 80x80 map with scores/confidences
scores.shape

# %%
# Each cell in the 80x80 map has a bbox associated
geometry.shape

# %%
scores

# %%
geometry

# %%
geometry[0,0]

# %% [markdown]
# # Decoding the values
# 
# The EAST model takes a 320x320 RGB image (or a multiple of that size) and returns a 80x80 (1) score/confidence map and a (2) map with the bounding box values. The decoding of the bounding boxes is explained in the paper and in [this Stackoverflow post](https://stackoverflow.com/questions/55583306/decoding-geometry-output-of-east-text-detection).
# ![EAST Bounding Boxes](../../assets/EAST_bboxes.jpg)g)

# %%
# Confidences
scores.shape[2:4]

# %%
# Bounding boxes
geometry.shape

# %%
# Get loop ranges
# Note: Y, X = row, column
rows, columns = scores.shape[2:4]
print(rows, columns)

# %%
# Initialize return lists
boxes = []
confidences = []

# %%
def geometric_data(geometry, y):
    """Extract the BBox data given a row:
    angles, x0, x1, x2, x3."""
    xData0 = geometry[0, 0, y] # complete rows of lenth 80!
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]
    
    return angles_data, xData0, xData1, xData2, xData3

# %%
def geometric_calculation(x, angles_data, xData0, xData1, xData2, xData3):
    """Given the BBox data in each row,
    get the start-end BBox coordinates.
    """
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    angle = angles_data[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]
    
    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    
    beginX = int(endX - w)
    beginY = int(endY - h)

    # FIXME: it seems that the code accounts for oriented BBoxes
    # but here the orientation is neglected...
    return beginX, beginY, endX, endY

# %%
rows, columns

# %%
scores

# %%
# Get loop ranges
# Note: Y, X = row, column
rows, columns = scores.shape[2:4]
print(rows, columns)

# Initialize return lists
boxes = []
confidences = []

for y in range(0, rows):
    data_scores = scores[0, 0, y]
    # Extract data in each row
    angles_data, xData0, xData1, xData2, xData3 = geometric_data(geometry, y)
    for x in range(0, columns):
        if data_scores[x] < min_confidence: # 0.9
            continue
        # For each row, get all bboxes with high confidence
        beginX, beginY, endX, endY = geometric_calculation(x, angles_data, xData0, xData1, xData2, xData3)
        confidences.append(data_scores[x])
        boxes.append((beginX, beginY, endX, endY))

# %%
confidences

# %%
# We see that several bounding boxes were detected
# for the same text.
# We need to apply non-maximal supression (NMS)
boxes

# %%
# Non-maximal supression (NMS)
detections = non_max_suppression(np.array(boxes), probs = confidences)

# %%
# Final bounding box
detections

# %%
proportion_H, proportion_W

# %%
# We need to expand the bounding box to the original image size
# To that end, we used the pre-saved ratios
img_copy = original.copy()
for (beginX, beginY, endX, endY) in detections:
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)
    
    # Region of interest
    roi = img_copy[beginY:endY, beginX:endX]
    
    cv2.rectangle(original, (beginX, beginY), (endX, endY), (0,255,0), 2)
cv2_imshow(original)

# %%
cv2_imshow(roi)

# %%
# We increase the size of the ROI by 50%
roi = cv2.resize(roi, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
cv2_imshow(roi)

# %% [markdown]
# # Text recognition

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract 
import pytesseract

# %%
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true
#!wget -O ./tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true

# %%
!tesseract --help-psm

# %%
# Now we recognize the text in each ROI we detected
# Since the ROI has one line we choose PSM 7
# However, note that it is recommendable to expand the ROI some pixels
config_tesseract = "--tessdata-dir tessdata --psm 7"

# %%
img_copy = original.copy()
for (beginX, beginY, endX, endY) in detections:
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)
    
    roi = img_copy[beginY:endY, beginX:endX]
    text = pytesseract.image_to_string(roi, lang = 'eng', config=config_tesseract)
    print(text)
    
    cv2.rectangle(original, (beginX, beginY), (endX, endY), (0,255,0), 2)
cv2_imshow(original)

# %% [markdown]
# ## Expanding the ROI

# %%
margin = 3
img_copy = original.copy()
for (beginX, beginY, endX, endY) in detections:
    beginX = int(beginX * proportion_W)
    beginY = int(beginY * proportion_H)
    endX = int(endX * proportion_W)
    endY = int(endY * proportion_H)
    
    roi = img_copy[beginY - margin:endY + margin, beginX - margin:endX + margin]
    cv2_imshow(roi)
    text = pytesseract.image_to_string(roi, lang = 'eng', config=config_tesseract)
    print(text)
    
    cv2.rectangle(img_copy, (beginX - margin, beginY - margin), (endX + margin, endY + margin), (0,255,0), 2)
cv2_imshow(img_copy)

```


### [`04_OCR_in_Videos.ipynb`](./lab/04_OCR_in_Videos.ipynb)

```python
# %% [markdown]
# # OCR in videos with Tesseract and EAST

# %% [markdown]
# # Importing the libraries

# %%
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
from PIL import Image
from PIL import ImageFont, ImageDraw, Image

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATA_PATH = "../../material/"

# %% [markdown]
# # Connecting with Google Drive

# %%
#from google.colab import drive
#drive.mount('/content/gdrive')

# %% [markdown]
# # Tesseract

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract 
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

# %%
import pytesseract

# %%
config_tesseract = "--tessdata-dir tessdata --psm 7"

# %%
def tesseract_OCR(img, config_tesseract):
    text = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
    return text

# %% [markdown]
# # Pre-processing

# %%
def pre_processing(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale image: 2x size
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Apply Otsu thresholding
    value, otsu = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return otsu

# %% [markdown]
# # EAST

# %%
#!cp /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Models/frozen_east_text_detection.pb ./
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Images images/
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Fonts fonts/
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Videos videos/

# %%
# EAST params
detector = DATA_PATH + "Models/frozen_east_text_detection.pb"
# EAST works with sizes which are multiples of 320
height_EAST, width_EAST = 640, 640 # 320x320

# %%
min_conf_EAST = 0.9

# %%
layers_EAST = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# %% [markdown]
# # Functions - EAST

# %%
def geometric_data(geometry, y):
    # Extract row-wise geometry data
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    angles_data = geometry[0, 4, y]
    
    return angles_data, xData0, xData1, xData2, xData3

# %%
def geometric_calculation(angles_data, xData0, xData1, xData2, xData3, x, y):
    # Compute a bounding box from a row-geometry data
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    angle = angles_data[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]
    
    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    
    beginX = int(endX - w)
    beginY = int(endY - h)
    
    return beginX, beginY, endX, endY

# %% [markdown]
# # Function to pre-process the image

# %%
def EAST_preprocessing(img, width, height, network, layer_names, min_confidence):
    # This function takes an image and a pre-trained EAST model
    # and obtains all the text bounding boxes in th eimage
    original = img.copy()
    (H, W) = img.shape[:2]
    
    proportion_W = W / float(width)
    proportion_H = H / float(height)
    
    img = cv2.resize(img, (width, height))
    (H, W) = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), swapRB=True, crop=False)
    
    network.setInput(blob)
    (scores, geometry) = network.forward(layer_names)
    
    (rows, columns) = scores.shape[2:4]
    
    boxes = []
    confidences = []
    
    for y in range(0, rows):
        data_scores = scores[0, 0, y]
        data_angles, x0_data, x1_data, x2_data, x3_data = geometric_data(geometry, y)
    
    for x in range(0, columns):
        if data_scores[x] < min_confidence:
            continue
        
        beginX, beginY, endX, endY = geometric_calculation(data_angles, x0_data, x1_data, x2_data, x3_data, x, y)
        confidences.append(data_scores[x])
        boxes.append((beginX, beginY, endX, endY))
    
    return proportion_W, proportion_H, confidences, boxes

# %% [markdown]
# # Function to write on the video

# %%
font = DATA_PATH+'Fonts/calibri.ttf'

# %%
def write_text(text, x, y, img, font, color=(50, 50, 255), font_size=22):
    # Write a text on the image
    # This function is used by text_background()
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img) 
    draw = ImageDraw.Draw(img_pil) 
    draw.text((x, y-font_size), text, font = font, fill = color) 
    img = np.array(img_pil) 
    
    return img

# %%
def text_background(text, x, y, img, font, font_size=32, color=(200, 255, 0)):
    # Place a box with a text on the image
    background = np.full((img.shape), (0,0,0), dtype=np.uint8)
    text_back = write_text(text, x, y, background, font, (255,255,255), font_size=font_size)
    text_back = cv2.dilate(text_back,(np.ones((3,5),np.uint8)))
    fx,fy,fw,fh = cv2.boundingRect(text_back[:,:,2])
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, -1)
    
    return img

# %% [markdown]
# # Loding EAST

# %%
EASTnet = cv2.dnn.readNet(detector)

# %% [markdown]
# # Loading the video file 

# %%
video_file = DATA_PATH+'Videos/test02.mp4'
capture = cv2.VideoCapture(video_file)
# Check that the video was captured
connected, video = capture.read()

# %%
connected

# %%
# A video is a collection of frames
video

# %%
# h, w, c
video.shape

# %%
video_width = video.shape[1]
video_height = video.shape[0]
print(video_width, video_height)

# %% [markdown]
# # Resizing the video

# %%
def resize_video(width, height, max_width = 600):
    # The size of each frame is very large, we should decrease it
    # to be able to run a OCR online/in time
    if width > max_width:
        proportion = width / height
        video_width = max_width
        video_height = int(video_width / proportion)
    else:
        video_width = width
        video_height = height
    return video_width, video_height

# %%
video_width, video_height = resize_video(video.shape[1], video.shape[0], 800)
print(video_width, video_height)

# %%
# Ratio
1284 / 720

# %%
800 / 1.78

# %% [markdown]
# # Video settings

# %%
# We want to save the resulting video with detected text bboxes
# https://www.programcreek.com/python/example/89348/cv2.VideoWriter_fourcc
name_video_file = 'result_east_tesseract.avi'
# Codec to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
output_video = cv2.VideoWriter(name_video_file, fourcc, fps, (video_width, video_height))

# %% [markdown]
# # Processing the video

# %%
# We process 20 frames and show one
show_frames = 20
current_frame = 0
# Margin added around the image
margin = 4

while (cv2.waitKey(1) < 0):
    # Get a frame from capture, whcih contains the video
    connected, frame = capture.read()
    
    if not connected:
        break

    # Resize to smalled size
    frame = cv2.resize(frame, (video_width, video_height))

    # Copy the image and pass it to EAST_preprocessing(), which
    # - processes the image (resize, thresholding, etc.)
    # - detects text strings using Tesseract
    image_cp = frame.copy()
    proportion_W, proportion_H, confidences, boxes = EAST_preprocessing(frame,
                                                                        width_EAST,
                                                                        height_EAST,
                                                                        EASTnet,
                                                                        layers_EAST,
                                                                        min_conf_EAST)
    # Reduce the number of bounding boxes with NMS
    detections = non_max_suppression(np.array(boxes), probs=confidences)
    
    for (beginX, beginY, endX, endY) in detections:
        # Loop through all detections
        # Get all bboxes and expand them to the original size
        beginX = int(beginX * proportion_W)
        beginY = int(beginY * proportion_H)
        endX = int(endX * proportion_W)
        endY = int(endY * proportion_H)
        # Extract the ROI of each of them
        cv2.rectangle(frame, (beginX, beginY), (endX, endY), (200,255,0), 2)
        roi = image_cp[beginY - margin:endY + margin, beginX - margin:endX + margin]
        # Perform OCR with the ROI
        processed_img = pre_processing(roi)
        text = tesseract_OCR(processed_img, config_tesseract)
        print(text)
        # Get the text and plot it with a box on image
        # http://www.asciitable.com/
        text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
        frame = text_background(text, beginX, beginY, frame, font, 20, (200,255,0))
        frame = write_text(text, beginX, beginY, frame, font, (0,0,0), 20)

    # Display result if necessary
    if current_frame <= show_frames:
        cv2_imshow(frame)
        current_frame += 1
    
    output_video.write(frame)

print('Finished!')
output_video.release()
cv2.destroyAllWindows()

# %% [markdown]
# # OCR in videos with EasyOCR

# %%
#!pip install easyocr

# %%
#!pip uninstall opencv-python-headless
#!pip install opencv-python-headless==4.1.2.30

# %%
from easyocr import Reader
import cv2
#from google.colab.patches import cv2_imshow
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# %%
languages_list = ['en','pt']
gpu = False # True 
font = DATA_PATH+'Fonts/calibri.ttf' 

# %%
video_file = DATA_PATH+"Videos/test02.mp4"
cap = cv2.VideoCapture(video_file)

connected, video = cap.read()
video_width = video.shape[1]
video_height = video.shape[0]

# %%
video_width, video_height = resize_video(video.shape[1], video.shape[0], 800)
print(video_width, video_height)

# %%
def box_coordinates(box):
    (lt, rt, br, bl) = box
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    return lt, rt, br, bl

# %%
def draw_img(img, lt, br, color=(200,255,0),thickness=2):
    cv2.rectangle(img, lt, br, color, thickness)
    
    return img

# %%
name_video_file = 'result_easy_ocr.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
output_video = cv2.VideoWriter(name_video_file, fourcc, fps, (video_width, video_height))

# %%
show_frames = 20
current_frame = 0
margin = 4

while (cv2.waitKey(1) < 0):
    connected, frame = cap.read()
    
    if not connected:
        break
    
    frame = cv2.resize(frame, (video_width, video_height))
    
    image_cp = frame.copy()
    reader = Reader(languages_list, gpu=gpu)
    results = reader.readtext(frame)
    for (box, text, prob) in results:
        lt, rt, br, bl = box_coordinates(box)
        frame = draw_img(frame, lt, br)
        frame = text_background(text, lt[0], lt[1], frame, font, 20, (200,255,0))
        frame = write_text(text, lt[0], lt[1], frame, font, (0,0,0), 20)
    
    if current_frame <= show_frames:
        cv2_imshow(frame)
        current_frame += 1
    
    output_video.write(frame)

print('Finished!')
output_video.release()
cv2.destroyAllWindows()

```
