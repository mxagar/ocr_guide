# OCR Guide: Projects

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

This sub-folder contains interesting projects and use-cases.

Table of contents:

- [OCR Guide: Projects](#ocr-guide-projects)
  - [Project 1: Searching Terms in Documents](#project-1-searching-terms-in-documents)
  - [Project 2: Receipt Scanning and OCR](#project-2-receipt-scanning-and-ocr)
  - [Project 3: License Plate Detection](#project-3-license-plate-detection)

## Project 1: Searching Terms in Documents

Notebook: [`OCR_Project_01_Searching_for_specific_terms.ipynb`](./OCR_Project_01_Searching_for_specific_terms.ipynb).

In this project, some book (aligned) pages are used as target images; in them:

- All the text is recognized.
- A word cloud is created using spacy and wordcloud.
- An app is built to find specific terms and highlight them on the images.

```python
# %% [markdown]
# # OCR Project 1 - Searching for specific terms

# %% [markdown]
# # Installing Tesseract

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract 

# %%
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

# %% [markdown]
# # Importing the libraries

# %%
import pytesseract
from pytesseract import Output
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt 
from PIL import ImageFont, ImageDraw, Image

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATAPATH = "./../material/"

# %% [markdown]
# # Connecting to Google Drive and visualizing the images

# %%
#from google.colab import drive
#drive.mount('/content/gdrive')

# %%
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Images/Images\ Project\ 1/ images/

# %%
directory_imgs = DATAPATH+"Images/Images Project 1"
paths = [os.path.join(directory_imgs, f) for f in os.listdir(directory_imgs)]
print(paths)

# %%
def show_img(img):
  fig = plt.gcf()
  fig.set_size_inches(20, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

# %%
for image in paths:
  image = cv2.imread(image) 
  show_img(image)

# %% [markdown]
# # Tesseract settings

# %%
config_tesseract = "--tessdata-dir tessdata"

# %%
def tesseract_ocr(img, config_tesseract):
    text = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
    return text

# %% [markdown]
# # Text recognition

# %%
full_text = ''
txt_file = 'results_ocr.txt'

# %%
for image in paths:
  #print(image)
  img = cv2.imread(image)
  file_image = os.path.split(image)[-1]
  #print(file_image)
  file_image_separate = '================\n' + str(file_image)
  #print(file_image_separate)
  full_text = full_text + file_image_separate + '\n'

  text = tesseract_ocr(img, config_tesseract)
  #print(text)
  full_text = full_text + text

# %%
full_text

# %% [markdown]
# ## Saving the results

# %%
file_txt = open(txt_file, 'w+')
file_txt.write(full_text + '\n')
file_txt.close()

# %% [markdown]
# ## Searching in the .txt file

# %%
term_search = 'computador' # computer

# %%
with open('./results_ocr.txt') as f:
    results = [i.start() for i in re.finditer(term_search, f.read())]

# %%
results

# %%
len(results)

# %% [markdown]
# ## Searching in the images

# %%
for image in paths:
    #print(image)
    img = cv2.imread(image)
    file_img = os.path.split(image)[-1]
    print('==================\n' + str(file_img))
    text = tesseract_ocr(img, config_tesseract)
    results = [i.start() for i in re.finditer(term_search, text)]
    print('Number of times the term {} appears: {}'.format(term_search, len(results)))
    print('\n')

# %% [markdown]
# ## Word cloud

# %%
# https://spacy.io
import spacy

# %% [markdown]
# > Update in 2023: for more recent spacy versions, now it's necessary to change the name of the package, from:
#   * `pt` to `pt_core_news_sm`
#   * `en` to `en_core_web_sm`
# 
# > This change is required in the *download command* below and also in the `spacy.load()` parameter

# %%
!python -m spacy download pt_core_news_sm

# %%
!python -m spacy download en_core_web_sm

# %%
nlp_en = spacy.load('en_core_web_sm')

# %%
print(spacy.lang.en.stop_words.STOP_WORDS)

# %%
len(spacy.lang.en.stop_words.STOP_WORDS)

# %%
nlp = spacy.load('pt_core_news_sm')  

# %%
stop_words = spacy.lang.pt.stop_words.STOP_WORDS
print(stop_words)

# %%
len(stop_words)

# %%
def preprocessing(text):
    text = text.lower()
    
    document = nlp(text)
    tokens_list = []
    for token in document:
        #print(token)
        tokens_list.append(token.text)
        #print(tokens_list)
    
    tokens = [word for word in tokens_list if word not in stop_words]
    #print(tokens)
    tokens = ' '.join([str(element) for element in tokens])
    #print(tokens)
    return tokens

# %%
preprocessing('Note que, se a máscara for simétrica as operações de correlação')

# %%
processed_full_text = preprocessing(full_text)

# %%
len(full_text), len(processed_full_text)

# %%
9586 - 8944

# %%
from wordcloud import WordCloud
plt.figure(figsize=(20,10))
plt.imshow(WordCloud().generate(full_text)); # of, what, the -> stopwords

# %%
from wordcloud import WordCloud
plt.figure(figsize=(20,10))
plt.imshow(WordCloud().generate(processed_full_text));

# %% [markdown]
# ## Named entity recognition
# 
# - Documentation: https://spacy.io/api/annotation#named-entities

# %%
document = nlp(processed_full_text)

# %%
from spacy import displacy
displacy.render(document, style = 'ent', jupyter = True)

# %%
for entity in document.ents:
    if entity.label_ == 'PER':
        print(entity.text, entity.label_)

# %% [markdown]
# # Text recognition in the images

# %% [markdown]
# ## Function to write in the images

# %%
font = DATAPATH+'Fonts/calibri.ttf'

# %%
def write_text(text, x, y, img, font, color=(50, 50, 255), font_size=16):
    font = ImageFont.truetype(font, font_size)
    img_pil = Image.fromarray(img) 
    draw = ImageDraw.Draw(img_pil) 
    draw.text((x, y-font_size), text, font = font, fill = color) 
    img = np.array(img_pil) 
    
    return img

# %% [markdown]
# ## Function to show the detections

# %%
min_conf = 30

# %%
def box(i, result, img, color=(255, 100, 0)):
    x = result["left"][i]
    y = result["top"][i]
    w = result["width"][i]
    h = result["height"][i]
    
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    
    return x, y, img

# %%
# computer - Computer

# %%
def ocr_process_image(img, term_search, config_tesseract, min_conf):
  result = pytesseract.image_to_data(img, config=config_tesseract, lang='por', output_type=Output.DICT)
  number_of_times = 0
  for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i])
    if confidence > min_conf:
      text = result['text'][i]
      if term_search.lower() in text.lower():
        x, y, img = box(i, result, img, (0,0,255))
        img = write_text(text, x, y, img, font, (50,50,225), 14)
        number_of_times += 1
  return img, number_of_times

# %% [markdown]
# ## Searching for specific terms

# %%
term_search = 'computador' # computer
for image in paths:
    #print(image)
    img = cv2.imread(image)
    img_original = img.copy()
    file_image = os.path.split(image)[-1]
    print('=================\n' + str(file_image))
    
    img, number_of_times = ocr_process_image(img, term_search, config_tesseract, min_conf)
    print('Number of times term {} appears in {}: {}'.format(term_search, file_image, number_of_times))
    print('\n')
    show_img(img)

# %% [markdown]
# ## Saving the results

# %%
term_search = 'sopa' # soup

# %%
os.makedirs('processed_images', exist_ok = True)

# %%
for image in paths:
    #print(image)
    img = cv2.imread(image)
    img_original = img.copy()
    file_image = os.path.split(image)[-1]
    img, number_of_times = ocr_process_image(img, term_search, config_tesseract, min_conf)
    if number_of_times > 0:
        show_img(img)
        new_file_image = 'processed_' + file_image
        new_image = './processed_images/' + str(new_file_image)
        cv2.imwrite(new_image, img)


```

## Project 2: Receipt Scanning and OCR

Notebook: [`OCR_Project_02_Scanner_OCR.ipynb`](./OCR_Project_02_Scanner_OCR.ipynb).

In this project, some misaligned images are used: a receipt and a book cover. With them:

- The edges and contours are detected to pick the boders of the paper with the text.
- The image is warped and croped using the borders.
- The aligned image is processed: brightness and contrast adjusted, thresholding applied.
- OCR is applied to the final processed image.

```python
# %% [markdown]
# # OCR Project 2: Scanner + OCR

# %% [markdown]
# # Importing the libraries

# %%
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATAPATH = "./../material/"

# %% [markdown]
# # Connecting to Google Drive

# %%
#from google.colab import drive
#drive.mount('/content/gdrive')

# %%
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Images/Images\ Project\ 2 images/ 

# %%
def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# %% [markdown]
# # Pre-processing the image

# %%
directory_imgs = DATAPATH+"Images/Images Project 2"
img = cv2.imread(directory_imgs+'/doc_rotated01.jpg')
original = img.copy()
show_img(img)
(H, W) = img.shape[:2]
print(H, W)

# %% [markdown]
# ## Grayscale

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(gray)

# %% [markdown]
# ## Gaussian Blur

# %%
blur = cv2.GaussianBlur(gray, (5, 5), 0)
show_img(blur)

# %% [markdown]
# ## Border detection (*Canny Edge*)

# %%
edged = cv2.Canny(blur, 60, 160)
show_img(edged)

# %% [markdown]
# # Contours detection

# %%
def find_contours(img): # EXTERNAL
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key = cv2.contourArea, reverse = True)[:6]
    return conts

# %%
conts = find_contours(edged.copy())

# %%
conts

# %% [markdown]
# ## Locatting the biggest contour
# 
# - Douglas-Peucker: http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

# %%
from re import A
for c in conts:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approximation) == 4:
        larger = approximation
        break

# %%
larger

# %%
cv2.drawContours(img, larger, -1, (120,255,0), 28)
cv2.drawContours(img, [larger], -1, (120,255,0), 2)
show_img(img)

# %% [markdown]
# ## Sorting the points
# 
# - top left
# - top right
# - bottom right
# - bottom left

# %%
def sort_points(points):
    points = points.reshape((4,2))
    #print(points.shape)
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    #print(new_points.shape)
    #print(new_points)
    add = points.sum(1)
    #print(add)
    
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    dif = np.diff(points, axis = 1)
    new_points[1] = points[np.argmin(dif)]
    new_points[3] = points[np.argmax(dif)]
    #print(new_points)
    
    return new_points

# %%
larger

# %%
192 + 78, 108 + 940

# %%
points_larger = sort_points(larger)
print(points_larger)

# %% [markdown]
# ## Transformation matrix

# %%
pts1 = np.float32(points_larger)
pts1

# %%
print(W, H)

# %%
pts2 = np.float32([[0,0], [W, 0], [W, H], [0, H]])
pts2

# %%
matrix = cv2.getPerspectiveTransform(pts1, pts2)
matrix

# %% [markdown]
# ## Perspective transformation

# %%
transform = cv2.warpPerspective(original, matrix, (W, H))
show_img(transform)

# %% [markdown]
# # OCR with Tesseract

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract 

# %%
import pytesseract

# %%
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

# %%
config_tesseract = "--tessdata-dir tessdata"

# %%
text = pytesseract.image_to_string(transform, lang='por', config=config_tesseract)
print(text)

# %%
increase = cv2.resize(transform, None, fx=1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
show_img(increase)

# %%
text = pytesseract.image_to_string(increase, lang='por', config=config_tesseract)
print(text)

# %% [markdown]
# # Improving image quality

# %%
show_img(transform)

# %% [markdown]
# ## Increasing brightness and contrast

# %%
transform.shape

# %%
brightness = 50
contrast = 80
adjust = np.int16(transform)
adjust.shape

# %%
adjust = adjust * (contrast / 127 + 1) - contrast + brightness
adjust = np.clip(adjust, 0, 255)
adjust = np.uint8(adjust)
show_img(adjust)

# %% [markdown]
# ## Adaptive thresholding

# %%
processed_img = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
show_img(processed_img)

# %% [markdown]
# ## Removing edges

# %%
margin = 18
img_edges = processed_img[margin:H - margin, margin:W - margin]
show_img(img_edges)

# %% [markdown]
# ## Comparison

# %%
fig, im = plt.subplots(2, 2, figsize=(15,12))
for x in range(2):
    for y in range(2):
        im[x][y].axis('off')
im[0][0].imshow(original)
im[0][1].imshow(img)
im[1][0].imshow(transform, cmap='gray')
im[1][1].imshow(img_edges, cmap='gray')
plt.show();

# %% [markdown]
# # Putting all together

# %%
def transform_image(image_file):
  img = cv2.imread(image_file)
  original = img.copy()
  show_img(img)
  (H, W) = img.shape[:2]

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (7, 7), 0)
  edged = cv2.Canny(blur, 60, 160)
  show_img(edged)
  conts = find_contours(edged.copy())
  for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(aprox) == 4:
      larger = aprox
      break

  cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
  cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)
  show_img(img)

  points_larger = sort_points(larger)
  pts1 = np.float32(points_larger)
  pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

  matrix = cv2.getPerspectiveTransform(pts1, pts2)
  transform = cv2.warpPerspective(original, matrix, (W, H))

  show_img(transform)
  return transform

# %%
def process_img(img):
  processed_img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
  processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
  return processed_img

# %%
directory_imgs = DATAPATH+"Images/Images Project 2"
img = transform_image(directory_imgs+'/book1.jpg')
img = process_img(img)
show_img(img)

```

## Project 3: License Plate Detection

Notebook: [`OCR_Project_03_Licence_plate_reading.ipynb`](./OCR_Project_03_Licence_plate_reading.ipynb).

In this project, car license plates are recognized:

- The image is processed to detect the contour that might be the border of the plate.
- The plate ROI is cropped.
- We can apply further processing to the ROI: thresholding, etc.
- We apply OCR with Tesseract to the ROI.
- We display the result.

```python
# %% [markdown]
# # OCR Project 03: Licence plate reading
# 

# %% [markdown]
# # Installing the libraries

# %%
#!sudo apt install tesseract-ocr
#!pip install pytesseract 
#!mkdir tessdata
#!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

# %% [markdown]
# # Connection to Google Drive

# %%
#from google.colab import drive
#drive.mount('/content/gdrive')

# %%
#!cp -R /content/gdrive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Images/Images\ Project\ 3 images/ 

# %% [markdown]
# # Importando as bibliotecas

# %%
import cv2
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt

# %%
def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# %%
DATAPATH = "./../material/"

# %% [markdown]
# # Pre-processing the image

# %%
directory_imgs = DATAPATH+"Images/Images Project 3"
img = cv2.imread(directory_imgs+'/car1.jpg')
(H, W) = img.shape[:2]
print(H, W)

# %%
show_img(img)

# %% [markdown]
# ## Grayscale

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(gray)

# %% [markdown]
# ## Blur

# %%
blur = cv2.bilateralFilter(gray, 11, 17, 17)
show_img(blur)

# %% [markdown]
# ## Edges (Canny Edge)

# %%
edged = cv2.Canny(blur, 30, 200) 
show_img(edged)

# %% [markdown]
# ## Contours

# %%
conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts) 
conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8] 

# %%
conts

# %% [markdown]
# ## Finding the region

# %%
location = None
for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
    if cv2.isContourConvex(aprox):
      if len(aprox) == 4:
          location = aprox
          break

# %%
location

# %%
mask = np.zeros(gray.shape, np.uint8) 

# %%
mask.shape

# %%
mask

# %%
img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
show_img(mask)

# %%
img_plate = cv2.bitwise_and(img, img, mask=mask)

# %%
show_img(img_plate)

# %%
(y, x) = np.where(mask==255)
(beginX, beginY) = (np.min(x), np.min(y))
(endX, endY) = (np.max(x), np.max(y))

# %%
beginX, beginY, endX, endY

# %%
plate = gray[beginY:endY, beginX:endX]

# %%
show_img(plate)

# %% [markdown]
# # Text recognition

# %%
config_tesseract = "--tessdata-dir tessdata --psm 6"

# %%
!tesseract --help-psm

# %%
text = pytesseract.image_to_string(plate, lang="por", config=config_tesseract)
print(text)

# %%
text

# %%
text = "".join(character for character in text if character.isalnum())
text

# %%
img_final = cv2.putText(img, text, (beginX, beginY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,255,0), 2, lineType=cv2.LINE_AA)
img_final = cv2.rectangle(img, (beginX, beginY), (endX, endY), (150, 255, 0), 2)
show_img(img_final)

# %%
def detect_plate(file_img):
  img = cv2.imread(file_img)
  (H, W) = img.shape[:2]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.bilateralFilter(gray, 11, 17, 17)
  edged = cv2.Canny(blur, 30, 200)
  show_img(edged)
  conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  conts = imutils.grab_contours(conts) 
  conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8] 

  location = None
  for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
    if cv2.isContourConvex(aprox):
      if len(aprox) == 4:
          location = aprox
          break

  beginX = beginY = endX = endY = None
  if location is None:
    plate = False
  else:
    mask = np.zeros(gray.shape, np.uint8) 

    img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
    img_plate = cv2.bitwise_and(img, img, mask=mask)

    (y, x) = np.where(mask==255)
    (beginX, beginY) = (np.min(x), np.min(y))
    (endX, endY) = (np.max(x), np.max(y))

    plate = gray[beginY:endY, beginX:endX]
    show_img(plate)

  return img, plate, beginX, beginY, endX, endY

# %%
def ocr_plate(plate):
  config_tesseract = "--tessdata-dir tessdata --psm 6"
  text = pytesseract.image_to_string(plate, lang="por", config=config_tesseract)
  text = "".join(c for c in text if c.isalnum())
  return text

# %%
def recognize_plate(file_img):
  img, plate, beginX, beginY, endX, endY = detect_plate(file_img)
  
  if plate is False:
    print("It was not possible to detect!")
    return 0

  text = ocr_plate(plate)
  print(text)
  img = cv2.putText(img, text, (beginX, beginY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150,255,0), 2, lineType=cv2.LINE_AA)
  img = cv2.rectangle(img, (beginX, beginY), (endX, endY), (150, 255, 0), 2)
  show_img(img)

  return img, plate

# %%
img, plate = recognize_plate(directory_imgs+'/car2.jpg')

# %% [markdown]
# # Improving the quality

# %%
img, plate = recognize_plate(directory_imgs+'/car3.jpg')

# %%
def preprocessing(img):
    increase = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    value, otsu = cv2.threshold(increase, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return otsu

# %%
processed_plate = preprocessing(plate)
show_img(processed_plate)
text = ocr_plate(processed_plate)
print(text)

```
