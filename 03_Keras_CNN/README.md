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
    - [`01_Custom_OCR_training_the_neural_network.ipynb`](#01_custom_ocr_training_the_neural_networkipynb)
    - [`02_Custom_OCR_Text_recognition.ipynb`](#02_custom_ocr_text_recognitionipynb)

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

### [`01_Custom_OCR_training_the_neural_network.ipynb`](01_Custom_OCR_training_the_neural_network.ipynb)

```python
# %% [markdown]
# # Custom OCR - training the neural network
# 
# 

# %% [markdown]
# # Importing the libraries

# %%
import tensorflow
tensorflow.__version__

# %%
import numpy as np
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %%
DATA_PATH = "./../material/"

# %% [markdown]
# # Loading the datasets

# %% [markdown]
# ## MNIST 0-9 

# %%
from tensorflow.keras.datasets import mnist

# %%
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# %%
train_data.shape, test_data.shape

# %%
28 * 28

# %%
train_labels.shape, test_labels.shape

# %%
train_data[0]

# %%
train_data[0].shape

# %%
train_labels[0]

# %%
train_labels

# %%
digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])

# %%
digits_data.shape

# %%
digits_labels.shape

# %%
np.random.randint(0, digits_data.shape[0])

# %%
index = np.random.randint(0, digits_data.shape[0])
plt.imshow(digits_data[index], cmap='gray')
plt.title('Class: ' + str(digits_labels[index]));

# %%
#sns.countplot(data=digits_labels)
df = pd.DataFrame({'labels': digits_labels})
sns.countplot(data=df, x='labels')

# %% [markdown]
# ## Kaggle A-Z

# %%
!wget https://iaexpert.academy/arquivos/alfabeto_A-Z.zip

# %%
zip_object = zipfile.ZipFile(file = './alfabeto_A-Z.zip', mode = 'r')
zip_object.extractall('./')
zip_object.close()

# %%
dataset_az = pd.read_csv('./A_Z Handwritten Data.csv').astype('float32')
dataset_az

# %%
alphabet_data = dataset_az.drop('0', axis = 1)
alphabet_labels = dataset_az['0']

# %%
alphabet_data.shape, alphabet_labels.shape

# %%
alphabet_labels

# %%
alphabet_data = np.reshape(alphabet_data.values, (alphabet_data.shape[0], 28, 28))

# %%
alphabet_data.shape

# %%
index = np.random.randint(0, alphabet_data.shape[0])
plt.imshow(alphabet_data[index], cmap = 'gray')
plt.title('Class: ' + str(alphabet_labels[index]));

# %%
#sns.countplot(alphabet_labels)
df = pd.DataFrame({'labels': alphabet_labels})
sns.countplot(data=df, x='labels')

# %% [markdown]
# ## Joining the datasets

# %%
digits_labels, np.unique(digits_labels)

# %%
alphabet_labels, np.unique(alphabet_labels)

# %%
alphabet_labels += 10

# %%
alphabet_labels, np.unique(alphabet_labels)

# %%
data = np.vstack([alphabet_data, digits_data])
labels = np.hstack([alphabet_labels, digits_labels])

# %%
data.shape, labels.shape

# %%
np.unique(labels)

# %%
data = np.array(data, dtype = 'float32')

# %%
data = np.expand_dims(data, axis = -1)

# %%
data.shape

# %% [markdown]
# # Pre-processing the data

# %%
data[0].min(), data[0].max()

# %%
data /= 255.0

# %%
data[0].min(), data[0].max()

# %%
np.unique(labels), len(np.unique(labels)) # softmax

# %%
le = LabelBinarizer()
labels = le.fit_transform(labels)

# %%
np.unique(labels)

# %%
labels

# %%
labels[0], len(labels[0])

# %%
labels[30000]

# %%
# OneHotEncoder
# A, B, C
# 0, 1, 2

# A, B, C
# 1, 0, 0
# 0, 1, 0
# 0, 0, 1

# %%
plt.imshow(data[0].reshape(28,28), cmap='gray')
plt.title(str(labels[0]));

# %%
classes_total = labels.sum(axis = 0)
classes_total

# %%
classes_total.max()

# %%
57825 / 6903

# %%
classes_weights = {}
for i in range(0, len(classes_total)):
    #print(i)
    classes_weights[i] = classes_total.max() / classes_total[i]

# %%
classes_weights

# %%
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=labels)

# %%
X_train.shape, X_test.shape

# %%
y_train.shape, y_test.shape

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
augmentation = ImageDataGenerator(rotation_range=10,
                                  zoom_range=0.05,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=False)

# %% [markdown]
# # Buiding the neural network
# 
# - Padding: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# %%
network = Sequential()

network.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 64, kernel_size=(3,3), activation='relu', padding='same'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding='valid'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(64, activation = 'relu'))
network.add(Dense(128, activation = 'relu'))

network.add(Dense(36, activation='softmax'))

network.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# %%
2 * 2 * 128

# %%
network.summary()

# %%
name_labels = '0123456789'
name_labels += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name_labels = [l for l in name_labels]

# %%
print(name_labels)

# %% [markdown]
# # Training the neural network

# %%
file_model = 'custom_ocr.model'
epochs = 20
batch_size = 128

# %%
checkpointer = ModelCheckpoint(file_model, monitor = 'val_loss', verbose = 1, save_best_only=True)

# %%
len(X_train) // batch_size

# %%
history = network.fit(augmentation.flow(X_train, y_train, batch_size=batch_size),
                      validation_data=(X_test, y_test),
                      steps_per_epoch=len(X_train) // batch_size,
                      epochs=epochs,
                      class_weight=classes_weights,
                      verbose=1,
                      callbacks=[checkpointer])

# %% [markdown]
# # Evaluating the neural network

# %%
X_test.shape

# %%
predictions = network.predict(X_test, batch_size=batch_size)

# %%
predictions

# %%
predictions[0]

# %%
len(predictions[0])

# %%
np.argmax(predictions[0])

# %%
name_labels[24]

# %%
y_test[0]

# %%
np.argmax(y_test[0])

# %%
name_labels[np.argmax(y_test[0])]

# %%
network.evaluate(X_test, y_test)

# %%
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names = name_labels))

# %%
history.history.keys()

# %%
plt.plot(history.history['val_loss']);

# %%
plt.plot(history.history['val_accuracy']);

# %% [markdown]
# # Saving the neural network on Google Drive

# %%
network.save('network', save_format= 'h5')

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
#!cp network /content/drive/MyDrive/Cursos\ -\ recursos/OCR\ with\ Python/Models/network

# %% [markdown]
# # Testing the neural network with images

# %%
from tensorflow.keras.models import load_model

# %%
#loaded_network = load_model('/content/drive/MyDrive/Cursos - recursos/OCR with Python/Models/network')
loaded_network = load_model('./network')

# %%
loaded_network

# %%
loaded_network.summary()

# %%
import cv2
#from google.colab.patches import cv2_imshow
img = cv2.imread(DATA_PATH+'Images/letter-m.jpg')
cv2_imshow(img)

# %%
img.shape

# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

# %%
cv2_imshow(gray)

# %%
value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2_imshow(thresh)

# %%
value

# %%
thresh.shape

# %%
img = cv2.resize(thresh, (28,28))
cv2_imshow(img)

# %%
img.shape

# %%
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis = -1)
img.shape

# %%
img = np.reshape(img, (1,28,28,1))
img.shape

# %%
prediction = loaded_network.predict(img)
prediction

# %%
np.argmax(prediction)

# %%
name_labels[22]

```

### [`02_Custom_OCR_Text_recognition.ipynb`](02_Custom_OCR_Text_recognition.ipynb)

```python
# %% [markdown]
# # Custom OCR - text recognition

# %% [markdown]
# # Importing the libraries

# %%
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import imutils

# %%
#from google.colab.patches import cv2_imshow
def cv2_imshow(img, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    plt.imshow(img)
    plt.show()

# %% [markdown]
# # Loading the neural network

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
DATA_PATH = "./../material/"

# %%
network = load_model(DATA_PATH+'Models/network')
network.summary()

# %% [markdown]
# # Loading the test image

# %%
img = cv2.imread(DATA_PATH+'Images/test-manuscript01.jpg')
cv2_imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %% [markdown]
# # Pre-processing the image

# %%
blur = cv2.GaussianBlur(gray, (3,3), 0)
cv2_imshow(blur)

# %%
adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
cv2_imshow(adaptive)

# %%
invertion = 255 - adaptive
cv2_imshow(invertion)

# %%
dilation = cv2.dilate(invertion, np.ones((3,3)))
cv2_imshow(dilation)

# %%
# Edge detection
edges = cv2.Canny(dilation, 40, 150)
cv2_imshow(edges)

# %%
dilation = cv2.dilate(edges, np.ones((3,3)))
cv2_imshow(dilation)

# %% [markdown]
# # Contour detection

# %%
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

# %%
conts = find_contours(dilation.copy())

# %%
#conts

# %%
# After the bounding boxes have been found
# we extract the ROIs that contain the letters
min_w, max_w = 4, 160
min_h, max_h = 14, 140
img_copy = img.copy()
for c in conts:
    #print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    #print(x, y, w, h)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        roi = gray[y:y+h, x:x+w]
        #cv2_imshow(roi)
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2_imshow(thresh)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)
cv2_imshow(img_copy)

# %% [markdown]
# # Processing the detected characters

# %% [markdown]
# ## ROI extraction
# 

# %%
def extract_roi(img):
    roi = img[y:y + h, x:x + w]
    return roi

# %% [markdown]
# ## Thresholding

# %%
def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh

# %% [markdown]
# ## Resizing

# %%
def resize_img(img, w, h):
    if w > h:
      resized = imutils.resize(img, width = 28)
    else:
      resized = imutils.resize(img, height = 28)
    
    (h, w) = resized.shape
    dX = int(max(0, 28 - w) / 2.0)
    dY = int(max(0, 28 - h) / 2.0)
    
    filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
    filled = cv2.resize(filled, (28,28))
    return filled

# %%
(x, y, w, h) = cv2.boundingRect(conts[6])
print(x, y, w, h)
test_img = thresholding(gray[y:y+h, x:x+w])
cv2_imshow(test_img)
(h, w) = test_img.shape
print(h, w)
test_img2 = resize_img(test_img, w, h)
cv2_imshow(test_img2)
print(test_img2.shape)

# %%
cv2_imshow(cv2.resize(test_img, (28,28)))

# %% [markdown]
# ## Normalization

# %%
def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis = -1)
    return img

# %%
test_img2.shape, normalization(test_img2).shape

# %% [markdown]
# ## Processing the detections

# %%
characters = []

# %%
def process_box(gray, x, y, w, h):
    roi = extract_roi(gray)
    thresh = thresholding(roi)
    (h, w) = thresh.shape
    resized = resize_img(thresh, w, h)
    cv2_imshow(resized)
    normalized = normalization(resized)
    characters.append((normalized, (x, y, w, h)))

# %%
for c in conts:
    #print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        process_box(gray, x, y, w, h)

# %%
characters[0]

# %%
boxes = [box[1] for box in characters]
boxes

# %%
pixels = np.array([pixel[0] for pixel in characters], dtype = 'float32')

# %%
pixels

# %% [markdown]
# # Recognition of characters

# %%
digits = '0123456789'
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
characters_list = digits + letters
characters_list = [l for l in characters_list]

# %%
print(characters_list)

# %%
pixels[0].shape

# %%
pixels.shape

# %%
predictions = network.predict(pixels)

# %%
predictions

# %%
predictions.shape

# %%
boxes

# %%
img_copy = img.copy()
for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    #print(i)
    probability = prediction[i]
    #print(probability)
    character = characters_list[i]
    #print(character)
    
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255,100,0), 2)
    cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
    print(character, ' -> ', probability * 100)
    
    cv2_imshow(img_copy)

# %%
def extract_roi(img, margin=2):
    roi = img[y - margin:y + h, x - margin:x + w + margin]
    return roi

# %%
conts = find_contours(dilation.copy())
characters = []
for c in conts:
  (x, y, w, h) = cv2.boundingRect(c)
  if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
    process_box(gray, x, y, w, h)

# %%
boxes = [b[1] for b in characters]
pixels = np.array([p[0] for p in characters], dtype='float32')
predictions = network.predict(pixels)

# %%
img_copy = img.copy()
for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    probability = prediction[i]
    character = characters_list[i]
    
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255,100,0), 2)
    cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
    print(character, ' -> ', probability * 100)
    
    cv2_imshow(img_copy)

# %% [markdown]
# # Other tests

# %%
def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 7)
    edges = cv2.Canny(blur, 40, 150)
    dilation = cv2.dilate(edges, np.ones((3,3)))  
    return gray, dilation

# %%
def prediction(predictions, characters_list):
    i = np.argmax(predictions)
    probability = predictions[i]
    character = characters_list[i]
    return i, probability, character

# %%
def draw_img(img_cp, character):
    cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.putText(img_cp, character, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

# %% [markdown]
# ## Problems with 0 and O, 1 an I, 5 and S

# %%
img = cv2.imread(DATA_PATH+'Images/test_manuscript02.jpg')
cv2_imshow(img)

# %%
gray, processed_img = preprocess_img(img)
cv2_imshow(gray)
cv2_imshow(processed_img)

# %%
conts = find_contours(processed_img.copy())
characters = []
for c in conts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        process_box(gray, x, y, w, h)

boxes = [b[1] for b in characters]
pixels = np.array([p[0] for p in characters], dtype="float32")
predictions = network.predict(pixels)

# %%
# A quick and dirty way to handle the confusion
# 1-I, 5-S, 0-O, we can
# 1. if letters are surrounded by letters, change numbers for letters
# 2. change the confusing numbers in the number list to be letters
digits_2 = 'OI234S6789'
letters_2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
characters_list_2 = digits_2 + letters_2
characters_list_2 = [l for l in characters_list_2]

# %%
img_cp = img.copy()
for (pred, (x, y, w, h)) in zip(predictions, boxes):
  i, probability, character = prediction(pred, characters_list_2)
  draw_img(img_cp, character)
cv2_imshow(img_cp)

# %% [markdown]
# ## Problems with undetected texts

# %%
img = cv2.imread(DATA_PATH+'Images/test_manuscript03.jpg')
cv2_imshow(img)

# %%
gray, processed_img = preprocess_img(img)
cv2_imshow(gray)
cv2_imshow(processed_img)

# %%
conts = find_contours(processed_img.copy()) # RETR_EXTERNAL
img_cp = img.copy()
for c in conts:
    (x, y, w, h) = cv2.boundingRect(c)
    #if (w >= l_min and w <= l_max) and (h >= a_min and h <= a_max):
    roi = gray[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 100, 0), 2)
cv2_imshow(img_cp)

# %%
height, width = img.shape[:2]
print(height, width)

# %%
contours_size = sorted(conts, key=cv2.contourArea, reverse=True)
contours_size

# %%
# The first contour ROI has no letters because
# a large contour is detected which contains the letter contours.
# Since we used the parameter to get external contours only in findContours(),
# the contained small letters are ignored.
# To adress that, the largest returned contour is taken and the ROI contained
# in it extracted. Then, the letter bounding boxes in that ROI are
# extracted as before.
for c in contours_size:
    (x, y, w, h) = cv2.boundingRect(c)
    
    if (w >= (width / 2)) and (h >= height / 2):
        cut_off = 8
        cut_img = img[y+cut_off:y + h - cut_off, x+cut_off:x + w - cut_off]
        cv2_imshow(cut_img)

# %%
gray, processed_img = preprocess_img(cut_img)
cv2_imshow(processed_img)

# %%
conts = find_contours(processed_img.copy())
characters = []
for c in conts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
        process_box(gray, x, y, w, h)

boxes = [b[1] for b in characters]
pixels = np.array([p[0] for p in characters], dtype="float32")
predictions = network.predict(pixels)

img_cp = cut_img.copy()
for (pred, (x, y, w, h)) in zip(predictions, boxes):
    i, probability, character = prediction(pred, characters_list_2)
    draw_img(img_cp, character)
cv2_imshow(img_cp)

```