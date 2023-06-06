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

## Project 2: Receipt Scanning and OCR

Notebook: [`OCR_Project_02_Scanner_OCR.ipynb`](./OCR_Project_02_Scanner_OCR.ipynb).

In this project, some misaligned images are used: a receipt and a book cover. With them:

- The edges and contours are detected to pick the boders of the paper with the text.
- The image is warped and croped using the borders.
- The aligned image is processed: brightness and contrast adjusted, thresholding applied.
- OCR is applied to the final processed image.

## Project 3: License Plate Detection

Notebook: [`OCR_Project_03_Licence_plate_reading.ipynb`](./OCR_Project_03_Licence_plate_reading.ipynb).

In this project, car license plates are recognized:

- The image is processed to detect the contour that might be the border of the plate.
- The plate ROI is cropped.
- We can apply further processing to the ROI: thresholding, etc.
- We apply OCR with Tesseract to the ROI.
- We display the result.
