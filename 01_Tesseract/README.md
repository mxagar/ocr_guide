# OCR Guide: Tesseract

This repository contains a guide and example code on Optical Character Recognition (OCR).

I compiled this material after trying several tutorials and courses; I list the most relevant ones here:

- [Udemy: Optical Character Recognition (OCR) in Python](https://www.udemy.com/course/ocr-optical-character-recognition-in-python/)
- [PyImageSearch: Tutorials on OCR](https://pyimagesearch.com/)

This sub-folder deals with OCR in general and the package Tesseract, which is the most common python library for OCR.

Table of contents:

- [OCR Guide: Tesseract](#ocr-guide-tesseract)
  - [1. Introduction](#1-introduction)
    - [Installation](#installation)
    - [Course Material](#course-material)
    - [Notebook](#notebook)

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

### Notebook


