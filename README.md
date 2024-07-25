# Cattle Species Detection with PyTorch
In this project, I classified 7 cattle breeds.

7 cattle breeds: Angus, Charolais, Hereford, Holstein, Jersey, Simmental, Montofon

Thus, it can be determined which breed a given cattle image belongs to.


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Data Preprocessing]((#data-preprocessing))


## Introduction

Accurate cattle species detection is important for efficient farm management and breeding programs. This project leverages deep learning techniques to build a robust classifier for various cattle species.

## Dataset

The Cattle dataset used for this project consists of images of different cattle species. Each image is labeled with the corresponding species.

## Installation

Clone the repository:

```bash
git clone https://github.com/KHRMNKY/Cattle_Species_Detection_with_Pytorch.git

cd Cattle_Species_Detection_with_Pytorch

pip install -r requirements.txt
```

## Using CLI
```bash
python cli.py --modelPath <"path model"> --imagePath <"image path">
```

## Using API
```bash
uvicorn api:app --reload
```

## Training

If you want, you can change the hyperparameters (epoch, lr) and train your own model by running the train_model.py file. This trained model will be saved in the models folder.

```bash
python train_model.py
```

## Model Architecture
The ResNet50 architecture was used and fine-tuned on our dataset with PyTorch.


## Data Preprocessing
In this section, the dataset has been prepared using PCA method and  kmeans2 clustering algorithm.
These operations are located in the preprocessing.py file.

