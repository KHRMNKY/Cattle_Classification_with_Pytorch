# Cattle Species Detection with PyTorch
Bu projede 7 sığır ırkı arasında sınıflandırma yaptım.

7 sığırı ırk : Angus, Charolais, Hereford, Holstein, Jersey, Simental, Montofon

Böylece modele verilen bir sığır görüntüden hangi ırka ait olduğu belirlenebilir.


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)


## Introduction

Accurate cattle species detection is important for efficient farm management and breeding programs. This project leverages deep learning techniques to build a robust classifier for various cattle species.

## Dataset

The dataset used for this project consists of images of different cattle species. Each image is labeled with the corresponding species. The dataset should be organized into subfolders for each species under a main directory.

## Installation

Clone the repository:

```bash
git clone https://github.com/KHRMNKY/Cattle_Species_Detection_with_Pytorch.git

cd Cattle_Species_Detection_with_Pytorch

pip install -r requirements.txt
```

## Usage
```bash
python main.py --modelPath <"model yolu"> --imagePath <"image yolu">
```

## Training

If you want, you can change the hyperparameters (epoch, lr) and train your own model by running the train_model.py file. This trained model will be saved in the models folder.

```bash
python train_model.py
```

## Model Architecture
The ResNet50 architecture was used and fine-tuned on our dataset with PyTorch.
