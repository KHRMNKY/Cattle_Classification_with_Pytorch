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


## Using API
```bash
uvicorn api:app --reload
```
![image](https://github.com/user-attachments/assets/8cae0c8d-1dd2-4228-97d9-24379a02a06a)


## Using CLI
```bash
python cli.py --modelPath <"path model"> --imagePath <"image path">
```
![image](https://github.com/user-attachments/assets/87b59a8c-f008-4360-b066-9521c5b30ac8)


## Deployment

### Live API
The project is now deployed and available online:

- **API URL:** https://cow.kahramankaya.com
- **API Documentation (Swagger UI):** https://cow.kahramankaya.com/docs#/
- **API Documentation (ReDoc):** https://cow.kahramankaya.com/redoc

### Usage Examples

#### Cattle Classification
The API provides two endpoints for cattle breed classification:

1. **`/predict`** - Returns only the predicted breed with highest confidence
2. **`/predicts`** - Returns all breed probabilities (detailed prediction)

#### Image Database Operations
The API also supports image storage and retrieval operations:

- **`POST /images/`** - Upload and store an image
- **`GET /images/{image_Id}`** - Retrieve image information by ID
- **`PUT /images/`** - Update image information
- **`DELETE /images/`** - Delete an image from database

### Supported Cattle Breeds
- Aberdeen Angus
- Charolais
- Hereford
- Holstein
- Jersey
- Montofon
- Simmental

### API Response Example
```json
{
  "label": "Holstein",
  "confidence": [0.1019, 0.1149, 0.0221, 0.0827, 0.1414, 0.3940, 0.1432]
}
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

