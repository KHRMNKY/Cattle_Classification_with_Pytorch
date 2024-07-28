import torch
import clip
import os
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device="cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device)



def class_embedding(class_path):
    embedding_tensor_list = []
    image_paths = [os.path.join(class_path, name) for name in os.listdir(class_path)]
    
    for index, image_path in enumerate(image_paths):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
            embedding_tensor_list.append(image_embedding)
            print(f"{index}. resim embedding olarak listeye eklendi..")

    return embedding_tensor_list, image_paths




def embedding_to_dataFrame(embedding_list):
    embedding_array_list = [i.squeeze(0).cpu().numpy() for i in embedding_list]
    df = pd.DataFrame(embedding_array_list)
    print("embedding.csv basarili bir sekilde olusturuldu..")
    return df.to_csv("./embedding.csv")




def calculate_pca():
    df = pd.read_csv("./embedding.csv")
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    pca = PCA(n_components=6)   
    pca_embedding = pca.fit_transform(df)
    return pca_embedding



def clustering(pca_embedding, k):
    centroid, labels = kmeans2(pca_embedding, k, minit='points')
    return labels
    



def cluster_dir(label):
    path = os.path.join(".", f"{label}.cluster")
    if not os.path.exists(path):
        os.mkdir(path)
    return path




if __name__ == "__main__":

    embedding_tensor_list, image_paths = class_embedding(".\\Cattle Dataset\\train\\jersey")
    embedding_to_dataFrame(embedding_tensor_list)

    pca_embedding = calculate_pca()
    k = 3

    labels = clustering(pca_embedding, k)
    print(image_paths, labels)

    for i in range(k):
        cluster_dir(i)


    for index, image_path in enumerate(image_paths):
        shutil.copy2(image_path, cluster_dir(labels[index]))

