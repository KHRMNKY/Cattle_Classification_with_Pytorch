import torch
from fastbook import *
import os
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import get_data_loader
from model_builder import create_model
import torchvision
from torch import nn



device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = create_model("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = get_data_loader(transform)

def accuracy(pred,target):
  correct_count=0
  total_count = len(target)
  tensor=torch.eq(pred.argmax(dim=1), target)
  for t in tensor:
    if t ==True:
      correct_count+=1
  acc=(correct_count/total_count)*100
  return acc




def image_pred(modelPath, imagePath, transform = transform):
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
       img = Image.open(imagePath).convert("RGB")
       img = transform(img).unsqueeze(0).to(device)
       loggit = model(img)
       preds = nn.Softmax(dim=1)(loggit)
       pred = torch.argmax(preds, dim=1).item()
    return pred, preds




def plt_show(imagePath, pred, preds): 
   preds = preds.cpu().numpy()
   image = Image.open(imagePath)
   #fig = plt.figure(figsize=(20, 10))
   plt.subplot(1, 2, 1)
   plt.title(f"{classes[pred]} {preds[0][pred]*100:.2f}%")
   plt.imshow(image)

   plt.subplot(1, 2, 2)
   plt.bar(classes, preds[0], color ='maroon', width = 0.9)
   plt.xlabel("Classes")
   plt.ylabel("predicted values")

   plt.show()




def prepare_valid_data():
  for cls in classes:
    path=f"./valid_data/{cls}"

    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print("directory already exists")
    
    urls = search_images_ddg(cls, max_images=5)
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    urls = [url for url in urls if url.endswith(image_extensions)]

    print(urls)
    for url in urls:
        try:
            download_url(dest=path, url=url)
        except:
            one_url = search_images_ddg(cls, max_images=1)
            try:
                download_url(dest=path, url=one_url[0])
            except:
                one_url = search_images_ddg(cls, max_images=1)
                download_url(dest=path, url=one_url[0])





def rename_imgs_in_directory(path):
    list_name_img = os.listdir(path)
    for index, name in enumerate(list_name_img):
        img_path = os.path.join(path, name)
        target = ".\\0"+ str(index) + ".jpg"
        os.rename(img_path, target)



def show_loss_and_accuracy(results, epochs):
    fig, ax = plt.subplots(1,2, figsize=(10, 7))

    ax[0].plot(range(epochs), results["train_loss"], label="train_loss")
    ax[0].plot(range(epochs), results["test_loss"], label="test_loss")

    plt.show()