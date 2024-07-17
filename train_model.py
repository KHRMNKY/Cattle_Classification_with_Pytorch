import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model, loss_fn, optimizer, train_loader,scheduler):
  trainLoss = 0
  model.train()
  for imgs ,labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    train_pred=model(imgs)
    loss= loss_fn(train_pred,labels)
    trainLoss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  trainLoss=trainLoss/len(train_loader)
  return trainLoss


def test_step(model, loss_fn , test_loader):
  testLoss=0
  total_acc=0
  model.eval()
  for imgs, labels in test_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    test_pred=model(imgs)
    loss= loss_fn(test_pred,labels)
    testLoss+=loss.item()
    acc=accuracy(test_pred,labels)
    total_acc+=acc
  total_acc=total_acc/len(test_loader)
  testLoss=testLoss/len(test_loader)
  return testLoss, total_acc



def training_loop (epochs, lr, train_loader, test_loader, model):
        
    results = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        torch.manual_seed(42)
        model0 = model
        model0.to(device)
        loss_fn=nn.CrossEntropyLoss()
        params=model0.parameters()
        lr=lr
        optimizer=torch.optim.SGD(params=params, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        trainLoss= train_step(model0, loss_fn, optimizer, train_loader, scheduler)
        results["train_loss"].append(trainLoss)

        with torch.inference_mode():
            testLoss, total_acc = test_step(model0, loss_fn, test_loader)
            results["test_loss"].append(testLoss)
    
            if epoch%5==0:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {trainLoss:.2f}, Test Loss: {testLoss:.2f}-----accuracy: {total_acc:.2f}%")

    modelName= f'model_({total_acc:.2f}%)_acc.pth'
    models_dir=os.path.join(".", "models")
    modelPath= os.path.join(models_dir, modelName)
    torch.save(model0.state_dict(), modelPath)
    print(f"model saved at {modelPath}")
    return results



if __name__ == "__main__":
    epochs = 50
    lr = 0.05
    results = training_loop(epochs = epochs, lr = lr, train_loader = train_loader, test_loader= test_loader , model= model)
    show_loss_and_accuracy(results, epochs)


