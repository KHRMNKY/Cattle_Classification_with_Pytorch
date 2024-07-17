from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_data_loader(transform):

        train_dataset = ImageFolder("./Cattle Dataset/train", transform=transform)
        test_dataset = ImageFolder("./Cattle Dataset/test", transform=transform)

        classes = train_dataset.classes

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        return train_loader, test_loader, classes

