from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def create_model(device):
    weights = ResNet50_Weights.DEFAULT
    #transform = weights.transforms()
    transform = transforms.Compose([
    transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    model = resnet50(weights=weights)

    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True).to(device)    

    return model, transform

