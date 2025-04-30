from PIL import Image
import torch
from torch import nn
from torchvision import models,transforms

trained_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

class CarClassfieResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        for param in self.model.parameters():  # Freezing the internal layers of trained model
            param.requires_grad = False

        for param in self.model.layer4.parameters():  # unfreezing layer4 and fc
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.486], std=[.229,.224,.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
          trained_model = CarClassfieResNet()
          trained_model.load_state_dict(torch.load("model/saved_model.pth"))
          trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _,predicted_class = torch.max(output , 1)
        return class_names[predicted_class.item()]
