import torch
import timm

import torch.nn as nn
import torch.nn.functional as F


class  SkinClassifier(nn.Module):
    def __init__(self, backbone_name ,n_class):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, n_class)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    n_class = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SkinClassifier('convnext_small_in22ft1k',n_class)
    model.to(device)

    image_test = torch.randn(1, 3, 256, 256).to(device)
    y_pred = model(image_test) 
    probs = F.softmax(y_pred, dim=-1)
    predictions = torch.argmax(probs, dim=-1)

    print(predictions)