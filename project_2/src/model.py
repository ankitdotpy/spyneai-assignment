import torch
import torch.nn as nn
import torchvision.models as models

class PredictionHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=256):
        super(PredictionHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CarAngleClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50'):
        super(CarAngleClassifier, self).__init__()
        self.model_name = model_name.lower()
        
        if self.model_name == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif self.model_name == 'resnet18':
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif self.model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif self.model_name == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif self.model_name == 'vit_b_16':
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            num_features = self.base_model.heads.head.in_features
            self.base_model.heads = nn.Identity()
        elif self.model_name == 'regnet_y_32gf':
            self.base_model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V2)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.prediction_head = PredictionHead(num_features, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        return self.prediction_head(features)

def get_model(num_classes, model_name='resnet50'):
    model = CarAngleClassifier(num_classes, model_name)
    return model