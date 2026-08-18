import torch.nn as nn
from torchvision import models


NUM_CLASSES = 100


def create_model(model_name, num_classes=NUM_CLASSES, pretrained=False):

    model_name = model_name.lower()

    if model_name == "resnet50":

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet50(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == "mobilenet_v3_small":

        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        model = models.mobilenet_v3_small(weights=weights)

        last_layer = model.classifier[3]

        if not isinstance(last_layer, nn.Linear):
            raise TypeError("Expected classifier[3] to be nn.Linear")

        model.classifier[3] = nn.Linear(
            last_layer.in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model