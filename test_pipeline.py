import torch

from datasets.cifar100 import get_cifar100_loaders
from models.model_factory import create_model


def test_model(model_name, images):

    print("=" * 50)
    print(f"Testing model: {model_name}")

    model = create_model(
        model_name=model_name,
        num_classes=100,
        pretrained=False,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print(f"Input shape : {images.shape}")
    print(f"Output shape: {outputs.shape}")

    predictions = outputs.argmax(dim=1)

    print(f"Predictions : {predictions[:5]}")

    print("Forward pass successful!")


def main():

    print("Loading CIFAR-100...")

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=8,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print()
    print("Dataset loaded successfully.")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"First labels: {labels[:5]}")

    print()

    test_model(
        "resnet50",
        images,
    )

    print()

    test_model(
        "mobilenet_v3_small",
        images,
    )


if __name__ == "__main__":
    main()