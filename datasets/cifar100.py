import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar100_loaders(
    data_dir="./data",
    batch_size=32,
    num_workers=0,
    val_ratio=0.1,
    seed=42,
    pin_memory=False,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
):
    """
    Create CIFAR-100 train, validation, and test dataloaders.

    CIFAR-100 training set:
        50,000 images

    We split it into:
        train: 90%
        validation: 10%

    CIFAR-100 official test set is kept untouched for final evaluation.
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    # Same CIFAR-100 training data, but with different transforms.
    train_dataset_full = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset_full = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    dataset_size = len(train_dataset_full)
    val_size = int(dataset_size * val_ratio)

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Debug mode:
    # Allows us to use only a small part of the dataset.
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]

    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    if max_test_samples is not None:
        test_indices = list(range(
            min(max_test_samples, len(test_dataset))
        ))

        test_dataset = Subset(
            test_dataset,
            test_indices,
        )

    train_dataset = Subset(
        train_dataset_full,
        train_indices,
    )

    val_dataset = Subset(
        val_dataset_full,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader