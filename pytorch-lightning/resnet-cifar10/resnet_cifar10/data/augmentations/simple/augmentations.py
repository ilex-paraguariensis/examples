from torchvision import transforms


def get_augmentations(image_size=256):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_trainsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transform, train_trainsform


def get_test_augmentations(image_size=256):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transform


def get_train_augmentations(image_size=256):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transform
