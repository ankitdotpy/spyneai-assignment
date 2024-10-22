import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class GaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_data_loaders(data_dir, batch_size=32, train_ratio=0.8):
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms when creating the datasets
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    # val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # Calculate sizes
    total_size = len(train_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Use random_split on the datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # _, val_dataset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, val_loader, train_dataset.dataset.class_to_idx

def analyze_dataset(data_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))

    print("Dataset Analysis:")
    print(f"Total classes: {len(class_counts)}")
    for class_name, count in sorted(class_counts.items(), key=lambda x: int(x[0])):
        print(f"  {class_name}: {count} images")
