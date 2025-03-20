import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")  # Convert to RGB mode
        if self.transform:
            image = self.transform(image)
        return image

# Calculate the mean and standard deviation of the dataset
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images in loader:
        images = images.view(images.size(0), images.size(1), -1)  # Reshape the tensor
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Set the dataset path and transformation
dataset_path = './data'
transform = transforms.ToTensor()

# Create the dataset and dataloader
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

# Calculate mean and standard deviation
mean, std = calculate_mean_std(loader)

print(f'Mean: {mean}')
print(f'Standard Deviation: {std}')
