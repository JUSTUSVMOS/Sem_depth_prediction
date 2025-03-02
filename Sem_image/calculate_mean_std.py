import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")  # 转换为 RGB 模式
        if self.transform:
            image = self.transform(image)
        return image

# 计算数据集的均值和标准差
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images in loader:
        images = images.view(images.size(0), images.size(1), -1)  # 重塑张量形状
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# 设置数据集路径和转换
dataset_path = './data'
transform = transforms.ToTensor()

# 创建数据集和数据加载器
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

# 计算均值和标准差
mean, std = calculate_mean_std(loader)

print(f'均值 (mean): {mean}')
print(f'标准差 (std): {std}')

