import os
import time
import datetime
import warnings
import random
import numpy as np
from zipfile import ZipFile  # 如果不使用zip，可忽略此模組
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image

warnings.filterwarnings("ignore")

from model import Resnet_UNet

###############################################
# 1. 讀取資料夾資料：修改路徑
###############################################
def loadFolder(folder):
    """
    從指定資料夾中讀取資料。
    預設資料夾結構為：
        folder/
          output.csv         ← CSV檔案，內容為：<圖片相對路徑>,<深度圖相對路徑>
          data/              ← 存放圖片與深度圖的資料夾
              image1.jpg
              image1_depth.png
              image2.jpg
              image2_depth.png
              ……
    """
    csv_path = os.path.join(folder, 'output.csv')
    nyu2_train = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = line.split(',')
                nyu2_train.append(tuple(row))
    nyu2_train = shuffle(nyu2_train, random_state=0)
    print(f"Loaded {len(nyu2_train)} samples from {csv_path}")
    return folder, nyu2_train

###############################################
# 2. 自訂 Dataset 與 影像轉換
###############################################
class depthDatasetMemory(Dataset):
    def __init__(self, base_folder, nyu2_train, transform=None):
        self.base_folder = base_folder  # 此處為 fine_tune_data
        self.nyu_dataset = nyu2_train   # CSV 內每一行包含兩個相對路徑，例如：data/image1.jpg, data/image1_depth.png
        self.transform = transform

    def __getitem__(self, idx):
        sample = list(self.nyu_dataset[idx])
        # 若路徑前有 "sem_images/" 則剝除
        if sample[0].startswith("sem_images/"):
            sample[0] = sample[0].replace("sem_images/", "", 1)
        if sample[1].startswith("sem_images/"):
            sample[1] = sample[1].replace("sem_images/", "", 1)
            
        image_path = os.path.join(self.base_folder, sample[0])
        depth_path = os.path.join(self.base_folder, sample[1])
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self, is_test=False, max_depth=205.0):
        """
        由於你已將 ground truth 深度圖處理為最小值 0、最大值 205，
        因此此處 max_depth 改為 205.0
        """
        self.is_test = is_test
        self.max_depth = max_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        depth = self.to_tensor_depth(depth)
        # 將深度圖正規化到 [0,1]
        depth = depth / self.max_depth
        depth = torch.clamp(depth, 0, 1)
        print("After ToTensor, depth range:", depth.min().item(), depth.max().item())
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        return transforms.functional.to_tensor(pic)
    
    def to_tensor_depth(self, pic):
        return torch.from_numpy(np.array(pic)).unsqueeze(0).float()

def ApplyTransformToBoth(image_transform, depth_transform=None):
    def _apply(sample):
        image, depth = sample['image'], sample['depth']
        if image_transform:
            image = image_transform(image)
        if depth_transform:
            depth = depth_transform(depth)
        return {'image': image, 'depth': depth}
    return _apply

def getDefaultTrainTransform():
    return transforms.Compose([
        # ToTensor(is_test=False, max_depth=205.0),
        ToTensor(is_test=False, max_depth=255.0),
        ApplyTransformToBoth(transforms.Normalize(mean=[0.4314]*3, std=[0.1755]*3))
    ])

def getNoTransform(is_test=False):
    return transforms.Compose([
        # ToTensor(is_test=is_test, max_depth=205.0)
        ToTensor(is_test=False, max_depth=205.0),
    ])

def getTrainingTestingData(batch_size, num_workers=0, folder='fine_tune_data'):
    base_folder, nyu2_train = loadFolder(folder)
    transformed_training = depthDatasetMemory(base_folder, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing  = depthDatasetMemory(base_folder, nyu2_train, transform=getNoTransform())
    train_loader = DataLoader(transformed_training, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(transformed_testing, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

###############################################
# 3. Loss 與 輔助函數 (不動)
###############################################
def gaussian(window_size, sigma):
    from math import exp
    gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret

def DepthNorm(depth, maxDepth=205.0):
    return depth

###############################################
# 4. Fine-tune 訓練流程：修改使用的資料夾路徑
###############################################
def fine_tune():
    print("Starting fine-tune ...")
    # 建立模型並載入預訓練權重
    model = Resnet_UNet(out_channels=1).cuda()
    print("Loading pre-trained weights...")
    model.load_state_dict(torch.load('./weight/model_weights.pth'))
    
    # 可選擇凍結部分層（例如 encoder）
    for param in model.encoder.conv1.parameters():
        param.requires_grad = False
    for param in model.encoder.bn1.parameters():
        param.requires_grad = False
    for param in model.encoder.layer1.parameters():
        param.requires_grad = False

    # 這裡使用 fine_tune_data 作為新的資料集資料夾
    batch_size = 16
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size, num_workers=0, folder='fine_tune_data')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_criterion = nn.L1Loss()
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, sample_batched in enumerate(train_loader):
            image = sample_batched['image'].cuda()
            depth = sample_batched['depth'].cuda(non_blocking=True)
            depth_n = DepthNorm(depth, maxDepth=205.0)
            
            output = model(image)
            l_depth = l1_criterion(output, depth_n)
            l_ssim  = torch.clamp((1 - ssim(output, depth_n, val_range=1.0)) * 0.5, 0, 1)
            loss    = l_ssim + 0.1 * l_depth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'model_weights_finetune.pth')
    print("Fine-tune finished. Saved new weights as 'model_weights_finetune.pth'.")

###############################################
# 5. 主函數入口
###############################################
if __name__ == "__main__":
    fine_tune()
