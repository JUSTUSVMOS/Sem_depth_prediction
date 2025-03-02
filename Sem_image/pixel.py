import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 指定圖片所在的資料夾路徑
folder_path = 'original_his'

# 初始化一個 256 長度的陣列，用來累加所有圖片的直方圖
combined_hist = np.zeros(256)

# 遍歷資料夾中的所有圖片檔案
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(folder_path, file_name)
        # 將圖片轉為灰階，若想分析 RGB 可自行修改
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        # 計算該圖片的直方圖
        hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0,256])
        # 將直方圖累加
        combined_hist += hist

# 繪製合併後的直方圖
plt.figure()
plt.title("Combined Pixel Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(combined_hist)
plt.xlim([0,256])
plt.show()
