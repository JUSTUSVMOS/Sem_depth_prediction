from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片並轉成灰階
img = Image.open('./unnamed.png').convert('L')
img_array = np.array(img)

# 取得圖片尺寸
height, width = img_array.shape

# 取出 y 軸中點的那一行像素值
mid_row = img_array[height // 2, :]

# 每 10 個點計算一次平均值
avg_values = [np.mean(mid_row[i:i+10]) for i in range(0, width, 10)]
avg_values = np.array(avg_values)

# 計算各分組的 x 軸位置（以分組中間位置表示）
group_positions = [i+10/2 for i in range(0, width, 10)]

# 畫出直條圖
plt.figure(figsize=(10, 5))
plt.bar(group_positions, avg_values, width=10, align='center')
plt.title('圖片 y 軸中點，每 10 個像素的平均值直條圖')
plt.xlabel('X 軸位置')
plt.ylabel('平均像素強度')
plt.show()
