import os
from PIL import Image

# 設定目標資料夾路徑
folder = './original_data'  # 請替換為實際的資料夾路徑
new_size = (1000, 750)          # 設定目標尺寸

# 遍歷資料夾中的所有檔案
for filename in os.listdir(folder):
    # 檢查檔案是否為圖片（副檔名常見格式）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        file_path = os.path.join(folder, filename)

        # 開啟圖片並調整大小
        with Image.open(file_path) as img:
            # 調整圖片大小為指定尺寸
            resized_img = img.resize(new_size)

            # 儲存調整尺寸後的圖片，覆蓋原檔案
            resized_img.save(file_path)

        print(f"{filename} 已調整尺寸為 {new_size} 後儲存。")

