import os
from PIL import Image
import numpy as np

def histogram_correction(image):
    # 將圖像轉換為灰度模式
    gray_image = image.convert("L")
    
    # 將圖像轉換為 numpy 數組
    image_array = np.array(gray_image)
    
    # 計算最低 0.2% 和最高 0.2% 的閾值
    low_percentile = np.percentile(image_array, 0.2)
    high_percentile = np.percentile(image_array, 99.8)
    
    # 將低於最低閾值的像素設為最小值，高於最高閾值的像素設為最大值
    image_array = np.clip(image_array, low_percentile, high_percentile)
    
    # 將像素值縮放到 0-255 的範圍
    image_array = ((image_array - low_percentile) / (high_percentile - low_percentile) * 255).astype(np.uint8)
    
    # 將處理過的數組轉換回 PIL 圖像
    corrected_image = Image.fromarray(image_array)
    
    return corrected_image

def process_folder(input_folder, output_folder):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取資料夾中所有的圖片文件
    for filename in os.listdir(input_folder):
        # 只對 .jpg 和 .jpeg 文件進行處理
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 打開圖片並進行直方圖校正
            with Image.open(input_path) as img:
                corrected_img = histogram_correction(img)
                # 保存處理後的圖片
                corrected_img.save(output_path)
                print(f"Processed and saved: {output_path}")

# 設定輸入資料夾和輸出資料夾
input_folder = './original_data'  # 替換為你的輸入資料夾路徑
output_folder = './original_data'  # 替換為你的輸出資料夾路徑

# 處理資料夾中的所有 .jpg 和 .jpeg 圖片
process_folder(input_folder, output_folder)
