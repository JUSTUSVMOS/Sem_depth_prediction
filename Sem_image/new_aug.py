import os
import cv2
import numpy as np
from pathlib import Path
import random

def augment_image(image, translation=None, scale=None, crop=None):
    """
    對圖片進行平移、縮放和隨機裁切。
    """
    rows, cols = image.shape[:2]

    # 平移
    if translation:
        tx, ty = translation
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # 縮放
    if scale:
        fx, fy = scale
        scaled_image = cv2.resize(image, (0, 0), fx=fx, fy=fy)
        # 恢復為原尺寸
        image = cv2.resize(scaled_image, (cols, rows))
    
    # 裁切
    if crop:
        x_start, y_start, crop_width, crop_height = crop
        cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
        # 恢復為原尺寸
        image = cv2.resize(cropped_image, (cols, rows))
    
    return image

def process_images(input_folder, output_folder, translations_count, scales_count, crops_count):
    """
    對每張圖片和對應的深度圖生成多個平移、縮放和裁切版本。
    """
    # 確保輸出資料夾存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 獲取所有原圖和深度圖
    images = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    
    for img_name in images:
        # 匹配對應的深度圖名稱
        depth_name = img_name.replace(".jpg", "_depth.png")
        
        # 確保深度圖存在
        img_path = os.path.join(input_folder, img_name)
        depth_path = os.path.join(input_folder, depth_name)
        
        if not os.path.exists(depth_path):
            print(f"深度圖 {depth_name} 不存在，跳過處理")
            continue
        
        # 讀取圖像和深度圖
        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # 平移增強
        for i in range(translations_count):
            tx = random.randint(-img.shape[1] // 4, img.shape[1] // 4)
            ty = random.randint(-img.shape[0] // 4, img.shape[0] // 4)
            translation = (tx, ty)

            img_aug = augment_image(img, translation=translation)
            depth_aug = augment_image(depth, translation=translation)

            img_aug_name = f"{img_name[:-4]}_translate_{i}.jpg"
            depth_aug_name = f"{img_name[:-4]}_translate_{i}_depth.png"

            cv2.imwrite(os.path.join(output_folder, img_aug_name), img_aug)
            cv2.imwrite(os.path.join(output_folder, depth_aug_name), depth_aug)

        # 縮放增強
        for i in range(scales_count):
            scale = random.uniform(0.8, 1.2)
            img_aug = augment_image(img, scale=(scale, scale))
            depth_aug = augment_image(depth, scale=(scale, scale))

            img_aug_name = f"{img_name[:-4]}_scale_{i}.jpg"
            depth_aug_name = f"{img_name[:-4]}_scale_{i}_depth.png"

            cv2.imwrite(os.path.join(output_folder, img_aug_name), img_aug)
            cv2.imwrite(os.path.join(output_folder, depth_aug_name), depth_aug)

        # 裁切增強
        for i in range(crops_count):
            crop_scale = random.uniform(0.08, 1.0)
            aspect_ratio = random.uniform(0.75, 1.33)

            crop_width = int(img.shape[1] * crop_scale)
            crop_height = int(crop_width / aspect_ratio)
            crop_width = min(crop_width, img.shape[1])
            crop_height = min(crop_height, img.shape[0])

            x_start = random.randint(0, img.shape[1] - crop_width)
            y_start = random.randint(0, img.shape[0] - crop_height)
            crop = (x_start, y_start, crop_width, crop_height)

            img_aug = augment_image(img, crop=crop)
            depth_aug = augment_image(depth, crop=crop)

            img_aug_name = f"{img_name[:-4]}_crop_{i}.jpg"
            depth_aug_name = f"{img_name[:-4]}_crop_{i}_depth.png"

            cv2.imwrite(os.path.join(output_folder, img_aug_name), img_aug)
            cv2.imwrite(os.path.join(output_folder, depth_aug_name), depth_aug)

        print(f"已處理並儲存：{img_name} 和 {depth_name}")

# 執行參數設定
input_folder = "original_data"  # 原始圖片資料夾
output_folder = "data_aug"         # 增強後儲存的資料夾

# 增強參數
translations_count = 10  # 每張圖片生成 3 個平移版本
scales_count = 10        # 每張圖片生成 5 個縮放版本
crops_count = 10         # 每張圖片生成 3 個裁切版本

# 執行增強處理
process_images(input_folder, output_folder, translations_count, scales_count, crops_count)
