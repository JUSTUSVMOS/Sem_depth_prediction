import os
import glob
import numpy as np
from PIL import Image

def random_translate_scale(img, depth_img, scale_min=1.0, scale_max=1.2):
    """
    對RGB與Depth影像做隨機縮放與平移（只做一次），回傳新的 (padded_img, padded_depth)。
    - scale_factor 在 [scale_min, scale_max] 間
    - 平移範圍：±(原圖寬高 // 4)
    """
    original_width, original_height = img.size

    # 隨機縮放因子
    scale_factor = np.random.uniform(scale_min, scale_max)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 以LANCZOS & NEAREST 分別縮放RGB與Depth
    scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
    depth_img = depth_img.convert("L")  # 確保深度是灰階模式
    scaled_depth = depth_img.resize((new_width, new_height), Image.NEAREST)

    # 建立一張與原圖同大小的空白畫布
    padded_img = Image.new('RGB', (original_width, original_height), (0, 0, 0))
    # 深度圖常用單通道，先用灰階底色 (89) 代表空值
    padded_depth = Image.new('L', (original_width, original_height), 89)

    # 隨機平移
    x_shift = np.random.randint(-original_width // 4, original_width // 4)
    y_shift = np.random.randint(-original_height // 4, original_height // 4)

    x_start = max(0, min((original_width - new_width) // 2 + x_shift, original_width - new_width))
    y_start = max(0, min((original_height - new_height) // 2 + y_shift, original_height - new_height))

    # 將縮放後的影像貼到畫布
    padded_img.paste(scaled_img, (x_start, y_start))
    padded_depth.paste(scaled_depth, (x_start, y_start))

    return padded_img, padded_depth

def random_resize_crop(img, depth_img, crop_ratio):
    """
    以 crop_ratio 的比例隨機裁切，再放大回原尺寸 (img.size)。
    只做一次裁切，確保 RGB/Depth 同步。
    """
    img_width, img_height = img.size
    crop_width = int(img_width * crop_ratio)
    crop_height = int(img_height * crop_ratio)

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Crop ratio太小，導致 crop 尺寸無效。")

    # 隨機決定左上角裁切點
    x_start = np.random.randint(0, img_width - crop_width + 1)
    y_start = np.random.randint(0, img_height - crop_height + 1)

    # 對RGB與Depth裁切
    cropped_img = img.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))
    cropped_depth = depth_img.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))

    # 放大回原尺寸
    resized_cropped_img = cropped_img.resize((img_width, img_height), Image.LANCZOS)
    resized_cropped_depth = cropped_depth.resize((img_width, img_height), Image.NEAREST)

    return resized_cropped_img, resized_cropped_depth

def single_augment_and_save(img, depth_img, output_dir, base_name, index,
                            scale_min=1.0, scale_max=1.2, crop_ratio=0.08):
    """
    針對同一對 (RGB, Depth) 圖片執行一次增強(隨機縮放平移+一次裁切)，最後輸出1張成對的結果。
    """
    # 第一步：隨機縮放+平移
    translated_img, translated_depth = random_translate_scale(
        img, depth_img, scale_min=scale_min, scale_max=scale_max
    )

    # 第二步：隨機裁切 (只做一次)
    final_img, final_depth = random_resize_crop(translated_img, translated_depth, crop_ratio)

    # 建立輸出路徑並存檔
    os.makedirs(output_dir, exist_ok=True)

    output_img_path = os.path.join(output_dir, f"{base_name}_{index}.jpg")
    output_depth_path = os.path.join(output_dir, f"{base_name}_{index}_depth.png")

    final_img.save(output_img_path)
    final_depth.save(output_depth_path)

    print(f"Saved final augmented pair:\n  {output_img_path}\n  {output_depth_path}")

def process_folder(input_dir, output_dir, num_images=1, scale_min=1.0, scale_max=1.2,
                   crop_ratio=0.08):
    """
    遍歷資料夾下所有 jpg 檔案，對應找 *_depth.png。
    對每對(RGB,Depth) 產生 num_images 次增強，每次只輸出1張成對結果。
    """
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        base_name = os.path.basename(img_path).replace('.jpg', '')
        depth_img_path = os.path.join(input_dir, f'{base_name}_depth.png')

        if not os.path.exists(depth_img_path):
            print(f"[警告] 找不到對應的深度圖：{depth_img_path}, 跳過此張。")
            continue

        # 讀取原圖 & 深度圖
        img = Image.open(img_path).convert('RGB')    # 確保RGB格式
        depth_img = Image.open(depth_img_path).convert('L')  # 保留單通道

        # 針對同一張圖做 num_images 次隨機增強（各生成一對）
        for i in range(num_images):
            single_augment_and_save(
                img, depth_img,
                output_dir=output_dir,
                base_name=base_name,
                index=f"{i+1}",
                scale_min=scale_min,
                scale_max=scale_max,
                crop_ratio=crop_ratio
            )

# --------------------------- 主程式 -------------------------------------
if __name__ == "__main__":
    # 參數設定
    input_dir = "./original_data"       # 原始RGB與Depth同資料夾
    output_dir = "./data"  # 輸出增強後結果
    num_images = 5             # 每張圖要產生幾個增強版本
    scale_min = 1.0
    scale_max = 1.2
    crop_ratio = 0.8          # 隨機裁切比例

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        num_images=num_images,
        scale_min=scale_min,
        scale_max=scale_max,
        crop_ratio=crop_ratio
    )
