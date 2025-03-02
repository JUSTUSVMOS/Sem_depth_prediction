import os
import csv

# 指定包含圖片和深度圖的資料夾路徑
folder_path = 'data'  # 替換成你的資料夾路徑
csv_file_path = 'output.csv'  # 指定輸出的 CSV 檔案路徑

# 獲取資料夾中所有文件名
file_names = os.listdir(folder_path)

# 過濾出所有的圖片文件（假設圖片文件是 .jpg 格式）
image_files = [f for f in file_names if f.endswith('.jpg')]

if not image_files:
    print("No image files found in the folder.")
else:
    print(f"Found {len(image_files)} image files.")

# 創建並打開 CSV 檔案以寫入
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 記錄已寫入的行數
    rows_written = 0
    
    # 遍歷所有圖片文件並寫入 CSV
    for image_file in image_files:
        # 將圖片文件名後綴替換為深度圖的後綴
        depth_file = image_file.replace('.jpg', '_depth.png')
        
        # 檢查深度圖是否存在於同一資料夾中
        if depth_file in file_names:
            # 寫入 CSV 文件，並加上 `data/` 前綴
            writer.writerow([f"sem_images/data/{image_file}", f"sem_images/data/{depth_file}"])
            rows_written += 1
        else:
            print(f"Depth file not found for {image_file}")

print(f"CSV file '{csv_file_path}' has been created with {rows_written} rows.")

