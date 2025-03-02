import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt  # 可选，用于调试

# 从 model.py 中导入模型定义
from model import Resnet_UNet

# -----------------------------
# 预处理函数，与训练保持一致
# -----------------------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4314, 0.4314, 0.4314],
                             std=[0.1755, 0.1755, 0.1755])
    ])
    return transform(image)

# -----------------------------
# 推理函数：对单张图像进行推理
# -----------------------------
def inference_single(image_path, model, max_depth=205.0):
    # 1) 读取图像并预处理
    input_image = Image.open(image_path).convert('RGB')
    original_image = input_image.resize((1000, 750))
    input_tensor = preprocess(input_image).unsqueeze(0).cuda()

    # 2) 模型推理
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # 预测输出应在 [0,1]
        output = torch.clamp(output, 0.0, 1.0)

    # 3) 后处理：转换为 numpy 数组，并还原到实际单位
    # 此时范围大致是 0～max_depth（这里传入 205.0）
    output_np = output.squeeze().cpu().numpy() * max_depth
    return original_image, output_np

# -----------------------------
# 加载 Ground Truth 图像（根据 jpg 文件名替换后缀）
# -----------------------------
def load_ground_truth(image_path):
    ground_truth_path = image_path.replace('.jpg', '_depth.png')
    if os.path.exists(ground_truth_path):
        return Image.open(ground_truth_path).convert('L')
    else:
        raise FileNotFoundError(f"Ground truth image not found at {ground_truth_path}")

# -----------------------------
# 主函数：遍历数据文件夹，进行推理并保存结果，同时打印出最大亮度信息
# -----------------------------
def main():
    # 数据所在文件夹（包含所有 .jpg 图像）
    data_folder = './fine_tune_data/original_data'
    # 输出结果保存文件夹
    output_folder = "inference_fine_tune_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 加载模型和预训练权重（请根据实际情况修改路径）
    model = Resnet_UNet(out_channels=1).cuda()
    model.load_state_dict(torch.load('./model_weights_finetune.pth'))
    print("Model loaded.")

    # 获取所有 .jpg 图像文件（假设文件名格式正确）
    image_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.jpg')])

    # 遍历文件夹内所有图像进行推理
    for img_file in image_files:
        img_path = os.path.join(data_folder, img_file)
        try:
            # 使用 max_depth=205.0 与当前数据集范围匹配
            original_img, pred_depth = inference_single(img_path, model, max_depth=205.0)
            gt_img = load_ground_truth(img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

        # 调整尺寸以便保存
        original_img = original_img.resize((1000, 750))
        gt_img = gt_img.resize((1000, 750))
        pred_depth_resized = cv2.resize(pred_depth, (1000, 750), interpolation=cv2.INTER_LINEAR)

        # 打印预测深度图的最小值和最大值
        print(f"{img_file}: pred min = {pred_depth_resized.min()}, pred max = {pred_depth_resized.max()}")

        # 自适应线性拉伸：将预测深度图映射到 0～255
        pred_min = pred_depth_resized.min()
        pred_max = pred_depth_resized.max()
        pred_uint8 = np.uint8(255 * (pred_depth_resized - pred_min) / (pred_max - pred_min + 1e-8))
        
        # 如果希望最终图片的亮度保持数据集比例（即最大为205），按比例压缩：
        fine_tuned_max = 255.0
        pred_final = np.uint8(pred_uint8.astype(np.float32) * (fine_tuned_max / 255.0))
        
        # 保存原图、ground truth 与预测深度图
        base_name = os.path.splitext(img_file)[0]
        original_img.save(os.path.join(output_folder, f"{base_name}_original.jpg"))
        gt_img.save(os.path.join(output_folder, f"{base_name}_gt.png"))
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_pred.png"), pred_final)
        
        print(f"Processed {img_file}: saved original, ground truth, and predicted depth images.")
    
    print("Inference completed. Results saved in:", output_folder)

if __name__ == "__main__":
    main()
