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
        transforms.Normalize(mean=[0.1976, 0.1976, 0.1976], std=[0.1545, 0.1545, 0.1545])
    ])
    return transform(image)

# -----------------------------
# 推理函数：对单张图像进行推理
# -----------------------------
def inference_single(image_path, model, max_depth=111.0):
    # 1) 读取图像并预处理
    input_image = Image.open(image_path).convert('RGB')
    original_image = input_image.resize((1000, 750))
    input_tensor = preprocess(input_image).unsqueeze(0).cuda()

    # 2) 模型推理
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # 预测输出应在 [0,1]
        output = torch.clamp(output, 0.0, 1.0)

    # 3) 后处理：转换为 numpy 数组，并还原到实际单位（这里用 max_depth 参数还原）
    output_np = output.squeeze().cpu().numpy()
    output_np = output_np * max_depth
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
# 主函数：遍历数据文件夹，进行推理并保存结果
# -----------------------------
def main():
    # 数据所在文件夹（包含所有 .jpg 图像）
    data_folder = './sem_images/data/'
    # 输出结果保存文件夹
    output_folder = "inference_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 加载模型和预训练权重（请根据实际情况修改路径）
    model = Resnet_UNet(out_channels=1).cuda()
    model.load_state_dict(torch.load('./model_weights.pth'))
    print("Model loaded.")

    # 获取所有 .jpg 图像文件（假设文件名格式正确）
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
    image_files = sorted(image_files)

    # 遍历文件夹内所有图像进行推理
    for img_file in image_files:
        img_path = os.path.join(data_folder, img_file)
        try:
            # 使用 max_depth=111.0，使预测值范围与代码2一致
            original_img, pred_depth = inference_single(img_path, model, max_depth=111.0)
            gt_img = load_ground_truth(img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

        # 调整尺寸以便保存
        original_img = original_img.resize((1000, 750))
        gt_img = gt_img.resize((1000, 750))
        pred_depth_resized = cv2.resize(pred_depth, (1000, 750), interpolation=cv2.INTER_LINEAR)

        # 保存原图和 ground truth（使用 PIL 保存）
        original_img.save(os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_original.jpg"))
        gt_img.save(os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_gt.png"))
        
        # 直接将预测深度图（0~111）转换为 uint8，不做自适应拉伸
        pred_uint8 = np.uint8(pred_depth_resized)
        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_pred.png"), pred_uint8)
        
        print(f"Processed {img_file}: pred min = {pred_depth_resized.min()}, pred max = {pred_depth_resized.max()}")

    print("Inference completed. Results saved in:", output_folder)

if __name__ == "__main__":
    main()
