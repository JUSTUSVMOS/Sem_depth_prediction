import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt  # Optional, for debugging

# Import model definition from model.py
from model import Resnet_UNet

# -----------------------------
# Preprocessing function, consistent with training
# -----------------------------
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4314, 0.4314, 0.4314],
                             std=[0.1755, 0.1755, 0.1755])
    ])
    return transform(image)

# -----------------------------
# Inference function: Process a single image
# -----------------------------
def inference_single(image_path, model, max_depth=205.0):
    # 1) Load and preprocess the image
    input_image = Image.open(image_path).convert('RGB')
    original_image = input_image.resize((1000, 750))
    input_tensor = preprocess(input_image).unsqueeze(0).cuda()

    # 2) Model inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # Predicted output should be in [0,1]
        output = torch.clamp(output, 0.0, 1.0)

    # 3) Post-processing: Convert to a numpy array and restore to actual scale
    # The range should be approximately 0 to max_depth (205.0 in this case)
    output_np = output.squeeze().cpu().numpy() * max_depth
    return original_image, output_np

# -----------------------------
# Load ground truth image (replace file extension accordingly)
# -----------------------------
def load_ground_truth(image_path):
    ground_truth_path = image_path.replace('.jpg', '_depth.png')
    if os.path.exists(ground_truth_path):
        return Image.open(ground_truth_path).convert('L')
    else:
        raise FileNotFoundError(f"Ground truth image not found at {ground_truth_path}")

# -----------------------------
# Main function: Iterate through the dataset folder, perform inference, and save results
# -----------------------------
def main():
    # Folder containing images (.jpg)
    data_folder = './fine_tune_data/original_data'
    # Output folder for inference results
    output_folder = "inference_fine_tune_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load model and pre-trained weights (modify path as needed)
    model = Resnet_UNet(out_channels=1).cuda()
    model.load_state_dict(torch.load('./model_weights_finetune.pth'))
    print("Model loaded.")

    # Retrieve all .jpg image files (assuming correct naming format)
    image_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.jpg')])

    # Iterate through images in the folder and perform inference
    for img_file in image_files:
        img_path = os.path.join(data_folder, img_file)
        try:
            # Use max_depth=205.0 to match the dataset range
            original_img, pred_depth = inference_single(img_path, model, max_depth=205.0)
            gt_img = load_ground_truth(img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

        # Resize for saving
        original_img = original_img.resize((1000, 750))
        gt_img = gt_img.resize((1000, 750))
        pred_depth_resized = cv2.resize(pred_depth, (1000, 750), interpolation=cv2.INTER_LINEAR)

        # Print min and max values of the predicted depth map
        print(f"{img_file}: pred min = {pred_depth_resized.min()}, pred max = {pred_depth_resized.max()}")

        # Adaptive linear stretching: Map predicted depth values to 0-255
        pred_min = pred_depth_resized.min()
        pred_max = pred_depth_resized.max()
        pred_uint8 = np.uint8(255 * (pred_depth_resized - pred_min) / (pred_max - pred_min + 1e-8))
        
        # To maintain dataset brightness ratio (max = 205), scale accordingly:
        fine_tuned_max = 255.0
        pred_final = np.uint8(pred_uint8.astype(np.float32) * (fine_tuned_max / 255.0))
        
        # Save original image, ground truth, and predicted depth map
        base_name = os.path.splitext(img_file)[0]
        original_img.save(os.path.join(output_folder, f"{base_name}_original.jpg"))
        gt_img.save(os.path.join(output_folder, f"{base_name}_gt.png"))
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_pred.png"), pred_final)
        
        print(f"Processed {img_file}: saved original, ground truth, and predicted depth images.")
    
    print("Inference completed. Results saved in:", output_folder)

if __name__ == "__main__":
    main()
