import os
import cv2
import numpy as np
import math

def compute_metrics(gt, pred):
    epsilon = 1e-8
    gt = gt.astype(np.float32).flatten()
    pred = pred.astype(np.float32).flatten()
    
    print("Ground truth min value:", np.min(gt))
    
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rel_abs = np.mean(np.abs(gt - pred) / (gt + epsilon))
    
    ratio = np.maximum(gt / (pred + epsilon), pred / (gt + epsilon))
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25**2)
    delta3 = np.mean(ratio < 1.25**3)
    
    return mae, rmse, rel_abs, delta1, delta2, delta3

folder = "./inference_fine_tune_results" 

files = os.listdir(folder)
print("Files in folder：", files)


gt_files = [f for f in files if f.endswith('_gt.png')]
print("Eligible gt profiles：", gt_files)

all_metrics = []

for gt_file in gt_files:
    pred_file = gt_file.replace('_gt.png', '_pred.png')
    gt_path = os.path.join(folder, gt_file)
    pred_path = os.path.join(folder, pred_file)
    
    if not os.path.exists(pred_path):
        print(f"No corresponding forecast file found：{gt_file}")
        continue

    gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    
    if gt_img is None or pred_img is None:
        print(f"Failed to read image: {gt_file} or {pred_file}")
        continue

    metrics = compute_metrics(gt_img, pred_img)
    all_metrics.append(metrics)
    sample_name = gt_file.replace('_gt.png', '')
    print(f"{sample_name}:")
    print(f"  MAE     = {metrics[0]:.4f}")
    print(f"  RMSE    = {metrics[1]:.4f}")
    print(f"  RelAbs  = {metrics[2]:.4f}")
    print(f"  δ<1.25  = {metrics[3]:.4f}")
    print(f"  δ<1.25^2= {metrics[4]:.4f}")
    print(f"  δ<1.25^3= {metrics[5]:.4f}\n")

if all_metrics:
    all_metrics = np.array(all_metrics)
    avg_metrics = np.mean(all_metrics, axis=0)
    print("Average Metrics over all files:")
    print(f"  MAE     = {avg_metrics[0]:.4f}")
    print(f"  RMSE    = {avg_metrics[1]:.4f}")
    print(f"  RelAbs  = {avg_metrics[2]:.4f}")
    print(f"  δ<1.25  = {avg_metrics[3]:.4f}")
    print(f"  δ<1.25^2= {avg_metrics[4]:.4f}")
    print(f"  δ<1.25^3= {avg_metrics[5]:.4f}")
