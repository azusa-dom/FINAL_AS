#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sij_gradcam_analysis.py

Leave-One-Subject-Out + Grad-CAM 可解释分析脚本
(6 vs 2 小样本，用少量增强+Fine-tune，观测模型关注区域并输出期刊级可视化三联图)
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import glob
import logging
from pathlib import Path # <--- ADDED THIS LINE!

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ————————————————————————————————————————————— #
# 1) 配置：项目根目录、数据路径、输出目录、随机种子、轻度增强
# ————————————————————————————————————————————— #
# project_root 指向 FINAL_AS 根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# 健康组 & AS 组目录

data_dirs = {
    "Healthy": [
        os.path.join(project_root, "data", "mri_health", "health1"),
        os.path.join(project_root, "data", "mri_health", "health2"),
    ],
    "AS": [
        os.path.join(project_root, "data", "mri_AS"),
    ],
}

# Output directory (will be further subdivided by class/subject)
base_output_dir = os.path.join(project_root, "results", "gradcam")
os.makedirs(base_output_dir, exist_ok=True)

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 轻度数据增强：±5°旋转 + 随机裁剪 + 色彩微调
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
# 推理/Grad-CAM 时仅 resize+normalize
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ————————————————————————————————————————————— #
# 2) Grad-CAM 类
# ————————————————————————————————————————————— #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.grad = None
        self.act = None
        # 注册钩子
        target_layer.register_forward_hook(lambda m,i,o: setattr(self, 'act', o))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self, 'grad', go[0]))

    def __call__(self, x, cls_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls_idx].backward(retain_graph=True)
        g = self.grad[0]   # [C,H,W]
        a = self.act[0]    # [C,H,W]
        w = g.mean(dim=(1,2), keepdim=True)  # [C,1,1]
        cam = (w * a).sum(dim=0).detach().cpu().numpy()
        # The heatmap is resized to the input image dimensions (224, 224) here
        cam = cv2.resize(cam, (x.shape[3], x.shape[2])) # x.shape is [B, C, H, W] -> (W, H)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ————————————————————————————————————————————— #
# 3) 辅助函数：掩码查找和应用
# ————————————————————————————————————————————— #

def find_mask_for_slice(slice_path):
    """
    在同一目录下搜索所有包含关键词 'mask' 的文件（不区分大小写），
    返回第一个匹配的完整路径；否则返回 None。
    """
    dirpath, fname = os.path.split(slice_path)
    basename, _ = os.path.splitext(fname)
    
    # Define patterns to search for: {basename}*mask*.* or {basename}*.mask.*
    # This handles various naming conventions like "image_mask.png", "image.mask.png"
    # Or "image (1)_mask.jpg" etc.
    patterns = [
        os.path.join(dirpath, f"{basename}*mask*.*"), # e.g., image_mask.png
        os.path.join(dirpath, f"{basename}.mask.*")   # e.g., image.mask.png
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=False)
        if matches:
            # You might want to sort matches and pick the "best" one if multiple exist
            # For now, return the first one found
            return matches[0]
    return None

def apply_heatmap_overlay(original_pil_img, heatmap_np):
    """
    将 CAM 热图 overlay 到原图上，heatmap_np 应为与原图同尺寸的 [0,1] 浮点数组。
    返回 overlay 后的 PIL.Image。
    """
    # Resize original image to 224x224 and convert to numpy for overlay
    img_np_resized = np.array(original_pil_img.resize((224,224))).astype(float) / 255.0
    
    # Apply jet colormap to heatmap and get RGB (discard alpha)
    heatmap_colored = plt.cm.jet(heatmap_np)[..., :3]
    
    # Blend images (adjust weights as needed)
    overlay_np = 0.4 * img_np_resized + 0.6 * heatmap_colored
    overlay_np = np.clip(overlay_np, 0, 1) # Ensure values are within [0, 1]

    return Image.fromarray((overlay_np * 255).astype(np.uint8))


# ————————————————————————————————————————————— #
# 4) 收集所有图像样本
# ————————————————————————————————————————————— #
all_images = []
# Fixed for AS script to only collect AS data with label 1
cls_key = "AS"
label = 1 # Explicitly set label for AS script

for root_dir_path in data_dirs[cls_key]:
    root = Path(root_dir_path)
    if not root.is_dir():
        logger.warning(f"Directory not found: {root_dir_path}")
        continue
    for subj_folder in root.iterdir(): # Iterate directly over subject folders
        if subj_folder.is_dir():
            subject_id = subj_folder.name
            for img_path in subj_folder.rglob("*"): # Recursively find images in subject folder
                # Use a common set of image extensions
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    all_images.append({
                        'path': str(img_path),
                        'subject': subject_id,
                        'label': label,
                        'cls': cls_key
                    })

if not all_images:
    logger.error("No images found in data directories. Please check paths and extensions.")
    sys.exit(1)

# ————————————————————————————————————————————— #
# 5) 初始化模型 & 目标层
# ————————————————————————————————————————————— #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Use explicit weights version
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
target_layer = model.layer4[-1]

# ————————————————————————————————————————————— #
# 6) 留一法 Fine-tune + Grad-CAM 输出期刊级三联图
# ————————————————————————————————————————————— #
logger.info(f"Loaded {len(all_images)} image slices across {len(set(i['subject'] for i in all_images))} subjects.")
subjects = sorted(set(i['subject'] for i in all_images))

for subj in tqdm(subjects, desc='Subject LOOCV'):
    train_set = [i for i in all_images if i['subject'] != subj]
    test_set = [i for i in all_images if i['subject'] == subj]
    
    # Skip subjects with no images in test_set (shouldn't happen with correct data)
    if not test_set:
        logger.warning(f"No test images for subject {subj}, skipping LOOCV for this subject.")
        continue

    # Initialize a fresh model and optimizer for each LOOCV fold
    fold_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    fold_model.fc = nn.Linear(fold_model.fc.in_features, 2)
    fold_model.to(device)
    fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-4)
    fold_criterion = nn.CrossEntropyLoss()
    
    fold_model.train()
    for epoch in range(3): # Small number of epochs for fine-tuning
        random.shuffle(train_set) # Shuffle training data for each epoch
        for item in train_set:
            try: # Robustly handle image loading errors
                img = Image.open(item['path']).convert('RGB')
                inp = train_transform(img).unsqueeze(0).to(device)
                lbl = torch.tensor([item['label']], device=device)
                fold_optimizer.zero_grad()
                out = fold_model(inp)
                loss = fold_criterion(out, lbl)
                loss.backward()
                fold_optimizer.step()
            except Exception as e:
                logger.warning(f"Error during training image {item['path']}: {e}. Skipping image.")

    fold_model.eval()
    camer = GradCAM(fold_model, fold_model.layer4[-1]) # Use fold_model's layer4

    for item in test_set:
        try:
            original_img = Image.open(item['path']).convert('RGB')
            inp = eval_transform(original_img).unsqueeze(0).to(device)
            out = fold_model(inp) # Use fold_model for inference
            cls_pred = out.argmax(dim=1).item()
            heatmap = camer(inp, cls_pred) # Heatmap is already 224x224 from camer()

            # --- Masking Logic ---
            heatmap_to_plot = heatmap
            title_suffix = ""
            
            mask_file = find_mask_for_slice(item['path'])
            if mask_file:
                try:
                    mask_img = Image.open(mask_file).convert('L') # 'L' for grayscale
                    mask_resized = mask_img.resize((224, 224), Image.NEAREST)
                    mask_np = np.array(mask_resized)
                    mask_np = (mask_np > 127).astype(np.float32) # Binarize the mask to 0s and 1s
                    
                    heatmap_masked = heatmap.copy()
                    heatmap_masked *= mask_np # Element-wise multiplication to apply the mask
                    heatmap_to_plot = heatmap_masked
                    title_suffix = " (masked)"
                    logger.info(f"Mask applied for {os.path.basename(item['path'])} using {os.path.basename(mask_file)}")
                except Exception as e:
                    logger.warning(f"Error loading or applying mask {mask_file!r} for {item['path']!r}: {e}. Proceeding without mask.")
                    title_suffix = " (mask error)"
            else:
                logger.info(f"No mask found for {item['path']!r}. Proceeding without mask.")
                title_suffix = " (no mask)"

            # --- Generate Overlay ---
            overlay_img = apply_heatmap_overlay(original_img, heatmap_to_plot)

            # --- Plotting ---
            fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(12,4))
            
            # Original Image
            ax0.imshow(original_img.resize((224,224))); ax0.axis('off'); ax0.set_title('Original')
            
            # Raw Grad-CAM Heatmap
            im1 = ax1.imshow(heatmap, cmap='jet', vmin=0, vmax=1); ax1.axis('off'); ax1.set_title('Raw Grad-CAM')
            
            # Overlay (with mask or without)
            ax2.imshow(overlay_img); ax2.axis('off'); ax2.set_title(f'Overlay{title_suffix}')
            
            # Colorbar
            cbar = fig.colorbar(im1, ax=[ax0,ax1,ax2], location='right', fraction=0.046, pad=0.04)
            cbar.set_label('Activation', rotation=270, labelpad=15)
            
            fig.suptitle(f"Subject: {item['subject']} | Class: {item['cls']}", fontsize=16)
            
            # Save path based on class and subject
            output_subj_dir = os.path.join(base_output_dir, item['cls'], item['subject'])
            os.makedirs(output_subj_dir, exist_ok=True)
            
            # Filename: gradcam_{subject_id}_{original_image_basename}.svg
            out_name_base = os.path.basename(item['path']).split('.')[0]
            out_path = os.path.join(output_subj_dir, f"gradcam_{item['subject']}_{out_name_base}.svg")
            
            plt.tight_layout(rect=[0,0,1,0.92]) # Adjust layout to make space for suptitle
            plt.savefig(out_path, format='svg') # Save as SVG
            plt.close(fig)
            logger.info(f"Saved Grad-CAM for {item['path']!r} to {out_path}")

        except Exception as e:
            logger.error(f"Failed to process {item['path']!r} for Grad-CAM: {e}", exc_info=True)

logger.info("\u2705 Grad-CAM analysis complete. Results saved under results/gradcam/")