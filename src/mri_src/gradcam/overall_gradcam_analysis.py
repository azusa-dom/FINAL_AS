#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final.py (Consolidated Grad-CAM Analysis Script)

Leave-One-Subject-Out + Grad-CAM 可解释分析脚本
支持指定只对 Healthy 或 AS 类别做 Grad-CAM 可视化，
并输出期刊级三联图 (Original / Raw CAM / Overlay)
"""

import os
import sys
import random
import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2 # OpenCV for contour drawing
from tqdm import tqdm
# mpl_toolkits.axes_grid1.make_axes_locatable is not needed if using fig.add_axes for colorbar
# from mpl_toolkits.axes_grid1 import make_axes_locatable 

# ————————————————————————————————————————————— #
# 全局配置：日志、种子
# ————————————————————————————————————————————— #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ————————————————————————————————————————————— #
# 0) 命令行参数：选择类别 & 输出目录
# ————————————————————————————————————————————— #
parser = argparse.ArgumentParser(
    description="Perform LOOCV Grad-CAM on either Healthy or AS MRI slices."
)
parser.add_argument(
    '--target-class', type=str, choices=['Healthy','AS'], default='Healthy',
    help="Which class to visualize (will train only on that class for Grad-CAM)."
)
parser.add_argument(
    '--out', type=str, default=None,
    help="Base output directory (default: project_root/results/gradcam)."
)
args = parser.parse_args()

# ————————————————————————————————————————————— #
# 1) 路径 & 数据增强
# ————————————————————————————————————————————— #
# FIX: Corrected project_root calculation.
# If __file__ is src/mri_src/gradcam/final.py, then parents[3] is FINAL_AS/
project_root = Path(__file__).resolve().parents[3] 

data_dirs = {
    "Healthy": [
        project_root/"data"/"mri_health"/"health1",
        project_root/"data"/"mri_health"/"health2",
    ],
    "AS": [
        project_root/"data"/"mri_AS",
    ],
}

base_output_dir = Path(args.out) if args.out else project_root/"results"/"gradcam"
base_output_dir.mkdir(parents=True, exist_ok=True)

# 微调时的数据增强
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
# 推理 & Grad-CAM
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ————————————————————————————————————————————— #
# 2) GradCAM 类
# ————————————————————————————————————————————— #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.grad = None
        self.act  = None
        # forward hook 保存 activation
        target_layer.register_forward_hook(lambda m,i,o: setattr(self,'act',o))
        # backward hook 保存 gradient
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grad',go[0]))

    def __call__(self, x, cls_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls_idx].backward(retain_graph=True)
        g = self.grad[0]  # [C,H,W]
        a = self.act[0]   # [C,H,W]
        w = g.mean(dim=(1,2), keepdim=True) # keepdim=True is correct here for broadcasting
        cam = (w * a).sum(dim=0).detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))  # to (W,H)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ————————————————————————————————————————————— #
# 3) 掩码查找 & Overlay
# ————————————————————————————————————————————— #
def find_mask_for_slice(slice_path):
    dirpath, fname = os.path.split(slice_path)
    basename, _    = os.path.splitext(fname)
    patterns = [
        os.path.join(dirpath, f"{basename}*mask*.*"),
        os.path.join(dirpath, f"{basename}.mask*.*"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None

def apply_heatmap_overlay(orig_img, heatmap_np, original_dimming_factor=0.7, heatmap_alpha=0.6):
    # Resize original image to 224x224 and convert to numpy for overlay
    img_np_resized = np.array(orig_img.resize((224,224))).astype(float)/255.0
    
    # Dim the original image slightly to make heatmap more prominent
    img_np_resized_dimmed = img_np_resized * original_dimming_factor
    
    # Apply jet colormap to heatmap and get RGB (discard alpha)
    heatmap_colored = plt.cm.jet(heatmap_np)[..., :3]
    
    # Proper alpha blending: (1 - alpha) * background + alpha * foreground
    overlay = (1 - heatmap_alpha) * img_np_resized_dimmed + heatmap_alpha * heatmap_colored
    overlay = np.clip(overlay,0,1) # Ensure values are within [0, 1]

    return Image.fromarray((overlay*255).astype(np.uint8))

# ————————————————————————————————————————————— #
# 4) 收集图像
# ————————————————————————————————————————————— #
all_images = []
cls_key = args.target_class
label   = 0 if cls_key=='Healthy' else 1

for root in data_dirs[cls_key]:
    if not root.exists():
        logger.warning(f"Directory not found: {root}")
        continue
    for subj_folder in root.iterdir():
        if not subj_folder.is_dir(): continue
        for img_path in subj_folder.rglob("*"):
            if img_path.suffix.lower() in {".png",".jpg",".jpeg"}: # Defined image extensions here for clarity
                all_images.append({
                    'path': str(img_path),
                    'subject': subj_folder.name,
                    'label': label,
                    'cls': cls_key
                })

if not all_images:
    logger.error("No images found. Check your data paths!")
    sys.exit(1)

logger.info(f"Found {len(all_images)} slices for class {cls_key} across "
            f"{len({i['subject'] for i in all_images})} subjects.")

# ————————————————————————————————————————————— #
# 5) 初始化模型
# ————————————————————————————————————————————— #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Use explicit weights for reproducibility
model_base.fc = nn.Linear(model_base.fc.in_features, 2)
model_base.to(device)

# ————————————————————————————————————————————— #
# 6) LOOCV 微调 + Grad-CAM
# ————————————————————————————————————————————— #
subjects = sorted({i['subject'] for i in all_images})
for subj in tqdm(subjects, desc="LOOCV Subjects"):
    # 划分训练/测试
    train_set = [x for x in all_images if x['subject']!=subj]
    test_set  = [x for x in all_images if x['subject']==subj]
    if not test_set:
        logger.warning(f"No test slices for {subj}, skip.")
        continue

    # 每折一个 fresh model
    fold_model     = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    fold_model.fc  = nn.Linear(fold_model.fc.in_features, 2)
    fold_model.to(device)
    optimizer      = torch.optim.Adam(fold_model.parameters(), lr=1e-4)
    criterion      = nn.CrossEntropyLoss()

    # 3 epochs 微调
    fold_model.train()
    for _ in range(3):
        random.shuffle(train_set)
        for item in train_set:
            try:
                img = Image.open(item['path']).convert('RGB')
                inp = train_transform(img).unsqueeze(0).to(device)
                lbl = torch.tensor([item['label']]).to(device)
                optimizer.zero_grad()
                out = fold_model(inp)
                loss = criterion(out, lbl)
                loss.backward()
                optimizer.step()
            except Exception as e:
                logger.warning(f"Train error {item['path']}: {e}")
    fold_model.eval()

    # Grad-CAM 对 test_set
    cam_generator = GradCAM(fold_model, fold_model.layer4[-1])
    for item in test_set:
        try:
            orig = Image.open(item['path']).convert('RGB')
            inp  = eval_transform(orig).unsqueeze(0).to(device)
            outp = fold_model(inp)
            pred = outp.argmax(dim=1).item()
            heat = cam_generator(inp, pred)

            # --- Masking Logic ---
            heatmap_to_plot = heat # Initialize with full heatmap
            title_suffix = ""
            mask_applied = False
            mask_np_binary = None # Initialize to None for contour drawing later
            
            mask_file = find_mask_for_slice(item['path'])
            if mask_file:
                try:
                    mask_img = Image.open(mask_file).convert('L') # 'L' for grayscale
                    mask_resized = mask_img.resize((224, 224), Image.NEAREST)
                    mask_np = np.array(mask_resized)
                    mask_np_binary = (mask_np > 127).astype(np.float32) # Binarize to 0s and 1s
                    
                    heatmap_masked = heat.copy()
                    heatmap_masked *= mask_np_binary # Element-wise multiplication to apply the mask
                    heatmap_to_plot = heatmap_masked
                    title_suffix = " (masked)"
                    mask_applied = True
                    logger.info(f"Mask applied for {os.path.basename(item['path'])} using {os.path.basename(mask_file)}")
                except Exception as e:
                    logger.warning(f"Error loading or applying mask {mask_file!r} for {item['path']!r}: {e}. Proceeding without mask.")
                    title_suffix = " (mask err)"
            else:
                logger.info(f"No mask found for {item['path']!r}. Proceeding without mask.")
                title_suffix = " (no mask)"

            # --- Generate Overlay ---
            # Use heatmap_to_plot (which might be masked or full) for overlay
            overlay_img_pil = apply_heatmap_overlay(orig, heatmap_to_plot)

            # --- Draw ROI Contour if mask was applied ---
            if mask_applied and mask_np_binary is not None:
                # Convert PIL image to NumPy array for OpenCV drawing
                overlay_np_for_contour = np.array(overlay_img_pil)
                
                # Find contours on the binary mask (need to scale to 255 for cv2.findContours)
                contours, _ = cv2.findContours((mask_np_binary*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours in black on the overlay
                cv2.drawContours(overlay_np_for_contour, contours, -1, (0, 0, 0), 1) # Black, thickness 1
                
                # Convert back to PIL Image
                overlay_img_pil = Image.fromarray(overlay_np_for_contour)


            # --- Plotting ---
            fig, axes = plt.subplots(1,3, figsize=(12,4), dpi=300) # Increased DPI for final output (e.g., 300)

            # Subplot Titles (Simplified and Consistent)
            axes[0].imshow(orig.resize((224,224))); axes[0].axis('off'); axes[0].set_title('Input SIJ slice', fontsize=12)
            im = axes[1].imshow(heat, cmap='jet', vmin=0,vmax=1); axes[1].axis('off'); axes[1].set_title('CAM heatmap', fontsize=12)
            axes[2].imshow(overlay_img_pil); axes[2].axis('off'); axes[2].set_title(f'Overlay{title_suffix}', fontsize=12)

            # Proper Colorbar Placement (far right, fixed coordinates, with ticks)
            # Coordinates are [left, bottom, width, height] in figure fraction
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # x, y, width, height - These are good "far right" coords
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label('Activation', rotation=270, labelpad=12, fontsize=10) # Label text, rotation, labelpad, font size
            cbar.set_ticks([0.0, 0.5, 1.0]) # Set specific ticks
            cbar.ax.tick_params(labelsize=8) # Smaller tick labels

            fig.suptitle(f"Subject: {item['subject']} | Class: {item['cls']}", fontsize=14)
            
            # Use tight_layout with rect for explicit control given manually added cax
            # rect controls [left, bottom, right, top] of the subplot area in figure fraction.
            # Adjust 'right' slightly to make room for the manually added cax.
            plt.tight_layout(rect=[0,0,0.9,0.92]) 

            # Save
            out_dir = base_output_dir/ item['cls'] / item['subject']
            out_dir.mkdir(parents=True, exist_ok=True)
            base = Path(item['path']).stem
            plt.savefig(out_dir/f"gradcam_{item['subject']}_{base}.svg", format='svg') # Save as SVG
            plt.close(fig)

        except Exception as e:
            logger.error(f"Grad-CAM failed {item['path']}: {e}", exc_info=True)

logger.info("✅ Grad-CAM analysis complete.")