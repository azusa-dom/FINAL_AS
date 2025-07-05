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

# ————————————————————————————————————————————— #
# 1) 配置：项目根目录、数据路径、输出目录、随机种子、轻度增强
# ————————————————————————————————————————————— #
# project_root 指向 FINAL_AS 根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
# 输出目录
output_dir = os.path.join(project_root, "results", "grad_cam_outputs")
os.makedirs(output_dir, exist_ok=True)
#print(f"Grad-CAM outputs will be saved to: {output_dir}")

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
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ————————————————————————————————————————————— #
# 3) 收集所有图像样本
# ————————————————————————————————————————————— #
all_images = []
for label, cls_key in enumerate(['Healthy','AS']):
    for root in data_dirs[cls_key]:
        if not os.path.isdir(root):
            print(f"Error: Directory not found: {root}", file=sys.stderr)
            sys.exit(1)
        for subj in os.listdir(root):
            subdir = os.path.join(root, subj)
            if not os.path.isdir(subdir):
                continue
            for fn in os.listdir(subdir):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    all_images.append({
                        'path': os.path.join(subdir, fn),
                        'subject': subj,
                        'label': label,
                        'cls': cls_key
                    })
# 检查
if not all_images:
    print("No images found in data directories.", file=sys.stderr)
    sys.exit(1)

# ————————————————————————————————————————————— #
# 4) 初始化模型 & 目标层
# ————————————————————————————————————————————— #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# 替换最后分类层为2类输出
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
target_layer = model.layer4[-1]

# ————————————————————————————————————————————— #
# 5) 留一法 Fine-tune + Grad-CAM 输出期刊级三联图
# ————————————————————————————————————————————— #
print(f"🔍 Loaded {len(all_images)} image slices across {len(set(i['subject'] for i in all_images))} subjects.")
subjects = sorted(set(i['subject'] for i in all_images))
for subj in tqdm(subjects, desc='Subject LOOCV'):
    # 划分训练/测试
    train_set = [i for i in all_images if i['subject'] != subj]
    test_set = [i for i in all_images if i['subject'] == subj]
    # Fine-tune 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(3):
        random.shuffle(train_set)
        for item in train_set:
            img = Image.open(item['path']).convert('RGB')
            inp = train_transform(img).unsqueeze(0).to(device)
            lbl = torch.tensor([item['label']], device=device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
    # 生成 Grad-CAM 并可视化
    model.eval()
    camer = GradCAM(model, target_layer)
    for item in test_set:
        original_img = Image.open(item['path']).convert('RGB')
        inp = eval_transform(original_img).unsqueeze(0).to(device)
        out = model(inp)
        cls_pred = out.argmax(dim=1).item()
        heatmap = camer(inp, cls_pred)
        # 专业可视化三联
        ori_gray = np.array(original_img.resize((224,224)))
        # 三联subplot
        fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(12,4))
        ax0.imshow(ori_gray, cmap='gray')
        ax0.set_title('Original')
        ax0.axis('off')
        im1 = ax1.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        ax1.set_title('Grad-CAM')
        ax1.axis('off')
        ax2.imshow(ori_gray, cmap='gray')
        ax2.imshow(cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET), alpha=0.5)
        ax2.set_title('Overlay')
        ax2.axis('off')
        # colorbar
        cbar = fig.colorbar(im1, ax=[ax0,ax1,ax2], location='right', fraction=0.046, pad=0.04)
        cbar.set_label('Activation', rotation=270, labelpad=15)
        fig.suptitle(f"Subject: {item['subject']} | Class: {item['cls']}", fontsize=16)
        out_name = f"{item['subject']}_{os.path.basename(item['path']).split('.')[0]}_academic.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
print("✅ Grad-CAM analysis complete. Results saved in:", output_dir)