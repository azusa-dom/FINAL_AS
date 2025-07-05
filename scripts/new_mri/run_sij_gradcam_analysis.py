#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sij_gradcam_analysis.py

Leave-One-Subject-Out + Grad-CAM 可解释分析
(6 vs 2 小样本，用少量增强+Fine-tune，输出专业期刊级三联对比图)
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

# ---------------------- 配置 ----------------------
# 回到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 数据路径
healthy_dirs = [
    os.path.join(project_root, 'data', 'mri_health', 'health1'),
    os.path.join(project_root, 'data', 'mri_health', 'health2')
]
as_dir = os.path.join(project_root, 'data', 'mri_AS')
# 输出目录
output_dir = os.path.join(project_root, 'results', 'grad_cam_outputs')
os.makedirs(output_dir, exist_ok=True)
# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------- 变换 ----------------------
# 训练时轻度增强
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
# 推理不增强
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------------- GradCAM ----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.grad = None
        self.act = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0]))
    def __call__(self, x, cls_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls_idx].backward(retain_graph=True)
        g = self.grad[0]   # [C,H,W]
        a = self.act[0]    # [C,H,W]
        w = g.mean(dim=(1,2), keepdim=True)
        cam = (w * a).sum(dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
        return cam

# ---------------------- 收集图像 ----------------------
all_images = []
# 健康组
for lbl, d in enumerate(healthy_dirs):
    if not os.path.isdir(d):
        print(f"Error: Healthy dir not found: {d}", file=sys.stderr)
        sys.exit(1)
    for subj in os.listdir(d):
        subdir = os.path.join(d, subj)
        if os.path.isdir(subdir):
            for fn in os.listdir(subdir):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    all_images.append({'path':os.path.join(subdir,fn),'subject':subj,'label':0,'cls':'Healthy'})
# AS 组
if not os.path.isdir(as_dir):
    print(f"Error: AS dir not found: {as_dir}", file=sys.stderr)
    sys.exit(1)
for subj in os.listdir(as_dir):
    subdir = os.path.join(as_dir, subj)
    if os.path.isdir(subdir):
        for fn in os.listdir(subdir):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                all_images.append({'path':os.path.join(subdir,fn),'subject':subj,'label':1,'cls':'AS'})
# 检查
if not all_images:
    print("No images found.", file=sys.stderr)
    sys.exit(1)
print(f"🔍 Total slices: {len(all_images)}, subjects: {len(set(i['subject'] for i in all_images))}")

# ---------------------- 模型 ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
target_layer = model.layer4[-1]

# ---------------------- LOOCV + GradCAM ----------------------
for subj in tqdm(sorted(set(i['subject'] for i in all_images)), desc='Subject LOOCV'):
    # 划分
    train = [i for i in all_images if i['subject']!=subj]
    test = [i for i in all_images if i['subject']==subj]
    # Fine-tune
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for ep in range(3):
        random.shuffle(train)
        for item in train:
            img = Image.open(item['path']).convert('RGB')
            inp = train_transform(img).unsqueeze(0).to(device)
            lbl = torch.tensor([item['label']], device=device)
            optimizer.zero_grad(); out = model(inp); loss=criterion(out,lbl); loss.backward(); optimizer.step()
    # 可视化
    model.eval()
    camer = GradCAM(model, target_layer)
    for item in test:
        orig = Image.open(item['path']).convert('RGB')
        inp = eval_transform(orig).unsqueeze(0).to(device)
        out = model(inp)
        cls_idx = out.argmax(dim=1).item()
        cam = camer(inp, cls_idx)
        # 期刊级三联图
        ori_gray = np.array(orig.resize((224,224)))
        fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(12,4))
        ax0.imshow(ori_gray, cmap='gray'); ax0.set_title('Original'); ax0.axis('off')
        im1 = ax1.imshow(cam, cmap='jet', vmin=0, vmax=1); ax1.set_title('Grad-CAM'); ax1.axis('off')
        ax2.imshow(ori_gray, cmap='gray');
        overlay = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET);
        ax2.imshow(overlay, alpha=0.5); ax2.set_title('Overlay'); ax2.axis('off')
        cbar = fig.colorbar(im1, ax=[ax0,ax1,ax2], location='right', fraction=0.046, pad=0.04)
        cbar.set_label('Activation', rotation=270, labelpad=15)
        fig.suptitle(f"Subject {item['subject']} | {item['cls']}", fontsize=16)
        outfn = f"{item['subject']}_{os.path.splitext(os.path.basename(item['path']))[0]}_academic.png"
        fig.savefig(os.path.join(output_dir,outfn), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
print("✅ Complete. Outputs in:", output_dir)