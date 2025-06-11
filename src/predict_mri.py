# 文件名: src/predict_mri.py

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# --- 这个脚本需要和您最终使用的训练脚本保持一致 ---
# --- 我将使用您上次提供的更专业的训练脚本中的逻辑 ---

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="使用交叉验证训练好的模型生成对验证集的预测")
    p.add_argument("--data_dir", type=str, default="AS_Finetune_Data_balanced",
                   help="包含影像数据的根目录")
    p.add_argument("--model_dir", type=str, default="models/mri_model",
                   help="存放已训练好的.pth模型文件的文件夹")
    p.add_argument("--output_dir", type=str, default="models/mri_model/mri_preds",
                   help="保存预测结果CSV文件的输出文件夹")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5, help="与训练时相同的KFold折数")
    return p.parse_args()

def get_subject_id_from_path(filepath):
    """从文件路径中提取受试者ID，需要根据您的文件名结构进行调整"""
    # 这是一个示例逻辑，它提取文件名中由下划线分隔的前两个部分
    # 例如: "sub-01_T2TSE_slice_050.png" -> "sub-01_T2TSE"
    # 您需要确保这个逻辑和您训练时使用的逻辑完全一致
    prefix = "_".join(Path(filepath).stem.split("_")[:2])
    return prefix

def build_model(num_classes, device):
    """构建模型架构，必须与训练时完全一致"""
    model = models.resnet50(weights=None) # 加载不带预训练权重的结构
    in_feats = model.fc.in_features
    # 使用和您训练脚本中完全相同的分类头
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feats, num_classes)
    )
    return model.to(device)

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"\nUsing device: {device}\n")

    # 创建输出文件夹
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1) 数据加载
    # 只使用验证集需要的最小变换
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    try:
        full_ds = datasets.ImageFolder(args.data_dir, transform=val_tf)
        print(f"Dataset loaded from '{args.data_dir}', found {len(full_ds)} images.")
    except FileNotFoundError:
        print(f"错误: 找不到数据目录 '{args.data_dir}'。")
        return

    # 2) 重新创建与训练时完全相同的交叉验证分组
    samples = full_ds.samples
    paths  = [p for p, _ in samples]
    labels = [l for _, l in samples]
    groups = [get_subject_id_from_path(p) for p in paths]
    
    gkf = GroupKFold(n_splits=args.n_splits)

    print("\n--- 开始生成各折的预测结果 ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(paths, labels, groups), 1):
        print(f"Processing Fold {fold}/{args.n_splits}...")

        # 3) 加载对应折的已训练模型
        model_path = Path(args.model_dir) / f"best_fold{fold}.pth"
        if not model_path.exists():
            print(f"  错误: 找不到模型文件 {model_path}，跳过此折。")
            continue
            
        model = build_model(len(full_ds.classes), device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 4) 创建该折的验证集DataLoader
        val_ds = Subset(full_ds, val_idx)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        fold_preds = []
        fold_labels = []
        fold_paths = [paths[i] for i in val_idx] # 获取验证集样本的原始路径

        # 5) 执行预测
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                
                outputs = model(imgs)
                # 使用softmax将模型输出转换为概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 我们通常关心的是阳性类别（标签为1）的概率
                positive_class_preds = probabilities[:, 1].cpu().numpy()
                
                fold_preds.extend(positive_class_preds)
                fold_labels.extend(labs.cpu().numpy())

        # 6) 保存结果到CSV文件
        # 创建一个包含文件路径、真实标签和预测概率的DataFrame
        results_df = pd.DataFrame({
            'filepath': fold_paths,
            'subject_id': [get_subject_id_from_path(p) for p in fold_paths],
            'labels': fold_labels,
            'preds': fold_preds
        })
        
        output_csv_path = Path(args.output_dir) / f"fold_{fold-1}_predictions.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"  ✅ 预测结果已保存至: {output_csv_path}")

    print("\n--- 所有预测文件已生成完毕 ---")

if __name__ == "__main__":
    main()
