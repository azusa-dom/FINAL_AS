import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np  # 新增导入
import os
import argparse
from tqdm import tqdm
from .dataset import TabularDataset
from .models import SimpleResNet, SimpleCNN, SimpleMLP
from .utils import get_kfold_strafied_sampler, get_class_weights


def train(args):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建 k-fold 数据加载器
    kfold_loader = get_kfold_strafied_sampler(args.data_dir, n_splits=5)
    
    # --- 新增代码: 创建用于保存预测结果的目录 ---
    preds_output_dir = os.path.join(args.model_dir, 'clinical_preds')
    os.makedirs(preds_output_dir, exist_ok=True)
    # --- 代码结束 ---

    for fold, (train_loader, val_loader) in enumerate(kfold_loader):
        print(f"\n===== Fold {fold} =====")

        # 模型初始化
        if args.model_name == 'resnet':
            model = SimpleResNet(num_classes=2).to(device) # 假设是二分类
        elif args.model_name == 'cnn':
            model = SimpleCNN(num_classes=2).to(device)
        else:
            # 获取输入特征维度
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].shape[1]
            model = SimpleMLP(input_dim=input_dim, num_classes=2).to(device)
        
        # 损失函数和优化器
        class_weights = get_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [T]"):
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            
            # --- 新增代码: 初始化用于收集当前fold预测和标签的列表 ---
            fold_true_labels = []
            fold_pred_logits = []
            # --- 代码结束 ---
            
            with torch.no_grad():
                for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [V]"):
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # --- 新增代码: 收集预测和标签 ---
                    fold_true_labels.append(labels.cpu().numpy())
                    fold_pred_logits.append(outputs.cpu().numpy())
                    # --- 代码结束 ---

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(args.model_dir, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model for fold {fold} saved to {model_save_path}")

        # --- 新增代码: 在fold所有epoch结束后, 保存该fold的最佳预测结果 ---
        print(f"Saving predictions for Fold {fold}...")
        # 将列表转换为Numpy数组
        fold_true_labels = np.concatenate(fold_true_labels, axis=0)
        fold_pred_logits = np.concatenate(fold_pred_logits, axis=0)
        
        # 获取logits的维度作为列名
        num_classes = fold_pred_logits.shape[1]
        logit_columns = [f'logit_{i}' for i in range(num_classes)]
        
        # 创建DataFrame
        df_preds = pd.DataFrame(fold_pred_logits, columns=logit_columns)
        df_preds['true_label'] = fold_true_labels
        
        # 定义保存路径并保存
        preds_save_path = os.path.join(preds_output_dir, f'fold_{fold}_predictions.csv')
        df_preds.to_csv(preds_save_path, index=False)
        print(f"✅ Predictions for fold {fold} saved to {preds_save_path}")
        # --- 代码结束 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple model on clinical data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the processed fold data.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the trained models.")
    parser.add_argument("--model_name", type=str, choices=['mlp', 'cnn', 'resnet'], default='resnet', help="Model to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    # 确保模型保存目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    
    train(args)
