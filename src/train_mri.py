# 文件: src/train_mri.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import time

# 确保可以从 src 目录导入我们自己的 dataset 模块
try:
    from dataset import ASFineTuneDataset 
except ImportError:
    print("错误：无法从 src.dataset 导入 ASFineTuneDataset。请确保文件路径和名称正确。")
    exit()


def create_mri_model():
    """
    加载一个在ImageNet上预训练的ResNet-50模型，并修改其最终层以适应我们的二分类任务。
    """
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    
    # 获取模型全连接层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 替换为新的全连接层，输出为2 (因为我们有'AS'和'Healthy'两个类别)
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model

def main():
    """
    主函数，包含了数据加载、交叉验证、训练和评估的全部流程。
    """
    # --- 1. 配置区域 ---
    # 定义超参数
    N_SPLITS = 5        # 交叉验证的折数 (对于23个样本，5折是常用选择)
    NUM_EPOCHS = 30     # 训练轮次
    BATCH_SIZE = 4      # 批次大小 (根据您的GPU显存调整)
    LEARNING_RATE = 1e-4 # 学习率
    DATA_DIR = 'AS_Finetune_Data' # 存放影像数据的根目录
    RANDOM_STATE = 42   # 设置随机种子以保证结果可复现
    
    # 设置设备 (优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    # --- 2. 数据加载与预处理 ---
    # 定义数据预处理和增强
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载完整的数据集
    try:
        full_dataset = ASFineTuneDataset(root_dir=DATA_DIR, transform=data_transforms)
        if len(full_dataset) == 0:
            print(f"错误: 在 '{DATA_DIR}' 目录中没有找到任何图片。请检查路径和文件结构。")
            return
        print(f"INFO: Dataset loaded successfully. Found {len(full_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请确保您已经在项目根目录下创建了 '{DATA_DIR}' 文件夹，并且其中包含 '0_Healthy' 和 '1_AS' 子文件夹。")
        return
        
    # --- 3. 交叉验证训练循环 ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    labels = [sample[1] for sample in full_dataset.samples]

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
        
        # 为当前折创建数据采样器和加载器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
        
        # 每一折都创建一个全新的模型
        model = create_mri_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 训练循环
        start_time = time.time()
        model.train()
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_idx)
            # print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

        # 验证循环
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        fold_accuracies.append(accuracy)
        fold_time = time.time() - start_time
        print(f"  Validation Accuracy for Fold {fold+1}: {accuracy:.2f}% | Time: {fold_time:.2f}s")
        
        # (可选) 保存每一折的模型
        # os.makedirs('models/mri_model', exist_ok=True)
        # torch.save(model.state_dict(), f'models/mri_model/mri_model_fold_{fold+1}.pth')

    # --- 4. 总结交叉验证结果 ---
    print("\n========== Cross-Validation Summary ==========")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Average Validation Accuracy ({N_SPLITS}-fold): {mean_accuracy:.2f}%")
    print(f"Standard Deviation: +/- {std_accuracy:.2f}%")
    print("==========================================")

if __name__ == '__main__':
    main()
