#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shap_dca.py - 基于真实疾病标签的 SHAP + DCA 可视化脚本
路径适配：/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/
"""

import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. 路径与列配置
CSV_PATH = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/raw_lab_data/Raw_Lab_Dataset.csv"
SAVE_DIR = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/analysis_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = [
    'Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27',
    'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4'
]
TARGET = 'Disease'  # 字符型标签，例如 'Rheumatoid Arthritis'

# 2. 模型结构
class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# 3. 主流程
def main():
    # 读取数据
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])  # 去除缺失值

    # 特征 + 标签
    X = df[FEATURES].values

    # 自动将疾病字符串转为数字标签（如 RA → 0, Lupus → 1, etc.）
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].values)

    # Tensor 转换
    device = torch.device("cpu")
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # 简单两层网络训练
    model = SimpleNet(X.shape[1]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # 划分背景与测试集
    background = X_tensor[:100]
    test_data = X_tensor[100:]
    y_test = y[100:]

    # SHAP 可解释性分析
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data)

    shap.summary_plot(
        shap_values[1],
        pd.DataFrame(test_data.cpu().numpy(), columns=FEATURES),
        show=False
    )
    plt.savefig(os.path.join(SAVE_DIR, "shap_summary.png"), bbox_inches='tight')
    plt.close()

    # DCA 决策曲线
    with torch.no_grad():
        probs = torch.softmax(model(test_data), dim=1)[:, 1].cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, 100)
    benefits = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = np.sum((pred == 1) & (y_test == 1))
        fp = np.sum((pred == 1) & (y_test == 0))
        n = len(y_test)
        net_benefit = (tp / n) - (fp / n) * (t / (1 - t))
        benefits.append(net_benefit)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, benefits, label='Model')
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "dca_curve.png"), bbox_inches='tight')
    plt.close()

    print("✅ 分析完成！图已保存于:", SAVE_DIR)
    print("🎯 标签编码映射:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} = {cls}")

if __name__ == "__main__":
    main()