#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_data.py - 实际版
功能：生成SHAP值分析和DCA曲线，用真实病人数据区分AS与非AS
"""

import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ─────────── 1. 加载真实数据 ───────────
def load_real_data(csv_path, features, target_label):
    df = pd.read_csv(csv_path)

    # 创建二分类标签：AS vs 其他
    df['label'] = (df['Disease'] == target_label).astype(int)

    # 仅保留需要列
    df = df[features + ['label']].dropna()

    # 对非数值列做编码
    for col in features:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# ─────────── 2. 模型结构 ───────────
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

# ─────────── 3. SHAP分析函数 ───────────
def analyze_shap(model, background, x_test, feature_names, save_dir):
    print("开始SHAP分析...")
    os.makedirs(save_dir, exist_ok=True)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_test)

    shap_df = pd.DataFrame(shap_values[1], columns=feature_names)
    shap_df.to_csv(os.path.join(save_dir, 'shap_values.csv'))

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[1],
        pd.DataFrame(x_test.cpu().numpy(), columns=feature_names),
        show=False
    )
    plt.savefig(os.path.join(save_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    print(f"SHAP分析完成，结果保存在: {save_dir}")
    return shap_values

# ─────────── 4. DCA绘图函数 ───────────
def plot_dca(y_true, y_prob, save_dir):
    print("绘制DCA曲线...")
    thresholds = np.linspace(0.01, 0.99, 100)
    benefits = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        benefits.append(net_benefit)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, benefits, label='Model')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'dca_curve.png'), bbox_inches='tight')
    plt.close()
    print(f"DCA曲线已保存在: {save_dir}")

# ─────────── 5. 主函数 ───────────
def main():
    print("开始数据分析...")

    # 设定路径
    csv_path = "/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS/data/raw_lab_data/Raw_Lab_Dataset.csv"
    save_dir = os.path.expanduser('~/Desktop/analysis_results')
    os.makedirs(save_dir, exist_ok=True)

    # 设定特征列和目标类别
    feature_names = [
        'Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27',
        'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4'
    ]
    target_label = 'Ankylosing Spondylitis'

    # 加载数据
    df = load_real_data(csv_path, feature_names, target_label)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values

    # 转换为Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    # 创建并训练模型
    model = SimpleNet(X.shape[1]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # SHAP分析
    background = X_tensor[:100]
    test_data = X_tensor[100:]
    shap_values = analyze_shap(model, background, test_data, feature_names, save_dir)

    # 模型预测 + DCA
    with torch.no_grad():
        logits = model(test_data)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    plot_dca(y[100:], probs, save_dir)

    print("\n分析完成！所有结果已保存到:", save_dir)
    print("生成的文件：")
    print("✔ shap_values.csv")
    print("✔ shap_summary.png")
    print("✔ dca_curve.png")

if __name__ == "__main__":
    main()