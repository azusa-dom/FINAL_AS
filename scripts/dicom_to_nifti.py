import argparse
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- 我们需要从你的项目中导入模型定义 ---
# 为了让这个脚本独立运行，我们把模型定义也复制过来
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

# --- 你的绘图函数 (保持不变，写得很好) ---

def plot_shap_summary(model, X_test_df, save_dir):
    """
    根据已训练的模型和测试数据生成并保存SHAP摘要图。
    """
    print("📊 正在生成 SHAP 图...")
    # SHAP 需要一个返回概率的函数
    def predict_proba_for_shap(x):
        # 转换为 tensor
        if isinstance(x, pd.DataFrame):
            x = x.values
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # 模型预测
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # 使用 KernelExplainer，它适用于任何模型
    # 我们对一小部分背景数据进行采样以加快速度
    background_data = shap.sample(X_test_df, 100) 
    explainer = shap.KernelExplainer(predict_proba_for_shap, background_data)
    
    # 计算测试数据的SHAP值
    shap_values = explainer.shap_values(X_test_df)

    # 绘制摘要图 (我们关心正类 '1' 的SHAP值)
    plt.figure()
    shap.summary_plot(shap_values[1], X_test_df, show=False, plot_type="dot")
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "shap_summary_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP 图已保存 → {save_path}")


def plot_decision_curve(y_true, y_pred_probs, save_dir):
    """
    生成并保存决策曲线分析图。
    """
    print("📈 正在生成 DCA 曲线图...")
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit_model = []
    
    n = len(y_true)
    for t in thresholds:
        # 当概率超过阈值t时，我们认为是阳性预测
        y_pred = (y_pred_probs >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit_model.append((tp / n) - (fp / n) * (t / (1 - t)))

    # 计算 "全治疗" 和 "不治疗" 策略的净获益
    p_all = np.mean(y_true)
    net_benefit_all = p_all - (1 - p_all) * (thresholds / (1 - thresholds))
    net_benefit_none = np.zeros_like(thresholds)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label="Model", color="crimson", linewidth=2)
    plt.plot(thresholds, net_benefit_all, label="Treat All", linestyle="--", color="black")
    plt.plot(thresholds, net_benefit_none, label="Treat None", linestyle=":", color="gray")
    plt.ylim(min(np.nanmin(net_benefit_model), -0.1), 0.5) # 通常将Y轴上限设为0.5
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA)")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "dca_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ DCA 图已保存 → {save_path}")


# --- 主逻辑：加载真实数据和模型 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用真实数据和模型生成SHAP和DCA图。")
    parser.add_argument("--model-path", type=str, required=True, help="已训练模型的路径 (e.g., 'results/final_run/best_model_fold_0.pth')")
    parser.add_argument("--data-path", type=str, required=True, help="用于评估的数据路径 (e.g., 'processed_data/fold_0_val.csv')")
    parser.add_argument("--save-dir", type=str, required=True, help="保存图表的目录 (e.g., 'results/final_run/plots')")
    args = parser.parse_args()
    
    print("--- 开始使用真实结果生成高级图表 ---")
    
    # 1. 加载数据
    print(f"📂 正在加载数据: {args.data_path}")
    val_df = pd.read_csv(args.data_path)
    
    # 分离特征、标签和ID
    id_column = 'Patient_ID'
    label_column = 'label'
    y_true = val_df[label_column].values
    feature_cols = [c for c in val_df.columns if c not in [id_column, label_column]]
    X_val_df = val_df[feature_cols]

    # 2. 加载模型
    print(f"🧠 正在加载模型: {args.model_path}")
    input_dim = len(feature_cols)
    num_classes = len(val_df[label_column].unique())
    model = SimpleMLP(input_size=input_dim, hidden_size=64, output_size=num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.eval() # 设为评估模式

    # 3. 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. 获取模型的预测概率
    print("🚀 正在获取模型预测...")
    X_val_tensor = torch.tensor(X_val_df.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_val_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()
    
    # 我们需要正类 (class 1) 的概率
    positive_class_probs = probabilities[:, 1]
    
    # 5. 调用绘图函数
    # 为 SHAP 图使用 DataFrame
    plot_shap_summary(model, X_val_df, args.save_dir)

    # 为 DCA 图使用 NumPy 数组
    plot_decision_curve(y_true, positive_class_probs, args.save_dir)

    print("\n✅ 所有高级图表已成功生成！")