import argparse
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def plot_shap_summary(model, X_test, feature_names, save_dir):
    """
    根据已训练的模型和测试数据生成并保存SHAP摘要图。
    
    Args:
        model: 任何与SHAP兼容的已训练模型对象。
        X_test (pd.DataFrame or np.ndarray): 用于解释的测试特征数据。
        feature_names (list): 特征名称列表。
        save_dir (str): 保存图像的目录。
    """
    print("📊 正在生成 SHAP 图...")
    # 创建一个解释器
    # 对于树模型（如XGBoost），使用 shap.TreeExplainer 会更快
    if hasattr(model, 'predict_proba'):
        explainer = shap.KernelExplainer(model.predict_proba, X_test)
    else:
        explainer = shap.KernelExplainer(model.predict, X_test)

    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)

    # 确保 X_test 是一个DataFrame以便正确显示特征名称
    if not isinstance(X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test

    # 绘制摘要图（beeswarm plot更具信息量）
    plt.figure()
    # 对于二分类问题，通常我们只关心正类的SHAP值
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test_df, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "shap_summary_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ SHAP 图已保存 → {save_path}")


def plot_decision_curve(y_true, y_pred_probs, save_dir):
    """
    根据真实标签和模型预测概率生成并保存决策曲线分析图。

    Args:
        y_true (np.ndarray): 真实标签 (0 或 1)。
        y_pred_probs (np.ndarray): 模型对正类的预测概率。
        save_dir (str): 保存图像的目录。
    """
    print("📈 正在生成 DCA 曲线图...")
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit_model = []
    
    # 计算模型的净获益
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        n = len(y_true)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit_model.append((tp / n) - (fp / n) * (t / (1 - t)))

    # 计算 "Treat All" 和 "Treat None" 策略的净获益
    p_all = np.mean(y_true)
    net_benefit_all = p_all - (1 - p_all) * (thresholds / (1 - thresholds))
    net_benefit_none = np.zeros_like(thresholds)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label="Model", color="crimson")
    plt.plot(thresholds, net_benefit_all, label="Treat All", linestyle="--", color="black")
    plt.plot(thresholds, net_benefit_none, label="Treat None", linestyle=":", color="gray")
    plt.ylim(min(np.nanmin(net_benefit_model), -0.1), max(np.nanmax(net_benefit_model), np.nanmax(net_benefit_all), 0.5))
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA)")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "dca_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ DCA 图已保存 → {save_path}")

if __name__ == "__main__":
    # --- 这是一个如何使用这些函数的演示 ---
    print("--- 开始绘图脚本演示 ---")
    
    # 1. 创建模拟数据和模型 (在您的真实代码中，您会加载这些)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42, stratify=y)
    
    # 假设这是一个您已经训练好的模型
    print("正在训练一个模拟模型...")
    # model = YourFusionModel().fit(X_train, y_train) 
    model = LogisticRegression().fit(X_train, y_train)
    print("模拟模型训练完成。")

    # 2. 指定保存目录
    save_directory = "results_plots_demo"
    os.makedirs(save_directory, exist_ok=True)

    # 3. 调用绘图函数
    # 获取测试集的预测概率用于DCA
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # 生成 SHAP 图
    # 注意：对于大数据集，SHAP可能很慢，可以对X_test进行采样
    plot_shap_summary(model, X_test, feature_names, save_directory)

    # 生成 DCA 图
    plot_decision_curve(y_test, test_probabilities, save_directory)

    print("\n✅ 绘图演示完成。")

