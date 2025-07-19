import os
import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union, Dict
from sklearn.preprocessing import StandardScaler

class ClinicalNetPro(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path("/Users/hydra/Downloads/MRI-AS-MRI_AS/FINAL_AS")

def load_fold_data(fold: int, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载指定折次的训练和验证数据"""
    train_path = data_dir / f"fold_{fold}_train.csv"
    val_path = data_dir / f"fold_{fold}_val.csv"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"找不到fold {fold}的数据文件")
        
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    return train_data, val_data

def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """准备模型输入数据"""
    # 假设最后一列是标签，其他都是特征
    X_train = train_df.iloc[:, :-1].values
    X_val = val_df.iloc[:, :-1].values
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val

def load_model(fold: int, input_dim: int) -> ClinicalNetPro:
    """加载指定折次的模型"""
    project_root = get_project_root()
    model_path = project_root / "results" / "clinical" / "clinical_model" / f"best_model_fold_{fold}.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    model = ClinicalNetPro(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    return model

def compute_shap_values(model: ClinicalNetPro, background_data: np.ndarray, test_data: np.ndarray, 
                       feature_names: List[str]) -> Dict:
    """计算SHAP值"""
    # 创建解释器
    explainer = shap.DeepExplainer(model, torch.FloatTensor(background_data))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(torch.FloatTensor(test_data))
    
    # 如果shap_values是列表，取第一个元素
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    return {
        'values': shap_values,
        'feature_names': feature_names,
        'data': test_data
    }

def plot_shap_summary(shap_dict: Dict, fold: int, output_dir: Path):
    """绘制SHAP汇总图"""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_dict['values'],
        shap_dict['data'],
        feature_names=shap_dict['feature_names'],
        show=False
    )
    plt.title(f'SHAP Summary Plot - Fold {fold}')
    
    # 保存图片
    output_path = output_dir / f"shap_summary_fold_{fold}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_interactions(shap_dict: Dict, fold: int, output_dir: Path):
    """绘制SHAP交互图"""
    # 获取最重要的特征索引
    mean_abs_shap = np.abs(shap_dict['values']).mean(0)
    top_feature_idx = np.argmax(mean_abs_shap)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature_idx,
        shap_dict['values'],
        shap_dict['data'],
        feature_names=shap_dict['feature_names'],
        show=False
    )
    plt.title(f'SHAP Interaction Plot - Fold {fold}')
    
    # 保存图片
    output_path = output_dir / f"shap_interaction_fold_{fold}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    try:
        # 设置项目路径
        project_root = get_project_root()
        data_dir = project_root / "data" / "processed_clinical"
        output_dir = project_root / "results" / "clinical" / "shap_plots"
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 项目根目录: {project_root}")
        print(f"📁 数据目录: {data_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        # 处理每个fold
        for fold in range(5):
            print(f"\n🔄 处理 Fold {fold}")
            
            try:
                # 加载数据
                train_data, val_data = load_fold_data(fold, data_dir)
                print(f"✅ 成功加载fold {fold}的数据")
                
                # 准备特征名称
                feature_names = train_data.columns[:-1].tolist()
                
                # 准备数据
                X_train, X_val = prepare_data(train_data, val_data)
                print(f"✅ 数据预处理完成")
                
                # 加载模型
                model = load_model(fold, X_train.shape[1])
                print(f"✅ 模型加载成功")
                
                # 计算SHAP值
                shap_dict = compute_shap_values(model, X_train, X_val, feature_names)
                print(f"✅ SHAP值计算完成")
                
                # 绘制图表
                plot_shap_summary(shap_dict, fold, output_dir)
                plot_shap_interactions(shap_dict, fold, output_dir)
                print(f"✅ 图表生成完成")
                
            except Exception as e:
                print(f"❌ 处理fold {fold}时出错: {str(e)}")
                continue
        
        print("\n✅ 所有处理完成!")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
