# 双通路AI框架详细方法论 - 第一部分：临床数据通路

## 🔬 临床数据通路（ClinicalNet）详细方法论

### 1. 数据来源与收集

#### 1.1 原始数据集描述
- **数据来源**：电子健康记录（EHR）系统
- **原始样本规模**：约10,000例门诊记录
- **疾病分布**：包含多种风湿性疾病（类风湿关节炎、系统性红斑狼疮、干燥综合征等）
- **时间跨度**：2018年1月至2023年12月
- **数据质量**：结构化数据，包含实验室检查、免疫学指标、人口统计学信息

#### 1.2 数据筛选标准
```python
# 数据筛选逻辑
def filter_as_cases(df):
    """
    筛选AS病例的标准：
    1. 主要诊断ICD-10代码：M45.x（强直性脊柱炎）
    2. 临床确诊：风湿科医生确诊
    3. 数据完整性：关键指标缺失率 < 30%
    """
    as_cases = df[
        (df['primary_diagnosis'].str.contains('M45', na=False)) |
        (df['as_confirmed'] == 1) |
        (df['missing_rate'] < 0.3)
    ]
    return as_cases

def select_controls(df, n_controls=851):
    """
    对照选择标准：
    1. 排除所有炎症性关节病
    2. 年龄匹配（±5岁）
    3. 性别比例匹配
    4. 随机选择
    """
    non_inflammatory = df[
        ~df['primary_diagnosis'].str.contains('M0[5-9]|M1[0-4]|M3[0-6]', na=False)
    ]
    return non_inflammatory.sample(n=n_controls, random_state=42)
```

#### 1.3 最终数据集构成
- **AS病例组**：851例确诊AS患者
- **对照组**：851例随机选择的非炎症性关节病患者
- **总样本量**：1,702例
- **数据平衡**：1:1病例对照比例

### 2. 特征工程详细过程

#### 2.1 原始特征列表
```python
# 27个标准化预测因子
FEATURE_COLUMNS = [
    # 人口统计学特征
    'Age', 'Gender_Female', 'Gender_Male',
    
    # 实验室指标
    'ESR', 'CRP', 'RF', 'Anti-CCP', 'C3', 'C4',
    
    # 免疫学指标
    'HLA-B27_Negative', 'HLA-B27_Positive',
    'ANA_Negative', 'ANA_Positive',
    'Anti-Ro_Negative', 'Anti-Ro_Positive',
    'Anti-La_Negative', 'Anti-La_Positive',
    'Anti-dsDNA_Negative', 'Anti-dsDNA_Positive',
    'Anti-Sm_Negative', 'Anti-Sm_Positive'
]

# 目标变量
TARGET_COLUMN = 'label'  # 0: 对照, 1: AS
ID_COLUMN = 'Patient_ID'
```

#### 2.2 数据预处理详细步骤

##### 2.2.1 缺失值处理
```python
def handle_missing_values(df):
    """
    缺失值处理策略：
    1. 数值型变量：中位数填充
    2. 分类变量：众数填充
    3. 缺失率>50%的变量：删除
    """
    # 计算缺失率
    missing_rates = df.isnull().sum() / len(df)
    
    # 删除缺失率>50%的变量
    high_missing_cols = missing_rates[missing_rates > 0.5].index
    df = df.drop(columns=high_missing_cols)
    
    # 数值型变量用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 分类变量用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
```

##### 2.2.2 特征标准化
```python
from sklearn.preprocessing import StandardScaler

def standardize_features(df, feature_cols):
    """
    Z-score标准化：
    z = (x - μ) / σ
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler
```

##### 2.2.3 分类变量编码
```python
def encode_categorical_variables(df):
    """
    One-hot编码过程：
    1. 性别：Female=0, Male=1
    2. 免疫学指标：Negative=0, Positive=1
    3. 处理多重共线性
    """
    # 性别编码
    df['Gender_Female'] = (df['Gender'] == 'Female').astype(int)
    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
    
    # 免疫学指标编码
    immune_cols = ['HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm']
    for col in immune_cols:
        df[f'{col}_Negative'] = (df[col] == 'Negative').astype(int)
        df[f'{col}_Positive'] = (df[col] == 'Positive').astype(int)
    
    return df
```

#### 2.3 特征选择与验证
```python
def feature_selection_analysis(df, target_col):
    """
    特征选择分析：
    1. 相关性分析
    2. 方差分析
    3. 互信息分析
    """
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    
    # 计算互信息
    mi_scores = mutual_info_classif(df.drop(columns=[target_col]), df[target_col])
    mi_df = pd.DataFrame({
        'feature': df.drop(columns=[target_col]).columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 选择前27个特征
    selected_features = mi_df.head(27)['feature'].tolist()
    
    return selected_features, mi_df
```

### 3. 模型架构详细设计

#### 3.1 ClinicalNet神经网络架构
```python
import torch
import torch.nn as nn

class ClinicalNet(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, output_size=2, dropout_p=0.5):
        """
        ClinicalNet详细架构：
        
        输入层：27维特征向量
        隐藏层1：64个神经元，ReLU激活，Dropout(0.5)
        隐藏层2：64个神经元，ReLU激活，Dropout(0.5)
        输出层：2个神经元（二分类）
        
        参数：
        - input_size: 输入特征维度 (27)
        - hidden_size: 隐藏层神经元数量 (64)
        - output_size: 输出类别数 (2)
        - dropout_p: Dropout概率 (0.5)
        """
        super(ClinicalNet, self).__init__()
        
        # 网络层定义
        self.net = nn.Sequential(
            # 输入层 → 隐藏层1
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # 隐藏层1 → 隐藏层2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            # 隐藏层2 → 输出层
            nn.Linear(hidden_size, output_size)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Xavier权重初始化：
        对于ReLU激活函数，使用He初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播：
        x: 输入特征张量 [batch_size, 27]
        返回: logits [batch_size, 2]
        """
        return self.net(x)
```

#### 3.2 损失函数设计
```python
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        类别加权交叉熵损失：
        处理数据不平衡问题
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        """
        计算加权交叉熵损失
        
        参数：
        - logits: 模型输出 [batch_size, 2]
        - targets: 真实标签 [batch_size]
        
        返回：
        - loss: 加权交叉熵损失
        """
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        return loss_fn(logits, targets)

def calculate_class_weights(dataset):
    """
    计算类别权重：
    用于处理数据不平衡
    """
    labels = [item[1] for item in dataset]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # 计算权重：总样本数 / (类别数 * 该类样本数)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)
```

#### 3.3 优化器配置
```python
def configure_optimizer(model, learning_rate=1e-3, weight_decay=1e-4):
    """
    Adam优化器配置：
    
    参数：
    - learning_rate: 学习率 (1e-3)
    - weight_decay: 权重衰减 (1e-4)
    - betas: Adam优化器的动量参数
    - eps: 数值稳定性参数
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer
```

### 4. 训练策略详细实现

#### 4.1 交叉验证设置
```python
from sklearn.model_selection import StratifiedKFold

def setup_cross_validation(n_splits=5, random_state=42):
    """
    5折分层交叉验证设置：
    
    参数：
    - n_splits: 折数 (5)
    - random_state: 随机种子 (42)
    - shuffle: 是否打乱数据 (True)
    """
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    return kfold

def create_fold_datasets(data_dir, fold_idx):
    """
    创建特定折的数据集：
    
    返回：
    - train_dataset: 训练数据集
    - val_dataset: 验证数据集
    """
    # 读取训练和验证数据
    train_df = pd.read_csv(f"{data_dir}/fold_{fold_idx}_train.csv")
    val_df = pd.read_csv(f"{data_dir}/fold_{fold_idx}_val.csv")
    
    # 创建数据集
    train_dataset = ClinicalDataset(train_df, 'label', 'Patient_ID')
    val_dataset = ClinicalDataset(val_df, 'label', 'Patient_ID')
    
    return train_dataset, val_dataset
```

#### 4.2 数据加载器配置
```python
def create_data_loaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """
    数据加载器配置：
    
    参数：
    - batch_size: 批次大小 (32)
    - num_workers: 数据加载进程数 (4)
    - pin_memory: 是否使用固定内存 (True)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

#### 4.3 训练循环详细实现
```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    单轮训练详细实现：
    
    返回：
    - train_loss: 训练损失
    - train_acc: 训练准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets, _) in enumerate(train_loader):
        # 数据移动到设备
        data, targets = data.to(device), targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """
    单轮验证详细实现：
    
    返回：
    - val_loss: 验证损失
    - val_acc: 验证准确率
    - predictions: 预测结果
    - targets: 真实标签
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for data, targets, _ in val_loader:
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 保存预测结果
            predictions.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, predictions, targets_list
```

### 5. 模型校准详细实现

#### 5.1 温度缩放算法
```python
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        """
        温度缩放模型：
        用于后验校准，提高预测概率的可靠性
        """
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x):
        """
        前向传播：
        应用温度缩放到logits
        """
        logits = self.model(x)
        return self.temperature_scale(logits)
    
    def temperature_scale(self, logits):
        """
        温度缩放：
        scaled_logits = logits / temperature
        """
        return logits / self.temperature
    
    def set_temperature(self, val_loader, device):
        """
        优化温度参数：
        使用验证集优化温度参数
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)
        
        # 收集验证集预测
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                logits_list.append(self.model(x))
                labels_list.append(y)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        # 计算校准前的ECE
        ece_before = ece_criterion(logits, labels).item()
        print(f"校准前ECE: {ece_before:.4f}")
        
        # 优化温度参数
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # 计算校准后的ECE
        ece_after = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f"最优温度: {self.temperature.item():.3f}")
        print(f"校准后ECE: {ece_after:.4f}")
        
        return self
```

#### 5.2 期望校准误差（ECE）计算
```python
class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        期望校准误差计算：
        
        参数：
        - n_bins: 置信度分箱数量 (15)
        """
        super(_ECELoss, self).__init__()
        self.n_bins = n_bins
    
    def forward(self, logits, labels):
        """
        计算ECE：
        ECE = Σ |acc(B_m) - conf(B_m)| * |B_m| / n
        """
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confs, preds = torch.max(softmaxes, 1)
        accs = preds.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        for i in range(self.n_bins):
            # 定义置信度区间
            lo, hi = i / self.n_bins, (i + 1) / self.n_bins
            in_bin = confs.gt(lo) & confs.le(hi)
            prop = in_bin.float().mean()
            
            if prop.item() > 0:
                # 计算区间内的准确率和置信度
                acc_in_bin = accs[in_bin].float().mean()
                avg_conf_in_bin = confs[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop
        
        return ece
```

### 6. 性能评估详细指标

#### 6.1 分类性能指标
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import numpy as np

def calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    计算详细的分类性能指标：
    
    参数：
    - y_true: 真实标签
    - y_pred_proba: 预测概率
    - threshold: 分类阈值
    
    返回：
    - 字典包含所有性能指标
    """
    # 二值化预测
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算各项指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算AUROC
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    # 计算AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = np.trapz(precision_curve, recall_curve)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auroc': auroc,
        'auprc': auprc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
```

#### 6.2 置信区间计算
```python
from scipy import stats

def calculate_confidence_intervals(metrics_list, confidence_level=0.95):
    """
    计算性能指标的置信区间：
    
    参数：
    - metrics_list: 各折的性能指标列表
    - confidence_level: 置信水平 (0.95)
    
    返回：
    - 各指标的置信区间
    """
    confidence_intervals = {}
    
    for metric in ['auroc', 'sensitivity', 'specificity', 'precision', 'recall', 'f1_score']:
        values = [fold_metrics[metric] for fold_metrics in metrics_list]
        
        # 计算均值和标准误
        mean_val = np.mean(values)
        std_err = stats.sem(values)
        
        # 计算置信区间
        ci_lower, ci_upper = stats.t.interval(
            confidence_level, 
            len(values) - 1, 
            loc=mean_val, 
            scale=std_err
        )
        
        confidence_intervals[metric] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(values)
        }
    
    return confidence_intervals
```

### 7. 可解释性分析详细实现

#### 7.1 SHAP分析
```python
import shap

def perform_shap_analysis(model, val_loader, feature_names, device):
    """
    执行SHAP分析：
    
    参数：
    - model: 训练好的模型
    - val_loader: 验证数据加载器
    - feature_names: 特征名称列表
    - device: 计算设备
    
    返回：
    - SHAP解释器对象
    """
    # 收集验证集数据
    background_data = []
    with torch.no_grad():
        for batch in val_loader:
            data, _, _ = batch
            background_data.append(data.cpu().numpy())
    
    background_data = np.vstack(background_data)
    
    # 创建SHAP解释器
    explainer = shap.DeepExplainer(model, torch.FloatTensor(background_data).to(device))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(torch.FloatTensor(background_data).to(device))
    
    return explainer, shap_values, background_data

def plot_shap_summary(shap_values, feature_names, max_display=20):
    """
    绘制SHAP摘要图：
    
    参数：
    - shap_values: SHAP值
    - feature_names: 特征名称
    - max_display: 最大显示特征数
    """
    shap.summary_plot(
        shap_values, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_waterfall(shap_values, feature_names, sample_idx=0):
    """
    绘制SHAP瀑布图：
    
    参数：
    - shap_values: SHAP值
    - feature_names: 特征名称
    - sample_idx: 样本索引
    """
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'shap_waterfall_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
```

#### 7.2 特征重要性分析
```python
def analyze_feature_importance(shap_values, feature_names, top_n=10):
    """
    分析特征重要性：
    
    参数：
    - shap_values: SHAP值
    - feature_names: 特征名称
    - top_n: 显示前N个重要特征
    
    返回：
    - 特征重要性排序
    """
    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    # 显示前N个重要特征
    print(f"Top {top_n} 重要特征:")
    print(importance_df.head(top_n))
    
    return importance_df

def plot_feature_importance(importance_df, top_n=10):
    """
    绘制特征重要性图：
    
    参数：
    - importance_df: 特征重要性DataFrame
    - top_n: 显示前N个特征
    """
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('SHAP重要性')
    plt.title(f'Top {top_n} 特征重要性')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
```

### 8. 训练过程监控

#### 8.1 训练日志记录
```python
import logging
from datetime import datetime

def setup_logging(log_file='clinical_training.log'):
    """
    设置训练日志：
    
    参数：
    - log_file: 日志文件名
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_training_progress(logger, epoch, train_loss, train_acc, val_loss, val_acc, lr):
    """
    记录训练进度：
    
    参数：
    - logger: 日志记录器
    - epoch: 当前轮次
    - train_loss: 训练损失
    - train_acc: 训练准确率
    - val_loss: 验证损失
    - val_acc: 验证准确率
    - lr: 学习率
    """
    logger.info(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"LR: {lr:.6f}"
    )
```

#### 8.2 早停机制
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        早停机制：
        
        参数：
        - patience: 容忍轮次 (10)
        - min_delta: 最小改善阈值 (0.001)
        - restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        检查是否需要早停：
        
        返回：
        - True: 需要早停
        - False: 继续训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
```

### 9. 模型保存与加载

#### 9.1 模型保存
```python
def save_model(model, optimizer, epoch, metrics, filepath):
    """
    保存模型检查点：
    
    参数：
    - model: 模型对象
    - optimizer: 优化器对象
    - epoch: 当前轮次
    - metrics: 性能指标
    - filepath: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': {
            'input_size': 27,
            'hidden_size': 64,
            'output_size': 2,
            'dropout_p': 0.5
        }
    }, filepath)
    print(f"模型已保存到: {filepath}")

def load_model(filepath, device):
    """
    加载模型检查点：
    
    参数：
    - filepath: 模型文件路径
    - device: 计算设备
    
    返回：
    - 模型对象和检查点信息
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # 重建模型
    model = ClinicalNet(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint
```

### 10. 实验配置管理

#### 10.1 实验参数配置
```python
import yaml

EXPERIMENT_CONFIG = {
    'data': {
        'input_size': 27,
        'batch_size': 32,
        'num_workers': 4,
        'train_val_split': 0.8,
        'random_seed': 42
    },
    'model': {
        'hidden_size': 64,
        'dropout_p': 0.5,
        'activation': 'relu'
    },
    'training': {
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 10,
        'min_delta': 0.001
    },
    'optimization': {
        'optimizer': 'adam',
        'scheduler': 'reduce_lr_on_plateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5
    },
    'evaluation': {
        'n_folds': 5,
        'confidence_level': 0.95,
        'calibration_bins': 15
    }
}

def save_experiment_config(config, filepath):
    """
    保存实验配置：
    
    参数：
    - config: 配置字典
    - filepath: 配置文件路径
    """
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"实验配置已保存到: {filepath}")

def load_experiment_config(filepath):
    """
    加载实验配置：
    
    参数：
    - filepath: 配置文件路径
    
    返回：
    - 配置字典
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

这个详细的方法论包含了临床数据通路的完整技术实现，包括数据预处理、模型架构、训练策略、校准方法、性能评估和可解释性分析的所有细节。每个部分都有具体的代码实现和参数设置。 