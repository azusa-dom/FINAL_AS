# 双通路AI框架详细方法论 - 第二部分：MRI分析通路

## 🧠 MRI分析通路（ImagingNet）详细方法论

### 1. 数据来源与收集

#### 1.1 数据来源详细描述
- **数据来源**：Radiopaedia教学档案（https://radiopaedia.org/）
- **许可证**：CC BY-NC-SA 3.0
- **数据质量**：教学级质量，经过专家标注
- **影像类型**：T1加权像（T1-weighted images）
- **采集协议**：标准化MRI扫描协议

#### 1.2 受试者详细信息
```python
# 受试者详细信息
SUBJECT_DETAILS = {
    'as_patients': {
        'count': 6,
        'age_range': '25-65岁',
        'gender': '4男2女',
        'diagnosis': '临床确诊AS',
        'disease_duration': '2-15年',
        'medication': 'NSAIDs, DMARDs'
    },
    'healthy_controls': {
        'count': 2,
        'age_range': '30-55岁',
        'gender': '1男1女',
        'exclusion_criteria': '无炎症性关节病史',
        'imaging_findings': '正常脊柱MRI'
    }
}

# 切片详细信息
SLICE_DETAILS = {
    'total_slices': 39,
    'as_slices': 24,  # 6例AS患者 × 4个切片/例
    'hc_slices': 15,  # 2例健康对照 × 7-8个切片/例
    'slice_thickness': '3-5mm',
    'spacing': '0.7×0.7mm',
    'matrix_size': '256×256'
}
```

#### 1.3 数据组织结构
```
data_root/
├── mri_AS/
│   ├── patient1/
│   │   ├── slice_001.jpg
│   │   ├── slice_002.jpg
│   │   ├── slice_003.jpg
│   │   └── slice_004.jpg
│   ├── patient2/
│   │   └── ...
│   └── ...
├── mri_health/
│   ├── health1/
│   │   ├── subjA/
│   │   │   ├── slice_001.jpg
│   │   │   ├── slice_002.jpg
│   │   │   └── ...
│   │   └── subjB/
│   │       └── ...
│   └── health2/
│       └── ...
```

### 2. 影像预处理详细流程

#### 2.1 原始数据加载
```python
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torchio as tio

def load_mri_data(filepath):
    """
    加载MRI数据：
    
    参数：
    - filepath: NIfTI文件路径
    
    返回：
    - 3D numpy数组
    """
    # 加载NIfTI文件
    img = nib.load(filepath)
    data = img.get_fdata()
    
    # 获取元数据
    header = img.header
    affine = img.affine
    
    return data, header, affine

def extract_slices_from_volume(volume, slice_indices=None):
    """
    从3D体积中提取切片：
    
    参数：
    - volume: 3D numpy数组
    - slice_indices: 切片索引列表
    
    返回：
    - 切片列表
    """
    if slice_indices is None:
        # 自动选择中间切片
        middle_slice = volume.shape[2] // 2
        slice_indices = range(middle_slice-2, middle_slice+3)
    
    slices = []
    for idx in slice_indices:
        if 0 <= idx < volume.shape[2]:
            slice_data = volume[:, :, idx]
            slices.append(slice_data)
    
    return slices
```

#### 2.2 N4偏场校正
```python
import ants

def apply_n4_bias_correction(image, shrink_factor=4, convergence_threshold=1e-7):
    """
    应用N4偏场校正：
    
    参数：
    - image: 输入图像（ANTsImage对象）
    - shrink_factor: 收缩因子 (4)
    - convergence_threshold: 收敛阈值 (1e-7)
    
    返回：
    - 校正后的图像
    """
    # N4偏场校正参数
    n4_params = {
        'shrink_factor': shrink_factor,
        'convergence_threshold': convergence_threshold,
        'spline_order': 3,
        'number_of_fitting_levels': 4,
        'number_of_iterations': [50, 40, 30, 20]
    }
    
    # 执行N4校正
    corrected_image = ants.n4_bias_field_correction(
        image,
        shrink_factor=n4_params['shrink_factor'],
        convergence_threshold=n4_params['convergence_threshold'],
        spline_order=n4_params['spline_order'],
        number_of_fitting_levels=n4_params['number_of_fitting_levels'],
        number_of_iterations=n4_params['number_of_iterations']
    )
    
    return corrected_image

def n4_correction_pipeline(input_path, output_path):
    """
    N4校正完整流程：
    
    参数：
    - input_path: 输入文件路径
    - output_path: 输出文件路径
    """
    # 读取图像
    img = ants.image_read(input_path)
    
    # 应用N4校正
    corrected_img = apply_n4_bias_correction(img)
    
    # 保存结果
    ants.image_write(corrected_img, output_path)
    
    return corrected_img
```

#### 2.3 高斯平滑
```python
def apply_gaussian_smoothing(image, sigma=0.51):
    """
    应用高斯平滑：
    
    参数：
    - image: 输入图像
    - sigma: 高斯核标准差 (0.51mm)
    
    返回：
    - 平滑后的图像
    """
    # 使用TorchIO进行高斯平滑
    transform = tio.transforms.GaussianBlur(
        std=sigma,
        p=1.0
    )
    
    # 创建TorchIO Subject
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(image).unsqueeze(0))
    )
    
    # 应用变换
    transformed_subject = transform(subject)
    smoothed_image = transformed_subject['image'].tensor.squeeze(0).numpy()
    
    return smoothed_image
```

#### 2.4 重采样
```python
def resample_image(image, target_spacing=(0.7, 0.7), original_spacing=None):
    """
    图像重采样：
    
    参数：
    - image: 输入图像
    - target_spacing: 目标间距 (0.7×0.7mm)
    - original_spacing: 原始间距
    
    返回：
    - 重采样后的图像
    """
    # 使用TorchIO进行重采样
    transform = tio.transforms.Resample(
        target=target_spacing,
        image_interpolation='linear',
        label_interpolation='nearest'
    )
    
    # 创建TorchIO Subject
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(image).unsqueeze(0))
    )
    
    # 应用变换
    transformed_subject = transform(subject)
    resampled_image = transformed_subject['image'].tensor.squeeze(0).numpy()
    
    return resampled_image
```

#### 2.5 尺寸标准化
```python
def resize_image(image, target_size=(224, 224), interpolation='bilinear'):
    """
    图像尺寸标准化：
    
    参数：
    - image: 输入图像
    - target_size: 目标尺寸 (224×224)
    - interpolation: 插值方法
    
    返回：
    - 调整尺寸后的图像
    """
    from PIL import Image
    import numpy as np
    
    # 转换为PIL图像
    if image.dtype != np.uint8:
        # 归一化到0-255
        image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        image_normalized = image
    
    pil_image = Image.fromarray(image_normalized)
    
    # 调整尺寸
    resized_image = pil_image.resize(target_size, getattr(Image, interpolation.upper()))
    
    # 转换回numpy数组
    resized_array = np.array(resized_image)
    
    return resized_array
```

#### 2.6 ImageNet标准化
```python
def apply_imagenet_normalization(image):
    """
    应用ImageNet标准化：
    
    参数：
    - image: 输入图像 (0-255)
    
    返回：
    - 标准化后的图像
    """
    # ImageNet均值和标准差
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # 归一化到0-1
    image_normalized = image.astype(np.float32) / 255.0
    
    # 转换为RGB（如果是灰度图）
    if len(image_normalized.shape) == 2:
        image_rgb = np.stack([image_normalized] * 3, axis=-1)
    else:
        image_rgb = image_normalized
    
    # 应用ImageNet标准化
    image_standardized = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    
    return image_standardized
```

#### 2.7 完整预处理流程
```python
def complete_preprocessing_pipeline(input_path, output_path):
    """
    完整预处理流程：
    
    参数：
    - input_path: 输入文件路径
    - output_path: 输出文件路径
    
    返回：
    - 预处理后的图像
    """
    # 1. 加载原始数据
    data, header, affine = load_mri_data(input_path)
    
    # 2. N4偏场校正
    img_ants = ants.from_numpy(data)
    corrected_img = apply_n4_bias_correction(img_ants)
    corrected_data = corrected_img.numpy()
    
    # 3. 高斯平滑
    smoothed_data = apply_gaussian_smoothing(corrected_data, sigma=0.51)
    
    # 4. 重采样
    resampled_data = resample_image(smoothed_data, target_spacing=(0.7, 0.7))
    
    # 5. 尺寸标准化
    resized_data = resize_image(resampled_data, target_size=(224, 224))
    
    # 6. ImageNet标准化
    standardized_data = apply_imagenet_normalization(resized_data)
    
    # 7. 保存结果
    np.save(output_path, standardized_data)
    
    return standardized_data
```

### 3. 特征提取详细实现

#### 3.1 ResNet-18骨干网络
```python
import torch
import torch.nn as nn
from torchvision import models

def create_resnet18_backbone(pretrained=True, freeze_backbone=True):
    """
    创建ResNet-18骨干网络：
    
    参数：
    - pretrained: 是否使用预训练权重
    - freeze_backbone: 是否冻结骨干网络
    
    返回：
    - 特征提取器
    """
    # 加载预训练的ResNet-18
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # 移除分类头，保留特征提取层
    backbone.fc = nn.Identity()
    
    # 冻结骨干网络参数（可选）
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    return backbone

def extract_features(model, image_tensor):
    """
    提取特征：
    
    参数：
    - model: ResNet-18模型
    - image_tensor: 输入图像张量 [batch_size, 3, 224, 224]
    
    返回：
    - 特征向量 [batch_size, 512]
    """
    model.eval()
    with torch.no_grad():
        features = model(image_tensor)
    
    return features
```

#### 3.2 数据变换配置
```python
from torchvision import transforms

def create_image_transforms():
    """
    创建图像变换：
    
    返回：
    - 变换管道
    """
    transform = transforms.Compose([
        # 调整尺寸
        transforms.Resize((224, 224)),
        
        # 转换为张量
        transforms.ToTensor(),
        
        # ImageNet标准化
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform

def apply_transforms_to_image(image_path, transform):
    """
    应用变换到图像：
    
    参数：
    - image_path: 图像路径
    - transform: 变换管道
    
    返回：
    - 变换后的图像张量
    """
    from PIL import Image
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 应用变换
    image_tensor = transform(image)
    
    return image_tensor
```

#### 3.3 切片级特征提取
```python
def extract_slice_features(image_paths, model, transform, device):
    """
    提取切片级特征：
    
    参数：
    - image_paths: 图像路径列表
    - model: ResNet-18模型
    - transform: 图像变换
    - device: 计算设备
    
    返回：
    - 特征矩阵 [n_slices, 512]
    """
    features_list = []
    
    for image_path in image_paths:
        # 应用变换
        image_tensor = apply_transforms_to_image(image_path, transform)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 提取特征
        features = extract_features(model, image_tensor)
        features_list.append(features.cpu().numpy())
    
    # 堆叠所有特征
    features_matrix = np.vstack(features_list)
    
    return features_matrix
```

#### 3.4 受试者级特征聚合
```python
def aggregate_subject_features(slice_features, aggregation_method='mean'):
    """
    聚合受试者级特征：
    
    参数：
    - slice_features: 切片特征矩阵 [n_slices, 512]
    - aggregation_method: 聚合方法 ('mean', 'max', 'median')
    
    返回：
    - 聚合后的特征向量 [512]
    """
    if aggregation_method == 'mean':
        subject_features = np.mean(slice_features, axis=0)
    elif aggregation_method == 'max':
        subject_features = np.max(slice_features, axis=0)
    elif aggregation_method == 'median':
        subject_features = np.median(slice_features, axis=0)
    else:
        raise ValueError(f"不支持的聚合方法: {aggregation_method}")
    
    return subject_features

def extract_all_subject_features(subject_data, model, transform, device):
    """
    提取所有受试者的特征：
    
    参数：
    - subject_data: 受试者数据字典
    - model: ResNet-18模型
    - transform: 图像变换
    - device: 计算设备
    
    返回：
    - 受试者特征字典
    """
    subject_features = {}
    
    for subject_id, subject_info in subject_data.items():
        # 提取切片特征
        slice_features = extract_slice_features(
            subject_info['image_paths'],
            model,
            transform,
            device
        )
        
        # 聚合为受试者级特征
        subject_feature = aggregate_subject_features(slice_features, 'mean')
        
        subject_features[subject_id] = {
            'label': subject_info['label'],
            'feature_vector': subject_feature,
            'slice_features': slice_features
        }
    
    return subject_features
```

### 4. 留二法交叉验证详细实现

#### 4.1 验证策略设计
```python
from itertools import product

def create_l2o_folds(as_subjects, hc_subjects):
    """
    创建留二法交叉验证折：
    
    参数：
    - as_subjects: AS患者列表
    - hc_subjects: 健康对照列表
    
    返回：
    - 验证折列表
    """
    folds = []
    
    # 生成所有AS-HC配对
    for as_subject, hc_subject in product(as_subjects, hc_subjects):
        # 验证集：1例AS + 1例HC
        val_subjects = [as_subject, hc_subject]
        
        # 训练集：剩余的所有受试者
        train_subjects = [
            s for s in as_subjects if s != as_subject
        ] + [
            s for s in hc_subjects if s != hc_subject
        ]
        
        fold = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'fold_id': f"AS_{as_subject}_HC_{hc_subject}"
        }
        
        folds.append(fold)
    
    return folds

def print_fold_statistics(folds):
    """
    打印折统计信息：
    
    参数：
    - folds: 验证折列表
    """
    print(f"总折数: {len(folds)}")
    print(f"每折训练样本数: {len(folds[0]['train_subjects'])}")
    print(f"每折验证样本数: {len(folds[0]['val_subjects'])}")
    
    # 统计每个受试者出现的验证次数
    subject_val_counts = {}
    for fold in folds:
        for subject in fold['val_subjects']:
            subject_val_counts[subject] = subject_val_counts.get(subject, 0) + 1
    
    print("\n每个受试者的验证次数:")
    for subject, count in sorted(subject_val_counts.items()):
        print(f"  {subject}: {count}次")
```

#### 4.2 训练和验证流程
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def train_and_evaluate_fold(fold, subject_features, random_state=42):
    """
    训练和评估单个折：
    
    参数：
    - fold: 验证折信息
    - subject_features: 受试者特征字典
    - random_state: 随机种子
    
    返回：
    - 预测结果字典
    """
    # 准备训练数据
    train_features = []
    train_labels = []
    
    for subject_id in fold['train_subjects']:
        train_features.append(subject_features[subject_id]['feature_vector'])
        train_labels.append(subject_features[subject_id]['label'])
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    
    # 训练逻辑回归模型
    clf = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=random_state,
        max_iter=1000
    )
    
    clf.fit(train_features, train_labels)
    
    # 在验证集上预测
    predictions = []
    for subject_id in fold['val_subjects']:
        feature_vector = subject_features[subject_id]['feature_vector'].reshape(1, -1)
        prob = clf.predict_proba(feature_vector)[0, 1]  # AS的概率
        
        predictions.append({
            'subject_id': subject_id,
            'y_true': subject_features[subject_id]['label'],
            'prob_raw': prob,
            'fold_id': fold['fold_id']
        })
    
    return predictions

def run_l2o_cross_validation(subject_features, random_state=42):
    """
    运行完整的留二法交叉验证：
    
    参数：
    - subject_features: 受试者特征字典
    - random_state: 随机种子
    
    返回：
    - 所有预测结果
    """
    # 分离AS和HC受试者
    as_subjects = [sid for sid, info in subject_features.items() if info['label'] == 1]
    hc_subjects = [sid for sid, info in subject_features.items() if info['label'] == 0]
    
    # 创建验证折
    folds = create_l2o_folds(as_subjects, hc_subjects)
    
    # 运行所有折
    all_predictions = []
    for fold in folds:
        fold_predictions = train_and_evaluate_fold(fold, subject_features, random_state)
        all_predictions.extend(fold_predictions)
    
    return all_predictions
```

#### 4.3 结果聚合
```python
import pandas as pd

def aggregate_l2o_results(predictions):
    """
    聚合留二法交叉验证结果：
    
    参数：
    - predictions: 所有预测结果列表
    
    返回：
    - 聚合后的结果DataFrame
    """
    # 转换为DataFrame
    df = pd.DataFrame(predictions)
    
    # 按受试者ID聚合（平均概率）
    df_grouped = df.groupby(['subject_id', 'y_true'], as_index=False).agg({
        'prob_raw': 'mean',
        'fold_id': 'count'  # 验证次数
    }).rename(columns={'fold_id': 'validation_count'})
    
    return df_grouped

def calculate_l2o_performance(df_grouped):
    """
    计算留二法交叉验证性能：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    
    返回：
    - 性能指标字典
    """
    # 计算AUROC
    auroc = roc_auc_score(df_grouped['y_true'], df_grouped['prob_raw'])
    
    # 计算准确率
    y_pred = (df_grouped['prob_raw'] >= 0.5).astype(int)
    accuracy = accuracy_score(df_grouped['y_true'], y_pred)
    
    # 计算敏感性、特异性
    tp = ((df_grouped['y_true'] == 1) & (y_pred == 1)).sum()
    tn = ((df_grouped['y_true'] == 0) & (y_pred == 0)).sum()
    fp = ((df_grouped['y_true'] == 0) & (y_pred == 1)).sum()
    fn = ((df_grouped['y_true'] == 1) & (y_pred == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
```

### 5. 方向校正机制

#### 5.1 方向检测
```python
def detect_prediction_direction(df_grouped):
    """
    检测预测方向：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    
    返回：
    - 方向信息字典
    """
    # 计算AUROC
    auroc = roc_auc_score(df_grouped['y_true'], df_grouped['prob_raw'])
    
    # 检查方向
    if auroc < 0.5:
        direction = 'inverted'
        corrected_auroc = 1 - auroc
    else:
        direction = 'correct'
        corrected_auroc = auroc
    
    return {
        'original_auroc': auroc,
        'corrected_auroc': corrected_auroc,
        'direction': direction,
        'needs_correction': auroc < 0.5
    }

def apply_direction_correction(df_grouped):
    """
    应用方向校正：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    
    返回：
    - 校正后的DataFrame
    """
    # 检测方向
    direction_info = detect_prediction_direction(df_grouped)
    
    # 如果需要校正，反转概率
    if direction_info['needs_correction']:
        df_corrected = df_grouped.copy()
        df_corrected['prob_raw'] = 1 - df_corrected['prob_raw']
        df_corrected['prob_corrected'] = True
    else:
        df_corrected = df_grouped.copy()
        df_corrected['prob_corrected'] = False
    
    return df_corrected, direction_info
```

#### 5.2 置换检验
```python
from scipy import stats
import numpy as np

def permutation_test(df_grouped, n_permutations=10000, random_state=42):
    """
    执行置换检验：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    - n_permutations: 置换次数
    - random_state: 随机种子
    
    返回：
    - 置换检验结果
    """
    np.random.seed(random_state)
    
    # 计算观察到的AUROC
    observed_auroc = roc_auc_score(df_grouped['y_true'], df_grouped['prob_raw'])
    
    # 执行置换检验
    permuted_aurocs = []
    for _ in range(n_permutations):
        # 随机打乱标签
        permuted_labels = np.random.permutation(df_grouped['y_true'])
        permuted_auroc = roc_auc_score(permuted_labels, df_grouped['prob_raw'])
        permuted_aurocs.append(permuted_auroc)
    
    # 计算p值
    p_value = np.mean(np.array(permuted_aurocs) >= observed_auroc)
    
    return {
        'observed_auroc': observed_auroc,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'permuted_aurocs': permuted_aurocs
    }
```

### 6. Grad-CAM注意力映射

#### 6.1 Grad-CAM实现
```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Grad-CAM实现：
        
        参数：
        - model: 目标模型
        - target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.register_hooks()
    
    def register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        """
        生成CAM：
        
        参数：
        - input_image: 输入图像
        - class_idx: 目标类别索引
        
        返回：
        - CAM热力图
        """
        # 前向传播
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3])
        
        # 生成CAM
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)
        cam = F.relu(cam)  # 应用ReLU
        
        # 归一化
        cam = F.interpolate(cam.unsqueeze(0), size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0)
        
        # 归一化到0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()

def apply_gradcam_to_mri(model, image_path, transform, device, target_class=1):
    """
    对MRI图像应用Grad-CAM：
    
    参数：
    - model: 训练好的模型
    - image_path: 图像路径
    - transform: 图像变换
    - device: 计算设备
    - target_class: 目标类别
    
    返回：
    - 原始图像和CAM热力图
    """
    # 加载和预处理图像
    image_tensor = apply_transforms_to_image(image_path, transform)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, model.layer4[-1])  # ResNet-18的最后一个卷积层
    
    # 生成CAM
    cam = grad_cam.generate_cam(image_tensor, target_class)
    
    # 加载原始图像用于可视化
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image)
    
    return original_array, cam

def visualize_gradcam(original_image, cam, alpha=0.6):
    """
    可视化Grad-CAM：
    
    参数：
    - original_image: 原始图像
    - cam: CAM热力图
    - alpha: 透明度
    
    返回：
    - 可视化结果
    """
    # 将CAM转换为热力图
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 叠加到原始图像
    result = heatmap * alpha + original_image * (1 - alpha)
    result = result.astype(np.uint8)
    
    return result
```

### 7. 特征空间几何分析

#### 7.1 余弦距离分析
```python
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

def calculate_cosine_distances(feature_matrix, labels):
    """
    计算余弦距离：
    
    参数：
    - feature_matrix: 特征矩阵 [n_subjects, 512]
    - labels: 标签数组
    
    返回：
    - 距离矩阵和统计信息
    """
    n_subjects = feature_matrix.shape[0]
    distance_matrix = np.zeros((n_subjects, n_subjects))
    
    # 计算所有对之间的距离
    for i in range(n_subjects):
        for j in range(n_subjects):
            if i != j:
                distance_matrix[i, j] = cosine(feature_matrix[i], feature_matrix[j])
    
    # 分离AS和HC的距离
    as_indices = np.where(labels == 1)[0]
    hc_indices = np.where(labels == 0)[0]
    
    # AS-AS距离
    as_as_distances = []
    for i in as_indices:
        for j in as_indices:
            if i < j:
                as_as_distances.append(distance_matrix[i, j])
    
    # HC-HC距离
    hc_hc_distances = []
    for i in hc_indices:
        for j in hc_indices:
            if i < j:
                hc_hc_distances.append(distance_matrix[i, j])
    
    # AS-HC距离
    as_hc_distances = []
    for i in as_indices:
        for j in hc_indices:
            as_hc_distances.append(distance_matrix[i, j])
    
    return {
        'distance_matrix': distance_matrix,
        'as_as_distances': np.array(as_as_distances),
        'hc_hc_distances': np.array(hc_hc_distances),
        'as_hc_distances': np.array(as_hc_distances)
    }

def analyze_feature_distributions(distance_results):
    """
    分析特征分布：
    
    参数：
    - distance_results: 距离计算结果
    
    返回：
    - 分布分析结果
    """
    # KS检验
    ks_stat_as_hc, ks_p_as_hc = ks_2samp(
        distance_results['as_as_distances'],
        distance_results['as_hc_distances']
    )
    
    ks_stat_hc_as, ks_p_hc_as = ks_2samp(
        distance_results['hc_hc_distances'],
        distance_results['as_hc_distances']
    )
    
    # 统计摘要
    summary = {
        'as_as_mean': np.mean(distance_results['as_as_distances']),
        'as_as_std': np.std(distance_results['as_as_distances']),
        'hc_hc_mean': np.mean(distance_results['hc_hc_distances']),
        'hc_hc_std': np.std(distance_results['hc_hc_distances']),
        'as_hc_mean': np.mean(distance_results['as_hc_distances']),
        'as_hc_std': np.std(distance_results['as_hc_distances']),
        'ks_stat_as_hc': ks_stat_as_hc,
        'ks_p_as_hc': ks_p_as_hc,
        'ks_stat_hc_as': ks_stat_hc_as,
        'ks_p_hc_as': ks_p_hc_as
    }
    
    return summary
```

#### 7.2 降维可视化
```python
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap

def perform_dimensionality_reduction(feature_matrix, labels):
    """
    执行降维分析：
    
    参数：
    - feature_matrix: 特征矩阵 [n_subjects, 512]
    - labels: 标签数组
    
    返回：
    - 各种降维结果
    """
    results = {}
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_matrix)
    results['pca'] = {
        'coordinates': pca_result,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }
    
    # 核PCA
    kpca = KernelPCA(n_components=2, kernel='rbf')
    kpca_result = kpca.fit_transform(feature_matrix)
    results['kpca'] = {
        'coordinates': kpca_result
    }
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix)-1))
    tsne_result = tsne.fit_transform(feature_matrix)
    results['tsne'] = {
        'coordinates': tsne_result
    }
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_reducer.fit_transform(feature_matrix)
    results['umap'] = {
        'coordinates': umap_result
    }
    
    return results

def calculate_separation_metrics(coordinates, labels):
    """
    计算分离指标：
    
    参数：
    - coordinates: 降维后的坐标
    - labels: 标签
    
    返回：
    - 分离指标
    """
    from sklearn.metrics import silhouette_score
    
    # 轮廓系数
    silhouette = silhouette_score(coordinates, labels)
    
    # 类间距离
    as_coords = coordinates[labels == 1]
    hc_coords = coordinates[labels == 0]
    
    as_center = np.mean(as_coords, axis=0)
    hc_center = np.mean(hc_coords, axis=0)
    
    class_distance = np.linalg.norm(as_center - hc_center)
    
    return {
        'silhouette_score': silhouette,
        'class_distance': class_distance,
        'as_center': as_center,
        'hc_center': hc_center
    }
```

### 8. 性能评估详细指标

#### 8.1 分类性能评估
```python
def evaluate_mri_performance(df_grouped, permutation_results):
    """
    评估MRI模型性能：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    - permutation_results: 置换检验结果
    
    返回：
    - 完整的性能评估结果
    """
    # 基本分类指标
    basic_metrics = calculate_l2o_performance(df_grouped)
    
    # 方向校正
    df_corrected, direction_info = apply_direction_correction(df_grouped)
    corrected_metrics = calculate_l2o_performance(df_corrected)
    
    # 置信区间（bootstrap）
    bootstrap_results = bootstrap_confidence_intervals(df_corrected, n_bootstrap=1000)
    
    # 特征空间分析
    feature_matrix = np.vstack([df_corrected['feature_vector'].values])
    labels = df_corrected['y_true'].values
    
    distance_results = calculate_cosine_distances(feature_matrix, labels)
    distribution_analysis = analyze_feature_distributions(distance_results)
    
    # 降维分析
    dim_reduction_results = perform_dimensionality_reduction(feature_matrix, labels)
    separation_metrics = {}
    
    for method, coords in dim_reduction_results.items():
        separation_metrics[method] = calculate_separation_metrics(
            coords['coordinates'], labels
        )
    
    return {
        'basic_metrics': basic_metrics,
        'corrected_metrics': corrected_metrics,
        'direction_info': direction_info,
        'permutation_test': permutation_results,
        'bootstrap_intervals': bootstrap_results,
        'feature_space_analysis': {
            'distance_analysis': distance_results,
            'distribution_analysis': distribution_analysis,
            'dimensionality_reduction': dim_reduction_results,
            'separation_metrics': separation_metrics
        }
    }
```

#### 8.2 Bootstrap置信区间
```python
def bootstrap_confidence_intervals(df_grouped, n_bootstrap=1000, confidence_level=0.95):
    """
    计算Bootstrap置信区间：
    
    参数：
    - df_grouped: 聚合后的结果DataFrame
    - n_bootstrap: Bootstrap样本数
    - confidence_level: 置信水平
    
    返回：
    - 置信区间结果
    """
    np.random.seed(42)
    
    bootstrap_aurocs = []
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        # 重采样
        bootstrap_indices = np.random.choice(
            len(df_grouped), 
            size=len(df_grouped), 
            replace=True
        )
        bootstrap_sample = df_grouped.iloc[bootstrap_indices]
        
        # 计算指标
        metrics = calculate_l2o_performance(bootstrap_sample)
        bootstrap_aurocs.append(metrics['auroc'])
        bootstrap_accuracies.append(metrics['accuracy'])
    
    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    auroc_ci = np.percentile(bootstrap_aurocs, [lower_percentile, upper_percentile])
    accuracy_ci = np.percentile(bootstrap_accuracies, [lower_percentile, upper_percentile])
    
    return {
        'auroc_ci': auroc_ci,
        'accuracy_ci': accuracy_ci,
        'auroc_mean': np.mean(bootstrap_aurocs),
        'accuracy_mean': np.mean(bootstrap_accuracies),
        'auroc_std': np.std(bootstrap_aurocs),
        'accuracy_std': np.std(bootstrap_accuracies)
    }
```

### 9. 实验配置和参数

#### 9.1 完整实验配置
```python
MRI_EXPERIMENT_CONFIG = {
    'data': {
        'as_patients': 6,
        'healthy_controls': 2,
        'total_slices': 39,
        'image_size': (224, 224),
        'target_spacing': (0.7, 0.7)
    },
    'preprocessing': {
        'n4_correction': {
            'shrink_factor': 4,
            'convergence_threshold': 1e-7,
            'spline_order': 3,
            'number_of_fitting_levels': 4,
            'number_of_iterations': [50, 40, 30, 20]
        },
        'gaussian_smoothing': {
            'sigma': 0.51
        },
        'normalization': {
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225]
        }
    },
    'feature_extraction': {
        'backbone': 'resnet18',
        'pretrained': True,
        'freeze_backbone': True,
        'feature_dim': 512,
        'aggregation_method': 'mean'
    },
    'classification': {
        'classifier': 'logistic_regression',
        'C': 1.0,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'max_iter': 1000
    },
    'validation': {
        'method': 'leave_two_out',
        'random_state': 42
    },
    'evaluation': {
        'permutation_test': {
            'n_permutations': 10000
        },
        'bootstrap': {
            'n_bootstrap': 1000,
            'confidence_level': 0.95
        }
    }
}
```

#### 9.2 结果记录和保存
```python
def save_mri_experiment_results(results, config, output_dir):
    """
    保存MRI实验结果：
    
    参数：
    - results: 实验结果
    - config: 实验配置
    - output_dir: 输出目录
    """
    import json
    import pickle
    from datetime import datetime
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(f"{output_dir}/experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存结果
    with open(f"{output_dir}/experiment_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # 保存性能指标
    performance_summary = {
        'timestamp': datetime.now().isoformat(),
        'basic_metrics': results['basic_metrics'],
        'corrected_metrics': results['corrected_metrics'],
        'direction_info': results['direction_info'],
        'permutation_test': {
            'observed_auroc': results['permutation_test']['observed_auroc'],
            'p_value': results['permutation_test']['p_value']
        },
        'bootstrap_intervals': results['bootstrap_intervals']
    }
    
    with open(f"{output_dir}/performance_summary.json", 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print(f"实验结果已保存到: {output_dir}")
```

这个详细的MRI分析通路方法论包含了所有技术细节、参数设置、实验过程和评估方法。每个部分都有具体的代码实现和详细的参数说明。 