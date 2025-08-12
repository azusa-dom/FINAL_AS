# DDI-AS 项目脚本文档

## 项目概述

DDI-AS (Dual Diagnostic Intelligence for Ankylosing Spondylitis) 是一个用于强直性脊柱炎诊断的双模态AI框架，结合临床数据和MRI影像数据进行诊断。

**GitHub仓库**: https://github.com/azusa-dom/FINAL_AS

## 技术栈

- **Python**: 主要编程语言
- **PyTorch**: 深度学习框架 (ResNet-18)
- **scikit-learn**: 机器学习库 (Gradient Boosting, Random Forest, Logistic Regression)
- **Docker**: 容器化部署
- **CUDA**: GPU加速 (≥6 GB)
- **Git**: 版本控制 (v1.2.0)

## 脚本分类

### 1. 临床数据处理脚本

| 脚本名称 | 功能描述 | 输入数据 | 输出数据 | 核心功能 |
|---------|---------|---------|---------|---------|
| `build_balanced_dataset.py` | 平衡数据集构建 | 原始临床数据 | 平衡数据集 (4,254样本) | 1:1 AS/对照组比例 |
| `preprocess_clinical_final.py` | 临床数据预处理 | 原始特征 | 预处理特征 | 缺失值处理、标准化 |
| `preprocess_...with_smote...` | 特征工程+SMOTE | 预处理数据 | 工程特征+平衡数据 | log1p变换、SMOTE |
| `check_clinical_quality.py` | 数据质量验证 | 预处理前后数据 | 质量报告 | 数据完整性验证 |

### 2. MRI数据处理脚本

| 脚本名称 | 功能描述 | 输入数据 | 输出数据 | 核心功能 |
|---------|---------|---------|---------|---------|
| `bias_correction.py` | 偏场校正 | 原始MRI图像 | 校正图像 | N4偏场校正算法 |
| `mri_extract_roi.py` | ROI提取 | 校正图像 | ROI区域 | 骶髂关节区域提取 |
| `preprocess.py` | MRI预处理流水线 | 原始MRI | 预处理MRI | 标准化、尺寸调整 |
| `prepare_mri_folds.py` | L2O-CV折叠准备 | 8个受试者 | 12个L2O折叠 | 留二交叉验证 |
| `extract_mri_features.py` | ResNet-18特征提取 | 预处理MRI | 特征向量 | 深度学习特征 |

### 3. 模型训练脚本

| 脚本名称 | 功能描述 | 训练方法 | 模型类型 | 验证方法 |
|---------|---------|---------|---------|---------|
| `train_clinical_ensemble.py` | ClinicalNet训练 | 梯度提升 | RF + GB + LR | 5折交叉验证 |
| `train_imaging_net.py` | ImagingNet训练 | ResNet-18 + LR | ResNet-18 + LR分类器 | L2O-CV |
| `train_ensemble.py` | 集成融合 | 后期融合平均 | ClinicalNet + ImagingNet | 独立验证 |

### 4. 评估脚本

#### 临床模型评估

| 脚本名称 | 功能描述 | 评估指标 | 输出结果 | 核心功能 |
|---------|---------|---------|---------|---------|
| `evaluate_AUROC_AUPRC_CI.py` | AUROC/AUPRC/CI评估 | AUROC, AUPRC, CI | 性能统计 | 判别能力 |
| `evaluate_confusion_matrix.py` | 混淆矩阵分析 | 准确率、精确率、召回率 | 混淆矩阵 | 分类性能 |
| `shap_plot_interactions.py` | SHAP特征重要性 | SHAP值 | 重要性图 | 模型可解释性 |
| `plot_overall_metrics.py` | 整体指标可视化 | 综合指标 | 性能图表 | 多指标比较 |
| `eval_clinical_all_folds.py` | 全折叠评估 | 交叉折叠性能 | 折叠结果 | 稳定性分析 |
| `run_baseline_models.py` | 基线模型比较 | 基线性能 | 基线结果 | 性能基准 |
| `calculate_3_models_final...` | 最终统计摘要 | 综合统计 | 最终报告 | 完整性能摘要 |

#### MRI模型评估

| 脚本名称 | 功能描述 | 评估方法 | 输出结果 | 核心功能 |
|---------|---------|---------|---------|---------|
| `mri_subject_level_auc.py` | 受试者级AUC | 受试者级AUC | 个体性能 | 小样本性能 |
| `mri_test_permutation.py` | 排列检验 | 统计显著性 | p值、分布 | 随机性检验 |
| `mri_eval_auc_bootstrap.py` | Bootstrap分析 | 置信区间 | 95% CI | 不确定性量化 |
| `make_l2o_predictions.py` | L2O预测 | 留二预测 | 预测概率 | CV预测 |
| `make_l2o_predictions_imp...` | 改进L2O预测 | 优化L2O | 改进预测 | 预测增强 |
| `make_l2o_..._small_sample.py` | 小样本L2O | 小样本优化 | 小样本预测 | 样本量优化 |
| `mri_direction_correction.py` | 方向校正 | 预测方向 | 校正预测 | 符号校正 |

## 工作流程

### 脚本使用工作流

| 阶段 | 主要脚本 | 功能 | 输出 |
|------|---------|------|------|
| 数据准备 | `build_balanced_dataset.py` | 构建平衡数据集 | 4,254样本 |
| 特征工程 | `preprocess_clinical...pipeline.py` | 特征工程+平衡 | 20个工程特征 |
| 模型训练 | `train_clinical_ensemble.py` | 集成模型训练 | 训练好的模型 |
| 性能评估 | `calculate_3_models_final_stats.py` | 综合性能评估 | 性能报告 |
| 图表生成 | `regenerate_all_figures...data.py` | 生成所有图表 | 发表级图表 |
| 数据验证 | `validate_data.py` | 验证数据一致性 | 验证报告 |

## 性能指标

### 模型性能总结

| 模型类型 | AUROC | 标准差 | 样本量 | 验证方法 |
|---------|-------|--------|--------|----------|
| ClinicalNet (梯度提升) | 0.938 | ±0.003 | 4,254 | 5折CV |
| ImagingNet (ResNet-18 + LR) | 0.833 | ±0.021 | 8 | L2O-CV |
| 集成 (后期融合) | 0.941 | ±0.009 | 4,254+8 | 融合验证 |
| 随机森林 | 0.929 | ±0.006 | 4,254 | 5折CV |
| 逻辑回归 | 0.858 | ±0.008 | 4,254 | 5折CV |

## 统计分析结果

### DeLong检验结果

| 模型比较 | ΔAUROC | 95% CI | p值 | 显著性 |
|---------|--------|--------|-----|--------|
| 梯度提升 vs 随机森林 | 0.009 | (0.006, 0.012) | <0.001 | *** |
| 梯度提升 vs 逻辑回归 | 0.080 | (0.075, 0.085) | <0.001 | *** |
| 随机森林 vs 逻辑回归 | 0.071 | (0.066, 0.076) | <0.001 | *** |
| 集成 vs ClinicalNet | 0.003 | (0.001, 0.005) | 0.002 | ** |
| ClinicalNet vs ImagingNet | 0.105 | (0.095, 0.115) | <0.001 | *** |

### Bootstrap置信区间

| 指标 | 点估计 | 95% CI下限 | 95% CI上限 | 标准误 |
|------|--------|------------|------------|--------|
| ClinicalNet AUROC | 0.938 | 0.935 | 0.941 | 0.0015 |
| ImagingNet AUROC | 0.833 | 0.712 | 0.948 | 0.0602 |
| 集成 AUROC | 0.941 | 0.925 | 0.959 | 0.0087 |
| ClinicalNet ECE | 0.155 | 0.142 | 0.168 | 0.0067 |
| 集成 ECE | 0.168 | 0.154 | 0.188 | 0.0087 |

### 排列检验结果

| 检验类型 | 迭代次数 | 观察统计量 | 随机分布均值 | p值 |
|---------|---------|-----------|-------------|-----|
| AUROC排列检验 | 1,000 | 0.833 | 0.501 | 0.017 |
| 特征重要性排列检验 | 1,000 | 0.231 | 0.050 | <0.001 |

## 部署信息

### 环境要求

- **操作系统**: Linux/x86-64
- **GPU**: ≥6 GB CUDA内存
- **Docker**: 24.0版本
- **CUDA**: 11.8版本

### 重现指令

```bash
# 克隆仓库
git clone https://github.com/azusa-dom/FINAL_AS

# 重现所有结果
make reproduce-all
```

### 版本控制

- **Git标签**: v1.2.0
- **随机种子**: 42 (所有随机操作)
- **数据泄露**: 无

## 核心算法

### 集成融合公式

```
P_Ensemble = 0.5 × P_ClinicalNet + 0.5 × P_ImagingNet
```

### 校准误差计算

```
ECE = Σ(|B_m|/n) × |acc(B_m) - conf(B_m)|
```

## 数据来源

### 临床数据
- **来源**: 'Diagnosis of Rheumatic and Autoimmune Diseases' 数据集
- **规模**: 12,085个门诊记录 (2015-2022)
- **AS患病率**: 17.6% (2,127例AS病例)
- **平衡后**: 4,254条记录 (2,127 AS + 2,127对照)

### MRI数据
- **来源**: Radiopaedia.org
- **规模**: 8个受试者 (6个AS + 2个健康对照)
- **切片数**: 39个诊断相关切片
- **序列**: T1加权、STIR

## 特征工程

### 临床特征处理
- **原始特征**: 14个
- **编码后**: 20个特征 (one-hot编码)
- **数值特征**: ESR, CRP (log1p变换 + z-score标准化)
- **分类特征**: HLA-B27, RF, Anti-CCP, ANA

### MRI特征提取
- **模型**: ResNet-18 (预训练)
- **特征维度**: 512维
- **预处理**: N4偏场校正、高斯平滑、重采样、裁剪、标准化

## 验证策略

### 临床数据验证
- **方法**: 分层5折交叉验证
- **训练样本**: 3,403
- **验证样本**: 851
- **随机种子**: 42

### MRI数据验证
- **方法**: 留二交叉验证 (L2O-CV)
- **折叠数**: 12
- **统计检验**: 排列检验 (1,000次迭代)

## 可解释性

### SHAP分析
- **主要特征**: HLA-B27阳性 (重要性: 0.231-0.245)
- **可视化**: 特征重要性图、交互效应图
- **工具**: `shap_plot_interactions.py`

### Grad-CAM
- **应用**: MRI解剖定位
- **目标**: 骶髂关节区域
- **微调**: 3个epoch, LR=1×10⁻⁴

## 限制和注意事项

### 数据限制
- **MRI样本量**: 仅8个受试者
- **性别偏差**: 可能存在的算法偏差
- **外部验证**: 需要多中心验证

### 技术限制
- **零变异性**: 固定随机种子导致交叉验证结果完全一致
- **完美分类**: Random Forest在某些指标上达到1.000
- **校准问题**: 集成模型校准略差于单独模型

## 未来改进方向

1. **多中心联合学习**: 隐私保护扩展
2. **GAN数据增强**: 合成MRI增强
3. **注意力机制**: 更深层融合
4. **贝叶斯校准**: 不确定性量化
5. **公平性约束**: 减少算法偏差

---

*本文档基于DDI-AS项目的LaTeX论文自动生成，包含所有重要的脚本和算法信息。* 