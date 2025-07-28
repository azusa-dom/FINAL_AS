# AS Diagnosis AI System

基于双通路人工智能框架的强直性脊柱炎诊断系统

## 📋 项目概述

本项目实现了一个创新的双通路AI诊断框架，用于强直性脊柱炎(Ankylosing Spondylitis, AS)的早期诊断。该系统解决了多模态数据异步问题，通过独立的临床和影像学管道提供可靠的诊断支持。

## 🏗️ 系统架构

### 双通路设计
- **临床数据管道**: 基于大规模EHR数据的ClinicalNet模型
- **MRI分析管道**: 基于小样本MRI数据的ImagingNet模型
- **融合就绪架构**: 支持未来多模态融合的模块化设计

### 技术特点
- ✅ 容器化部署 (Docker)
- ✅ HL7 FHIR兼容API
- ✅ 数据版本控制 (DVC)
- ✅ 可解释性分析 (Grad-CAM, SHAP)
- ✅ 概率校准和决策分析

## 📁 项目结构

```
FINAL_AS/
├── src/
│   ├── clinical_data_src/          # 临床数据处理
│   │   ├── clinical_data_preparation/
│   │   ├── training_clinical_data/
│   │   └── evaluation_clinical_data/
│   ├── mri_src/                    # MRI分析
│   │   ├── preprocessing/
│   │   ├── feature_extraction/
│   │   ├── analysis/
│   │   └── gradcam/
│   ├── utils/                      # 通用工具
│   ├── visualization/              # 可视化模块
│   └── api/                        # FHIR API接口
├── data/                           # 数据目录
├── models/                         # 模型文件
├── results/                        # 结果输出
├── Dockerfile                      # 容器配置
├── requirements.txt                # 依赖包
└── README.md                       # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd FINAL_AS

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 临床数据预处理
python src/clinical_data_src/clinical_data_preparation/preprocess_clinical_final.py \
    data/raw/clinical_data.csv \
    data/processed/clinical/

# MRI数据预处理
python src/mri_src/preprocessing/preprocess.py \
    data/raw/mri/ \
    data/processed/mri/
```

### 3. 模型训练

```bash
# 训练临床模型
python src/clinical_data_src/training_clinical_data/train_clinical_mondrian.py \
    --data_dir data/processed/clinical/ \
    --model_dir models/clinical/ \
    --epochs 50

# MRI特征提取和分析
python src/mri_src/analysis/make_l2o_predictions.py \
    --data-root data/processed/mri/ \
    --out-csv results/mri/l2o_predictions.csv
```

### 4. 启动API服务

```bash
# 使用Docker
docker build -t as-diagnosis-ai .
docker run -p 8080:8080 as-diagnosis-ai

# 或直接运行
python src/api/fhir_server.py
```

## 📊 性能指标

### 临床模型 (ClinicalNet)
- **AUROC**: 0.924 (95% CI: 0.915-0.932)
- **敏感性**: 98.6%
- **特异性**: 77.9%
- **校准误差 (ECE)**: 0.016

### MRI模型 (ImagingNet)
- **AUROC**: 0.83 (permutation p = 0.017)
- **样本量**: 8个受试者 (39个切片)
- **特征维度**: 512维

## 🔬 方法学特点

### 1. 数据不平衡处理
- SMOTE过采样技术
- 类别权重平衡
- 分层交叉验证

### 2. 概率校准
- 温度缩放校准
- 期望校准误差 (ECE) 评估
- 决策曲线分析 (DCA)

### 3. 可解释性
- SHAP特征重要性分析
- Grad-CAM注意力映射
- 特征空间几何分析

### 4. 小样本学习
- ImageNet预训练特征提取
- Leave-Two-Out交叉验证
- 方向性校正机制

## 📈 结果可视化

系统提供多种可视化功能：

```python
# 校准曲线
python src/clinical_data_src/evaluation_clinical_data/plot_overall_metrics.py

# Grad-CAM注意力图
python src/mri_src/gradcam/As_run_sij_gradcam_analysis.py

# 特征空间投影
python src/mri_src/mri_feature_analysis/feature_space_geometry.py
```

## 🔧 API使用

### 临床数据诊断
```bash
curl -X POST "http://localhost:8080/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "request_type": "clinical",
    "clinical_data": {
      "patient_id": "P001",
      "age": 35,
      "sex": "M",
      "hla_b27": "positive",
      "esr": 45.2,
      "crp": 18.5
    }
  }'
```

### MRI诊断
```bash
curl -X POST "http://localhost:8080/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "request_type": "mri",
    "mri_data": {
      "patient_id": "P001",
      "image_path": "/path/to/mri/image.nii.gz",
      "sequence_type": "T1"
    }
  }'
```

## 📚 参考文献

本系统基于以下研究论文实现：
- 强直性脊柱炎诊断延迟的临床挑战
- 多模态AI中的非配对数据障碍
- 双通路AI框架的设计与实现

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- 项目Issues: [GitHub Issues]

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员和开发者。
