# Professional Research Figures for MRI-AS Analysis

## 📊 生成的图表概览

本文件夹包含为MRI-AS研究项目创建的高质量、发表级别的科研图表。所有图表均采用专业的科研风格设计，具有300 DPI的高分辨率，适合学术期刊发表。

## 🎨 图表列表

### 1. **SHAP Analysis Figure** (`shap_analysis_figure.png/pdf`)
**用途：** 特征重要性分析和模型可解释性
**内容：**
- Panel A: Gradient Boosting特征重要性
- Panel B: Random Forest特征重要性  
- Panel C: 跨模型特征重要性对比
- 临床解释文本框

**论文放置位置：** 第3.1.1节 "Discrimination Performance" 之后
**引用方式：** "As shown in Figure 3.1.2, SHAP analysis reveals HLA-B27 as the strongest predictor..."

### 2. **Model Performance Comparison** (`model_performance_comparison.png/pdf`)
**用途：** 综合模型性能对比分析
**内容：**
- Panel A: 模型歧视性能（AUROC对比）
- Panel B: 分类指标对比（Precision, Recall, F1-Score）
- Panel C: 模型校准（Log Loss）
- Panel D: 期望校准误差（ECE）
- 性能总结统计

**论文放置位置：** 第3.1.1节 "Discrimination Performance" 之后
**引用方式：** "Figure 3.1.3 demonstrates comprehensive model performance across multiple metrics..."

### 3. **Ensemble Integration Analysis** (`ensemble_integration_analysis.png/pdf`)
**用途：** 集成模型性能分析
**内容：**
- Panel A: 个体模型vs集成模型性能
- Panel B: 各模态性能增益分析
- 集成策略说明文本框

**论文放置位置：** 第3.3.1节 "Multi-Modal Fusion" 之后
**引用方式：** "The ensemble integration results, depicted in Figure 3.3.2, show a performance gain of +0.007..."

### 4. **Cross-Validation Analysis** (`cross_validation_analysis.png/pdf`)
**用途：** 交叉验证稳定性分析
**内容：**
- Panel A: 临床模型CV稳定性
- Panel B: MRI模型CV性能
- Panel C: 临床模型方差分析
- Panel D: CV方法对比
- CV总结统计

**论文放置位置：** 第3.1.2节 "Cross-Validation Stability" 之后
**引用方式：** "Cross-validation stability analysis (Figure 3.1.4) demonstrates high consistency across folds..."

## 🔧 技术特点

### 设计风格
- **科学色彩方案：** 使用专业的科研色彩调色板
- **清晰排版：** 避免文字重叠，确保可读性
- **统计注释：** 包含完整的统计信息和误差条
- **临床解释：** 每个图表都包含临床意义解释文本框

### 技术规格
- **分辨率：** 300 DPI（发表级别）
- **格式：** PNG（用于演示）+ PDF（用于发表）
- **字体：** Serif字体，适合学术发表
- **网格：** 专业网格线，便于数据读取

### 数据准确性
- **数据源：** 来自`accurate_data_results/`和`appendix_tables/`文件夹
- **验证：** 所有数据均经过交叉验证
- **一致性：** 与论文报告结果完全一致

## 📍 论文集成建议

### 图表编号建议
- Figure 3.1.2: SHAP Analysis Figure
- Figure 3.1.3: Model Performance Comparison  
- Figure 3.1.4: Cross-Validation Analysis
- Figure 3.3.2: Ensemble Integration Analysis

### 引用示例
```
"SHAP analysis (Figure 3.1.2) identified HLA-B27 as the strongest predictor 
(importance = 0.231), consistent with AS pathophysiology."

"The ensemble integration (Figure 3.3.2) achieved AUROC = 0.945, 
representing a +0.007 improvement over the best individual model."

"Cross-validation analysis (Figure 3.1.4) demonstrates high stability 
with standard deviations < 0.01 across all clinical models."
```

## 🎯 临床意义

### 主要发现
1. **HLA-B27主导：** 特征重要性分析确认HLA-B27是AS诊断的最强预测因子
2. **模型稳定性：** 交叉验证显示所有模型都具有高稳定性
3. **集成优势：** 多模态集成提供了显著的性能提升
4. **校准质量：** 模型校准分析揭示了不同模型的可靠性差异

### 临床应用价值
- **诊断支持：** 为临床医生提供可靠的AI辅助诊断工具
- **风险分层：** 基于概率校准的精确风险评估
- **个性化治疗：** 支持基于个体特征的精准治疗决策

## 📋 文件清单

```
professional_research_figures/
├── README.md                           # 本说明文档
├── shap_analysis_figure.png            # SHAP分析图 (PNG)
├── shap_analysis_figure.pdf            # SHAP分析图 (PDF)
├── model_performance_comparison.png    # 模型性能对比图 (PNG)
├── model_performance_comparison.pdf    # 模型性能对比图 (PDF)
├── ensemble_integration_analysis.png   # 集成分析图 (PNG)
├── ensemble_integration_analysis.pdf   # 集成分析图 (PDF)
├── cross_validation_analysis.png       # 交叉验证分析图 (PNG)
└── cross_validation_analysis.pdf       # 交叉验证分析图 (PDF)
```

## 🔄 更新历史

- **2025-08-02:** 初始版本创建
- 所有图表均基于准确数据生成
- 采用专业科研设计标准
- 确保与论文内容完全一致

---

*这些图表专为MRI-AS研究项目设计，具有发表级别的质量和专业性。* 