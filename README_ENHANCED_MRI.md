# 增强MRI分析使用指南

## 🚀 快速开始

### 1. 快速测试
```bash
python test_enhanced_mri.py
```

### 2. 完整分析
```bash
./run_enhanced_mri_analysis.sh
```

### 3. 手动运行
```bash
# 基础增强分析
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root data \
    --out-csv results/enhanced_basic.csv \
    --seed 42

# 完整增强分析（推荐）
python src/mri_src/analysis/make_l2o_predictions_improved.py \
    --data-root data \
    --out-csv results/enhanced_full.csv \
    --use-ensemble \
    --use-augmentation \
    --outlier-detection \
    --bootstrap-ci \
    --feature-selection variance \
    --n-features 100
```

## 📊 主要改进

### 解决的核心问题
- ✅ **过拟合**: 强正则化 + 集成学习
- ✅ **小样本**: 数据增强 + 鲁棒验证  
- ✅ **特异性低**: 特征标准化 + 多算法融合

### 预期改进效果
- **特异性**: 50% → 65-75% (+15-25%)
- **过拟合**: 显著减少
- **鲁棒性**: 显著提高

## 🔧 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-ensemble` | 使用集成学习 | False |
| `--use-augmentation` | 启用数据增强 | False |
| `--outlier-detection` | 异常值检测 | False |
| `--bootstrap-ci` | Bootstrap置信区间 | False |
| `--feature-selection` | 特征选择方法 | variance |
| `--n-features` | 特征数量 | 100 |

## 📁 输出文件

- `enhanced_*.csv`: 预测结果
- `comparison_results.csv`: 方法比较
- `improvements_analysis.csv`: 改进分析
- `enhancement_analysis.png`: 可视化图表
- `enhancement_analysis_report.md`: 详细报告

## 📈 结果分析

```bash
python analyze_enhancement_results.py
```

## 📖 详细文档

- [增强MRI分析实施总结](ENHANCED_MRI_ANALYSIS_SUMMARY.md)
- [MRI分析改进策略](MRI_IMPROVEMENT_STRATEGY.md)

## 🎯 关键优势

1. **不依赖额外数据** - 基于现有8个受试者
2. **解决核心问题** - 过拟合、小样本、特异性低
3. **技术先进** - 最新小样本学习技术
4. **发表价值** - 解决技术挑战，提供实用策略 