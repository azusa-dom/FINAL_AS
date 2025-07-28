# Bias Field Correction 模块

## 概述

这个模块提供了用于医学图像Bias Field校正的完整解决方案，基于SimpleITK的N4算法实现。该模块支持单个文件和批量处理，具有完善的错误处理、日志记录和参数配置功能。

## 主要功能

### 1. 单个文件处理
- 支持多种mask创建方法（Otsu、Binary、None）
- 可配置的N4算法参数
- 完善的错误处理和日志记录

### 2. 批量处理
- 自动扫描目录中的NIfTI文件
- 进度显示和统计信息
- 支持多种参数组合

### 3. 参数调优
- 收敛阈值控制
- 最大迭代次数设置
- B-spline参数配置
- 拟合层数控制

## 安装依赖

确保已安装以下依赖包：

```bash
pip install SimpleITK numpy
```

## 使用方法

### 命令行使用

#### 基本用法
```bash
python src/mri_src/preprocessing/bias_correction.py input_folder output_folder
```

#### 高级参数
```bash
python src/mri_src/preprocessing/bias_correction.py \
    input_folder \
    output_folder \
    --mask-method otsu \
    --convergence-threshold 0.001 \
    --max-iterations 50 \
    --spline-order 3 \
    --fitting-levels 4 \
    --control-points 4 \
    --log-level INFO
```

### Python API使用

#### 单个文件处理
```python
from mri_src.preprocessing.bias_correction import bias_field_correction_single

success = bias_field_correction_single(
    input_path="data/input.nii.gz",
    output_path="results/corrected.nii.gz",
    mask_method="otsu",
    convergence_threshold=0.001,
    max_iterations=50
)

if success:
    print("处理成功")
else:
    print("处理失败")
```

#### 批量处理
```python
from mri_src.preprocessing.bias_correction import bias_field_correction_batch

success_count, total_count = bias_field_correction_batch(
    input_folder="data/raw_mri",
    output_folder="results/bias_corrected",
    mask_method="otsu",
    convergence_threshold=0.001,
    max_iterations=50
)

print(f"成功处理 {success_count}/{total_count} 个文件")
```

## 参数说明

### Mask方法
- `otsu`: 使用Otsu阈值分割创建mask（推荐）
- `binary`: 使用简单二值化
- `none`: 使用全1mask

### N4算法参数
- `convergence_threshold`: 收敛阈值（默认0.001）
- `max_iterations`: 最大迭代次数（默认50）
- `spline_order`: B-spline阶数（默认3）
- `number_of_fitting_levels`: 拟合层数（默认4）
- `number_of_control_points`: 控制点数量（默认4）

### 日志级别
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息（默认）
- `WARNING`: 警告信息
- `ERROR`: 错误信息

## 性能优化建议

### 快速处理
```python
# 适用于快速预览或测试
bias_field_correction_single(
    input_path="input.nii.gz",
    output_path="output.nii.gz",
    convergence_threshold=0.01,
    max_iterations=20,
    number_of_fitting_levels=2
)
```

### 高质量处理
```python
# 适用于最终处理
bias_field_correction_single(
    input_path="input.nii.gz",
    output_path="output.nii.gz",
    convergence_threshold=0.0001,
    max_iterations=100,
    number_of_fitting_levels=6
)
```

### 平衡处理
```python
# 默认参数，平衡质量和速度
bias_field_correction_single(
    input_path="input.nii.gz",
    output_path="output.nii.gz"
)
```

## 错误处理

模块包含完善的错误处理机制：

1. **输入验证**: 检查文件格式和路径有效性
2. **异常捕获**: 捕获并记录处理过程中的异常
3. **状态返回**: 返回处理成功/失败状态
4. **日志记录**: 详细的错误信息记录

## 测试

运行测试脚本验证功能：

```bash
python src/mri_src/preprocessing/test_bias_correction.py
```

运行示例脚本：

```bash
python src/mri_src/preprocessing/bias_correction_example.py
```

## 与原始代码的改进

### 1. 错误处理
- ✅ 添加了输入文件验证
- ✅ 完善的异常处理机制
- ✅ 处理状态返回

### 2. 日志记录
- ✅ 结构化日志输出
- ✅ 可配置的日志级别
- ✅ 进度显示

### 3. 参数配置
- ✅ 可配置的N4算法参数
- ✅ 多种mask创建方法
- ✅ 命令行参数支持

### 4. 代码结构
- ✅ 模块化设计
- ✅ 类型注解
- ✅ 详细文档字符串
- ✅ 单元测试

### 5. 功能扩展
- ✅ 单个文件处理函数
- ✅ 批量处理优化
- ✅ 性能测试
- ✅ 使用示例

## 注意事项

1. **内存使用**: 大图像处理可能需要较多内存
2. **处理时间**: N4算法可能需要较长时间，特别是高质量设置
3. **文件格式**: 仅支持NIfTI格式（.nii, .nii.gz）
4. **依赖**: 需要SimpleITK库

## 故障排除

### 常见问题

1. **文件不存在错误**
   - 检查输入路径是否正确
   - 确保文件格式为NIfTI

2. **内存不足**
   - 减少图像尺寸
   - 使用快速处理参数

3. **处理时间过长**
   - 调整收敛阈值
   - 减少最大迭代次数
   - 减少拟合层数

4. **SimpleITK错误**
   - 检查SimpleITK版本
   - 确保图像数据完整性

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License 