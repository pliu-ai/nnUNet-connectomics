# Watershed 3D Binary Mask Segmentation

这个模块提供了对3D binary mask进行watershed分割的功能，支持TIFF和NIfTI (.nii.gz)格式的文件。

## 功能特性

- 支持TIFF和NIfTI (.nii.gz)格式的3D图像文件
- 使用SimpleITK处理NIfTI文件，保持affine变换信息
- 使用tifffile处理TIFF文件
- 可调节的watershed参数
- 支持单文件和批量文件夹处理

## 使用方法

### 单文件处理

```bash
python watershed.py -i input_mask.nii.gz -o output_segmentation.nii.gz
```

### 批量文件夹处理

```bash
python watershed.py -i /path/to/input/folder -o /path/to/output/folder --folder --pattern "*.nii.gz"
```

### 参数说明

- `-i, --input`: 输入文件或文件夹路径
- `-o, --output`: 输出文件或文件夹路径
- `--min_distance`: 种子之间的最小距离 (默认: 5)
- `--sigma`: 距离变换的高斯模糊sigma值 (默认: 1.0)
- `--min_size`: 保留对象的最小大小 (默认: 100)
- `--connectivity`: 连通性 (6或26, 默认: 6)
- `--folder`: 处理整个文件夹而不是单个文件
- `--pattern`: 文件夹处理的文件模式 (默认: *.nii.gz)

## 算法流程

1. 加载3D binary mask
2. 应用高斯模糊平滑
3. 计算距离变换
4. 寻找局部最大值作为种子
5. 应用watershed算法
6. 移除小对象
7. 使用cc3d进行连通域分析
8. 保存结果

## 依赖库

- numpy
- SimpleITK (用于NIfTI文件)
- tifffile (用于TIFF文件)
- scikit-image
- scipy
- cc3d

## 示例

```python
from watershed import watershed_3d, watershed_from_file

# 处理单个文件
watershed_from_file('input.nii.gz', 'output.nii.gz', 
                   min_distance=5, sigma=1.0, min_size=100)

# 或者直接处理numpy数组
import numpy as np
binary_mask = np.random.randint(0, 2, (100, 100, 100))
segmentation = watershed_3d(binary_mask, min_distance=5, sigma=1.0, min_size=100)
```
