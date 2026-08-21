# 中国近海次表层海洋要素重建

本项目基于多源海表观测数据与深度学习方法，开展中国近海次表层海洋要素重建研究。当前已完成次表层温度场重建，并系统比较了不同 U-Net 深度、残差连接和 CBAM 注意力机制的效果。

研究区域覆盖渤海、黄海、东海及南海北部（11--43°N，104--136°E）。后续将扩展至次表层盐度、纬向流速和经向流速重建，并开展更多代表性深度学习方法的对比。

## 当前工作

- 整合海表温度、海表高度、纬向风和经向风等多源海表变量，重建次表层温度。
- 构建并比较 2 层、3 层、4 层和 5 层 U-Net。
- 在 4 层 U-Net 基线上比较残差连接及 CBAM 通道-空间注意力增强。
- 使用 RMSE、MAE、Bias 和 Pearson 相关系数评价重建结果。
- 分析不同深度、季节以及近岸和外海区域的重建误差特征。

## 主要结果

在固定数据集、输入变量和训练流程的控制实验下：

- 4 层 U-Net 在 2--5 层 U-Net 中表现出较好的精度与训练稳定性。
- 残差连接本身未带来稳定增益。
- ResCBAM U-Net_4L 在第一目标层取得平均 RMSE 0.5977 ℃，较基础 4 层 U-Net 降低 1.97%。
- 在 0.49 m、5.08 m、9.57 m、15.81 m 和 25.21 m 的五个目标层中，ResCBAM U-Net_4L 均取得最低 RMSE；25.21 m 处平均 RMSE 为 0.820 ℃，基础 U-Net 为 0.854 ℃。

## 项目结构

```text
.
├── config.toml                 # 实验配置
├── main.py                     # 训练、验证和推理入口
├── config_loader.py            # 配置加载与运行目录管理
├── data_loader.py              # 数据读取、样本构建和掩膜处理
├── data_normalization.py       # 归一化统计量的计算与复用
├── network.py                  # U-Net 及结构增强模型
├── solver.py                   # 训练、验证、保存和推理流程
├── losses.py                   # 含无效网格掩膜的回归损失
├── evaluation.py               # RMSE、MAE、Bias、R 等评价指标
├── plotnc.py                   # 结果绘图
├── plot_3d_sea_depth.py        # 三维深度结果绘图
├── test.py                     # 测试辅助脚本
├── result/
│   └── result.txt              # 主要实验结果汇总
└── data/                       # 本地数据及数据处理脚本（数据不随仓库提供）
```

## 数据与配置

当前温度重建的默认配置位于 `config.toml`：

| 类别 | 内容 |
| --- | --- |
| 输入变量 | 海表温度（`sst`）、海表高度异常（`sla`）、纬向风（`uwnd`）、经向风（`vwnd`） |
| 重建目标 | GLORYS 次表层温度（`thetao`） |
| 训练时段 | 1993--2017 年 |
| 验证时段 | 2018--2023 年 |
| 默认模型 | `ResCBAM_Unet_4L` |
| 空间范围 | 11--43°N，104--136°E |

原始数据不包含在本仓库中。请根据 `config.toml` 中 `[ds_info]` 各数据集的 `path` 字段，将预处理后的数据放置在本地 `data/` 目录下，或将这些字段修改为自己的数据路径。

不同来源的数据需要统一为相同的日尺度、空间网格、研究区域和海陆掩膜。目标变量中的无效网格可保留为 NaN，训练和评价时会通过掩膜排除。

## 环境

建议使用 Python 3.10 及支持 CUDA 的 PyTorch 环境。核心依赖包括：

```text
torch
numpy
xarray
netCDF4
pandas
tomli
tqdm
matplotlib
scikit-learn
```

可按自身 CUDA 环境安装 PyTorch，其余依赖可使用：

```powershell
pip install numpy xarray netCDF4 pandas tomli tqdm matplotlib scikit-learn
```

## 运行

先在 `config.toml` 中确认模型类型、数据路径、目标深度、训练时段和训练参数。

默认入口位于 `main.py`：

```powershell
python main.py
```

当前 `main.py` 会依次执行训练、验证，并对若干指定日期进行推理示例。若只需要训练、验证或单日推理，请在 `main.py` 中保留对应的 `Solver` 方法调用，例如：

```python
from solver import Solver

solver = Solver()
solver.train()
solver.valid(show_evaluations=True)
solver.test(2021, 1, 1)
```

每次新训练会在 `result/<时间戳>/` 下保存运行配置、训练日志、最佳模型和相关输出。评价结果应使用 `RMSE`、`MAE`、`Bias` 和 `R` 进行报告；训练损失仅用于训练过程的模型选择。


## 说明

- `data/` 中只跟踪数据处理 Python 脚本，不包含原始数据。
- 当前项目以次表层温度重建为基线，后续工作将扩展到温盐流场整体重建及多种深度学习方法的比较。
