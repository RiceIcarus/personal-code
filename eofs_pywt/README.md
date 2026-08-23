# EOF 与 Morlet 小波分析

这是一个面向气象、海洋等经纬度网格数据的 Python 小工具，结合经验正交函数（EOF）分析和连续小波变换（CWT），用于提取主要空间模态、对应的时间变化以及不同时间尺度上的周期特征。

## 功能简介

### EOF（经验正交函数）

`eof_analysis` 接收三维数组 `data3d`（维度顺序为 **时间、纬度、经度**）和一维纬度数组 `lat`：

- 按纬度使用 `sqrt(cos(latitude))` 进行面积权重，减小高纬网格面积差异的影响；
- 将网格展平后调用 `eofs.standard.Eof` 求解空间模态；
- 返回 EOF 空间型、PC 主成分时间序列和各模态的方差解释率。

返回值形状分别为 `(n_modes, ny, nx)`、`(nt, n_modes)` 和 `(n_modes,)`。

### Morlet 连续小波变换（PyWavelets）

项目提供两种分析路径：

- `morlet_cwt_pc`：先做 EOF，再对第一个 PC 序列进行重采样、去趋势和 Morlet 小波变换；
- `morlet_cwt_mean`：先对空间网格做纬度加权平均，再进行重采样、去趋势和小波变换。

小波函数返回系数 `coeffs`、对应周期 `periods` 和处理后的时间轴 `new_time`。`dt_out` 可使用 pandas 的重采样频率（例如 `'Y'` 年平均、`'M'` 月平均）。

## 安装

建议使用 Python 3.8 及以上版本：

```bash
pip install -r requirements.txt
```

也可以在项目目录执行可编辑安装：

```bash
pip install -e .
```

## 快速示例

```python
import numpy as np
import pandas as pd
from eofs_pywt import eof_analysis, morlet_cwt_pc

# data3d: (时间, 纬度, 经度)，lat: (纬度,)
data3d = np.random.randn(120, 10, 20)
lat = np.linspace(15, 75, 10)
time = pd.date_range('2000-01-01', periods=120, freq='MS')

eofs, pcs, var_frac = eof_analysis(data3d, lat, n_modes=3)
coeffs, periods, new_time = morlet_cwt_pc(
    data3d, lat, time, n_modes=3, dt_out='Y'
)
```

其中，`var_frac` 可用于判断每个 EOF 模态对总方差的贡献；小波系数的绝对值平方 `abs(coeffs) ** 2` 可作为局地周期功率使用。

## 绘图与示例脚本

`eofs_pywt.py` 包含：

- `read_nc`：按年份拼接 CRA40 NetCDF 数据；
- `plot_nc`：绘制单时刻经纬度场；
- `plot_eofs_res`：并列绘制 EOF 空间型和 PC 时间序列；
- `plot_morlet_cwt_res`：绘制小波实部或功率谱。

直接运行该脚本前，请将 CRA40 文件放在脚本中 `read_nc` 约定的目录结构下，并根据数据修改变量名、年份和区域选择。

## 项目结构

```text
eofs_pywt/
├── eofs_pywt.py             # 数据读取、分析与绘图示例
├── eofs_pywt/
│   ├── __init__.py           # 导出核心分析函数
│   ├── eofs_pywt_0.py        # 轻量核心实现
│   └── eofs_pywt_00.py       # 含类型标注和绘图函数的实现
├── requirements.txt
└── pyproject.toml
```

## 使用提示

- 输入数据应为规则经纬度网格，且纬度数组长度必须与 `data3d.shape[1]` 一致。
- 时间序列需要能够转换为 `pandas.DatetimeIndex`；小波分析前会进行重采样和线性去趋势。
- EOF 模态的正负号在数学上等价，解读空间型与 PC 时应结合起来判断。
- 小波结果两端可能存在边界效应，解释长周期信号时应保持谨慎。

