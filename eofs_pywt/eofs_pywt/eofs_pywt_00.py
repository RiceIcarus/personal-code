import numpy as np
import pandas as pd
from eofs.standard import Eof
import pywt
from scipy.signal import detrend
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec


def eof_analysis(data3d: np.ndarray, lat: np.ndarray, n_modes: int = 1) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    **EOF分析函数**
    函数功能: 对三维数据进行EOF分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度，一维数据
    :param n_modes: 模态数，默认为1
    :return:
        - EOF模态，三维数据（模态数，纬度，经度）
        - PC（主成分）时间序列，二维数据（时间，模态数）
        - var_frac方差解释度，一维数据
    """
    nt, ny, nx = data3d.shape
    data2d = data3d.reshape(nt, -1)  # 展平成二维

    weights_2d = np.sqrt(np.cos(np.deg2rad(lat)))  # 纬度权重
    weights_1d = np.tile(weights_2d, nx)

    solver = Eof(data2d, weights=weights_1d)
    eofs = solver.eofs(neofs=n_modes)  # eof模态
    eofs = (eofs / weights_1d[None, :]).reshape(n_modes, ny, nx)  # 还原成三维，除去权重
    pcs = solver.pcs(npcs=n_modes)  # pc时间序列，一列一个模态
    var_frac = solver.varianceFraction(neigs=n_modes)  # 方差解释度
    return eofs, pcs, var_frac


def morlet_cwt_pc(data3d: np.ndarray, lat: np.ndarray, time: np.ndarray, n_modes: int = 1, dt_out: str = 'Y') \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    **morlet小波分析函数**
    函数功能: 用eof分析的pc时间序列做morlet小波分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度，一维数据
    :param time: 时间序列，一维数据
    :param n_modes: 模态数，默认为1
    :param dt_out: 时间平均类型，'Y'年平均，'M'月平均，默认为'Y'
    :return:
        - coeffs小波系数，二维数据（尺度，时间）
        - periods各尺度周期，一维数据
        - new_time处理后的时间，一维数据
    """
    eofs, pcs, var_frac = eof_analysis(data3d, lat, n_modes=n_modes)
    pc = pcs[:, 0]
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)
    pc_mean = pd.Series(pc, index=time).resample(dt_out).mean()  # 年平均
    new_time = pc_mean.index
    pc_y = detrend(pc_mean.values)  # 去趋势

    scales = np.arange(2, len(pc_y) // 2 + 1)
    coeffs, freq = pywt.cwt(pc_y, scales, 'morl', sampling_period=1.0)  # 小波变换
    periods = 1 / freq
    return coeffs, periods, new_time


def morlet_cwt_mean(data3d: np.ndarray, lat: np.ndarray, time: np.ndarray, dt_out: str = 'Y') \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    **morlet小波分析函数**
    函数功能: 对温度数据直接做平均后做morlet小波分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度，一维数据
    :param time: 时间序列，一维数据
    :param dt_out: 可选参数，时间平均类型，'Y'年平均，'M'月平均，默认为'Y'
    :return:
        - coeffs小波系数，二维数据（尺度，时间）
        - periods各尺度周期，一维数据
        - new_time处理后的时间，一维数据
    """
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)
    nt, ny, nx = data3d.shape

    weights_2d = np.sqrt(np.cos(np.deg2rad(lat)))  # 纬度权重
    weights_3d = np.broadcast_to(weights_2d.reshape(ny, 1), (ny, nx))
    area_mean = (data3d * weights_3d[None, ...]).sum(axis=(1, 2)) / weights_3d.sum()

    ts = pd.Series(area_mean, index=time)
    ts = ts.resample(dt_out).mean()
    new_time = ts.index
    ts_y = detrend(ts.values)

    scales = np.arange(2, len(ts_y) // 2 + 1)
    coeffs, freq = pywt.cwt(ts_y, scales, 'morl', sampling_period=1.0)  # 小波变换
    periods = 1 / freq
    return coeffs, periods, new_time


def plot_eofs_res(eofs: np.ndarray, pcs: np.ndarray, var_frac: np.ndarray, time: np.ndarray, lon: np.ndarray,
                  lat: np.ndarray, n_modes: int = 1):
    """
    **eofs分析结果绘制函数**
    函数功能: 将eof_analysis函数输出的eofs分析结果绘制成图
    :param eofs: EOF模态，三维数据（模态数，纬度，经度）
    :param pcs: PC（主成分）时间序列，二维数据（时间，模态数）
    :param var_frac: var_frac方差解释度，一维数据
    :param time: 时间序列，一维数据
    :param lon: 经度，一维数据
    :param lat: 纬度，一维数据
    :param n_modes: 模态数，默认为1
    :return: 无返回值，会输出一张绘制好的eofs分析图结果
    """
    plt.figure(figsize=(16, 3 * n_modes))
    for i in range(n_modes):
        gs = gridspec.GridSpec(n_modes, 2, width_ratios=[1, 1])

        # 画pc时间序列
        ax1 = plt.subplot(gs[2 * i])
        ax1.plot(time, pcs[:, i], color='k')
        print(f'PC{i + 1}({var_frac[i] * 100:.2f}%)')
        ax1.set_title(f'PC{i + 1}({var_frac[i] * 100:.2f}%)')
        ax1.set_xlabel('time')
        ax1.set_ylabel('Amplitude')

        # 画eof模态
        ax2 = plt.subplot(gs[2 * i + 1])
        im = ax2.imshow(eofs[i], cmap='RdBu_r', origin='lower', aspect='auto',
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()])
        ax2.set_title(f'EOF{i + 1}')
        plt.colorbar(im, ax=ax2)
        plt.xlabel('Longitude')  # 经度
        plt.ylabel('Latitude')  # 纬度
        plt.grid(True)

        xticks = np.arange(np.floor(lon.min()), np.ceil(lon.max()) + 1, 30)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'{int(x)}°' for x in xticks])
        yticks = np.arange(np.floor(lat.min()), np.ceil(lat.max()) + 1, 30)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f'{int(y)}°' for y in yticks])

    plt.tight_layout()
    plt.show()


# 绘制morlet小波分析结果
def plot_morlet_cwt_res(coeffs: np.ndarray, periods: np.ndarray, time: np.ndarray, type: str = 'real'):
    """
    **morlet小波分析结果绘制函数**
    函数功能: 将morlet_cwt_pc函数或morlet_cwt_mean函数输出的morlet小波分析结果绘制成图
    :param coeffs: 小波系数，二维数据（尺度，时间）
    :param periods: 各尺度周期，一维数据
    :param time: 时间序列，一维数据
    :param type: 绘图形式，'real'画实部图，'power'画功率图，默认为'real'
    :return: 无返回值，会输出一张绘制好的morlet小波分析结果图
    """
    plt.figure(figsize=(9, 4))
    plt.yscale('log')
    plt.ylim(min(periods), max(periods))
    plt.title('Morlet Wavelet')
    plt.xlabel('Time')
    plt.ylabel('Period')
    if type == 'real':
        plt.contourf(time, periods, np.real(coeffs), levels=31, cmap='RdBu_r')  # 画实部图
        plt.colorbar(label='Real Part')
    elif type == 'power':
        plt.contourf(time, periods, np.abs(coeffs) ** 2, levels=31, cmap='YlOrRd')  # 画功率图
        plt.colorbar(label='Power')
    plt.tight_layout()
    plt.show()
