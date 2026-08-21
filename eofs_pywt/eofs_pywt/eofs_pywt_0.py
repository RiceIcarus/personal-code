import numpy as np
import pandas as pd
from eofs.standard import Eof
import pywt
from scipy.signal import detrend


def eof_analysis(data3d, lat, n_modes=1):
    """
    **EOF分析函数**
    函数功能：对三维数据进行EOF分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度数据
    :param n_modes: 可选参数，模态数，默认为1
    :return:
        - EOF模态（numpy.ndarray，维度为（模态数，纬度，经度））
        - PC（主成分）时间序列（numpy.ndarray，维度为（时间，模态数））
        - var_frac方差解释度（numpy.ndarray，一维）
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


def morlet_cwt_pc(data3d, lat, time, n_modes=1, dt_out='Y'):
    """
    **morlet小波分析函数**
    函数功能：用eof分析的pc时间序列做morlet小波分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度数据
    :param time: 时间序列数据
    :param n_modes: 可选参数，模态数，默认为1
    :param dt_out: 可选参数，时间平均类型，'Y'年平均，'M'月平均，默认为'Y'
    :return:
        - coeffs小波系数（维度为（尺度，时间））
        - periods各尺度周期
        - new_time处理后的时间
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


def morlet_cwt_mean(data3d, lat, time, dt_out='Y'):
    """
    **morlet小波分析函数**
    函数功能：对温度数据直接做平均后做morlet小波分析
    :param data3d: 三维数据（时间，纬度，经度）
    :param lat: 纬度数据
    :param time: 时间序列数据
    :param dt_out: 可选参数，时间平均类型，'Y'年平均，'M'月平均，默认为'Y'
    :return:
        - coeffs小波系数（维度为（尺度，时间））
        - periods各尺度周期
        - new_time处理后的时间
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
