import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
from eofs.standard import Eof
import pywt
from scipy.signal import detrend


# 生成伪数据
def generate_data(nt, ny, nx):
    time = np.arange(nt)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    T1 = np.sin(2 * np.pi * time / 20)
    T2 = np.sin(2 * np.pi * time / 12)
    A1 = np.sin(2 * np.pi * X / nx) * np.cos(2 * np.pi * Y / ny)
    A2 = np.sin(2 * np.pi * Y / ny) * np.cos(4 * np.pi * X / nx)
    data3d = T1[:, None, None] * A1[None, :, :] + T2[:, None, None] * A2[None, :, :]
    data3d += 0.2 * np.random.randn(nt, ny, nx)
    return data3d


# 读取 NC 数据
def read_nc(name):
    temdata = []
    for year in range(2001, 2021):
        nc_path = f"CRA40_{name}_nc/CRA40_{name}_{year}_GLB_2P5_YEAR_V1_0_0.nc"
        with xr.open_dataset(nc_path) as ds:
            temdata.append(ds)
    return xr.concat(temdata, dim='time')


# EOF 分析 输入：data3d(t, y, x), lat(y),n_mode模态数
#         返回：eof模态, pc时间序列, var_frac方差解释度
def eof_analysis(data3d, lat, n_modes=1):
    nt, ny, nx = data3d.shape
    data2d = data3d.reshape(nt, -1)  # 拉成 (t, y*x)

    weights_2d = np.sqrt(np.cos(np.deg2rad(lat)))  # 纬度权重
    weights_1d = np.tile(weights_2d, nx)  # 对应每个格点
    solver = Eof(data2d, weights=weights_1d)

    eofs = solver.eofs(neofs=n_modes)  # (n_modes, y*x)
    eofs = (eofs / weights_1d[None, :]).reshape(n_modes, ny, nx)
    pcs = solver.pcs(npcs=n_modes)  # (t, n_modes)
    var_frac = solver.varianceFraction(neigs=n_modes)
    return eofs, pcs, var_frac


# 3. Morlet 小波 —— 对 PC 序列
def morlet_cwt_pc(pc, time, dt_out='Y'):
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)
    pc_mean = pd.Series(pc, index=time).resample(dt_out).mean()
    new_time = pc_mean.index
    pc_y = detrend(pc_mean.values)

    scales = np.arange(2, len(pc_y) // 2 + 1)
    coeffs, freq = pywt.cwt(pc_y, scales, 'mor', sampling_period=1.0)
    periods = 1 / freq
    return coeffs, periods, new_time


# 4. Morlet 小波 —— 对区域平均序列
def morlet_cwt_mean(data3d, lat, time, dt_out='y'):
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)
    nt, ny, nx = data3d.shape

    weights_2d = np.sqrt(np.cos(np.deg2rad(lat)))
    weights_3d = np.broadcast_to(weights_2d.reshape(ny, 1), (ny, nx))
    area_mean = (data3d * weights_3d[None, ...]).sum(axis=(1, 2)) / weights_3d.sum()

    ts = pd.Series(area_mean, index=time)
    ts = ts.resample(dt_out).mean()
    new_time = ts.index
    ts_y = detrend(ts.values)

    scales = np.arange(2, len(ts_y) // 2 + 1)
    coeffs, freq = pywt.cwt(ts_y, scales, 'mor', sampling_period=1.0)
    periods = 1 / freq
    return coeffs, periods, new_time


# 5. 绘制 NC 单时刻二维图
def plot_nc(data, lat, lon):
    plt.figure(figsize=(10, 4))
    plt.contourf(lon, lat, data, levels=32, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

    ax = plt.gca()
    xticks = np.arange(np.floor(lon.min()), np.ceil(lon.max()) + 1, 30)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(x)}' for x in xticks])
    yticks = np.arange(np.floor(lat.min()), np.ceil(lat.max()) + 1, 30)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(y)}' for y in yticks])
    plt.tight_layout()
    plt.show()


# 6. 绘制 EOF 结果
def plot_eofs_res(eofs, pcs, var_frac, time, lon, lat, n_modes=1):
    plt.figure(figsize=(16, 3 * n_modes))
    for i in range(n_modes):
        gs = gridspec.GridSpec(n_modes, 2, width_ratios=[1, 2])

        # PC 时间序列
        ax1 = plt.subplot(gs[2 * i])
        ax1.plot(time, pcs[:, i], color='k')
        ax1.set_title(f'PC{i + 1} ({var_frac[i] * 100:.2f}%)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')

        # EOF 空间模态
        ax2 = plt.subplot(gs[2 * i + 1])
        im = ax2.imshow(eofs[i], cmap='RdBu_r', origin='lower', aspect='auto',
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()])
        ax2.set_title(f'EOF{i + 1}')
        plt.colorbar(im, ax=ax2)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)

        xticks = np.arange(np.floor(lon.min()), np.ceil(lon.max()) + 1, 30)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'{int(x)}' for x in xticks])
        yticks = np.arange(np.floor(lat.min()), np.ceil(lat.max()) + 1, 30)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f'{int(y)}' for y in yticks])

    plt.tight_layout()
    plt.show()


# 7. 绘制 Morlet 小波结果
def plot_morlet_cwt_res(coeffs, periods, time, type='real'):
    plt.figure(figsize=(9, 4))
    plt.yscale('log')
    plt.ylim(min(periods), max(periods))
    plt.title('Morlet Wavelet')
    plt.xlabel('Time')
    plt.ylabel('Period')

    if type == 'real':
        plt.contourf(time, periods, np.real(coeffs), levels=31, cmap='RdBu_r')
        plt.colorbar(label='Real Part')
    elif type == 'power':
        plt.contourf(time, periods, np.abs(coeffs) ** 2, levels=31, cmap='YlOrRd')
        plt.colorbar(label='Power')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    expname = 'TEMLand'  # 'TEM' or 'TEMLand'
    data = read_nc(expname)  # 读取 2001–2020 数据
    data = data.sel(latitude=slice(0, 90))  # 只要北半球
    print(data)

    # EOF 分析
    if expname == 'TEM':
        dataname = 'TMP_250mb'  # 250 hPa 温度
    elif expname == 'TEMLand':
        dataname = 'TMP_2maboveground'  # 2 m 气温
    else:
        raise ValueError('unsupported expname')
    eofs, pcs, var_frac = eof_analysis(data[dataname].values, data.latitude.values, n_modes=3)

    # 绘制 EOF
    plot_eofs_res(eofs, pcs, var_frac, data.time.values, data.longitude.values, data.latitude.values, n_modes=3)
    # 对第一主成分做 Morlet 小波
    coeffs, periods, new_time = morlet_cwt_pc(pcs[:, 0], data.time.values, dt_out='y')
    # 绘制小波
    plot_morlet_cwt_res(coeffs, periods, new_time, type='real')  # 或 'power'
