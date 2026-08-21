import os
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt

import config_loader
from evaluation import rmse, mae, bias

import warnings

warnings.filterwarnings('ignore', message='facecolor will have no effect')


def _plot_map(
    ax,
    lon,
    lat,
    data,
    cmap='RdYlBu_r',
    title='',
    cbar_label='',
    cbar_orientation='vertical',
    cbar_shrink=None,
    cbar_pad=None,
    cbar_aspect=None,
    **norm_kwargs,
):
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', **norm_kwargs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.5)

    lon_ticks = np.arange(int(lon.min()), int(lon.max()) + 1, 5)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f'{t}°E' for t in lon_ticks], fontsize=10)
    lat_ticks = np.arange(int(lat.min()), int(lat.max()) + 1, 5)
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels([f'{t}°N' for t in lat_ticks], fontsize=10)

    ax.set_xlabel('longitude', fontsize=12, labelpad=10)
    ax.set_ylabel('latitude', fontsize=12, labelpad=15)
    ax.set_title(title, fontsize=14 if title else 12, pad=20)

    if cbar_orientation == 'horizontal':
        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            shrink=0.9 if cbar_shrink is None else cbar_shrink,
            pad=0.08 if cbar_pad is None else cbar_pad,
            aspect=35 if cbar_aspect is None else cbar_aspect,
        )
    else:
        cbar = plt.colorbar(
            im,
            ax=ax,
            shrink=0.72 if cbar_shrink is None else cbar_shrink,
            pad=0.02 if cbar_pad is None else cbar_pad,
            aspect=25 if cbar_aspect is None else cbar_aspect,
        )
    cbar.set_label(cbar_label, fontsize=12)
    return im


def plotnc1(nc_path, varname, time, depth=None):
    with xr.open_dataset(nc_path) as ds:
        if 'depth' in ds.dims:
            ds = ds.isel(depth=depth)
        data_p = ds[varname].isel(time=time).values
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        units = ds[varname].attrs.get('units', 'unknown')

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    title = f'{varname} distribution in the coastal waters of china'
    _plot_map(ax, lon, lat, data_p, title=title, cbar_label=f'{varname} ({units})')
    plt.tight_layout()
    plt.show()


def plot_ssd5_first_day(
    nc_path=r'data\SSD_5\2021\sea_searface_data_5var_202101.nc',
    glory_path=r'data\GLORYS4V1_thetao\2021\cmems_cnsea_0.25deg_thetao_202101.nc',
    quiver_step=6,
    glory_depth=0,
):
    with xr.open_dataset(nc_path) as ds:
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat = ds[lat_name].values
        lon = ds[lon_name].values
        time_value = ds['time'].values[0]
        sst = ds['sst'].isel(time=0).values
        sla = ds['sla'].isel(time=0).values
        uwnd = ds['uwnd'].isel(time=0).values
        vwnd = ds['vwnd'].isel(time=0).values
        wind_speed = np.sqrt(uwnd**2 + vwnd**2)

    with xr.open_dataset(glory_path) as ds_glory:
        glory = ds_glory['thetao'].isel(time=0, depth=glory_depth).values
        glory_lat = ds_glory['latitude'].values
        glory_lon = ds_glory['longitude'].values
        glory_depth_value = float(np.ravel(ds_glory['depth'].isel(depth=glory_depth).values)[0])
        # glory_units = ds_glory['thetao'].attrs.get('units', 'C°')

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True
        )
        ax_glory, ax_sst, ax_sla, ax_wind = axes.flat

        _plot_map(
            ax_glory,
            glory_lon,
            glory_lat,
            glory,
            cmap='RdYlBu_r',  # plasma, RdYlBu_r
            title=f'GLORYS THETAO ({glory_depth_value:.2f} m)',
            cbar_label='thetao (C°)',
        )

        _plot_map(
            ax_sst,
            lon,
            lat,
            sst,
            cmap='RdYlBu_r',
            title='SST',
            cbar_label=f'sst ({ds["sst"].attrs.get("units", "C°")})',
        )

        sla_abs_max = np.nanmax(np.abs(sla))
        _plot_map(
            ax_sla,
            lon,
            lat,
            sla,
            cmap='RdBu_r',
            vmin=-sla_abs_max,
            vmax=sla_abs_max,
            title='SLA',
            cbar_label=f'sla ({ds["sla"].attrs.get("units", "m")})',
        )

        _plot_map(
            ax_wind,
            lon,
            lat,
            wind_speed,
            cmap='YlOrRd',
            title='Wind Speed + Vector',
            cbar_label=f'wind speed ({ds["uwnd"].attrs.get("units", "m/s")})',
        )

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        ax_wind.quiver(
            lon_grid[::quiver_step, ::quiver_step],
            lat_grid[::quiver_step, ::quiver_step],
            uwnd[::quiver_step, ::quiver_step],
            vwnd[::quiver_step, ::quiver_step],
            transform=ccrs.PlateCarree(),
            color='black',
            scale=180,
            width=0.0022,
        )

    fig.suptitle(f'5 variables on first day: {str(time_value)[:10]}', fontsize=16)
    plt.savefig(r'paper\U-Net OM paper\figs\input variables', bbox_inches='tight')
    plt.show()


def compare_nc(pred_path, outpath=None):
    with xr.open_dataset(pred_path) as ds_pred:
        pred = np.squeeze(ds_pred['thetao'].isel(time=0).values)
        lat = ds_pred['latitude'].values
        lon = ds_pred['longitude'].values
        pred_depth = ds_pred['depth'].values.item()
        pred_time = ds_pred['time'].values[0]

    date = os.path.basename(pred_path).split('.')[0].split('_')[-1]
    year, month = date[:4], date[4:6]
    glory_path = next((Path(config_loader.ds_info['glory']['path']) / str(year)).glob(f'*{year}{month}.nc'))

    with xr.open_dataset(glory_path) as ds_glory:
        label = np.squeeze(ds_glory['thetao'].sel(time=pred_time, depth=pred_depth, method='nearest').values)

    diff = pred - label

    vmin = min(np.nanmin(pred), np.nanmin(label))
    vmax = max(np.nanmax(pred), np.nanmax(label))
    diff_abs_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.6), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True
    )
    ax_pred, ax_label, ax_diff = axes

    _plot_map(
        ax_pred, lon, lat, pred, vmin=vmin, vmax=vmax, cmap='RdYlBu_r', title='Prediction', cbar_label='°C'
    )
    _plot_map(
        ax_label, lon, lat, label, vmin=vmin, vmax=vmax, cmap='RdYlBu_r', title='GLORYS', cbar_label='°C'
    )
    _plot_map(
        ax_diff,
        lon,
        lat,
        diff,
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
        cmap='RdBu_r',
        title='Difference',
        cbar_label='°C',
    )

    fig.suptitle('ResCBAM U-Net (4 layers)', fontsize=16)  # , x=0.7, y=1.03

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def rmse_depth(outpath=None):
    unet4l_rmse = [
        np.mean([0.6094, 0.6097, 0.6098]),
        np.mean([0.6312, 0.6361, 0.6374]),
        np.mean([0.6666, 0.6715, 0.6738]),
        np.mean([0.7734, 0.7778, 0.7793]),
        np.mean([0.8467, 0.8517, 0.8628]),
    ]
    resunet4l_rmse = [
        np.mean([0.6063, 0.6078, 0.6165]),
        np.mean([0.6214, 0.6256, 0.6266]),
        np.mean([0.6712, 0.6702, 0.6862]),
        np.mean([0.7583, 0.7769, 0.7813]),
        np.mean([0.8469, 0.8627, 0.8667]),
    ]
    rescbam_rmse = [
        np.mean([0.5948, 0.5979, 0.6004]),
        np.mean([0.6096, 0.6133, 0.6154]),
        np.mean([0.6615, 0.6675, 0.6715]),
        np.mean([0.7510, 0.7528, 0.7528]),
        np.mean([0.8121, 0.8169, 0.8296]),
    ]
    depths = np.arange(5)
    depths_ticks = [0.49, 5.08, 9.57, 15.81, 25.21]
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(depths - width, unet4l_rmse, width, label='U-Net (4-layer)', color='#4480E1')
    bars2 = ax.bar(depths, resunet4l_rmse, width, label='Res U-Net (4-layer)', color='#25DAA1')
    bars3 = ax.bar(depths + width, rescbam_rmse, width, label='ResCBAM U-Net (4-layer)', color='#E7A262')

    ax.set_xlabel('Subsurface Depth (m)', fontsize=10)
    ax.set_ylabel('RMSE (°C)', fontsize=10)
    ax.set_title('RMSE Comparison Across Depths', fontsize=14)
    ax.set_xticks(depths)
    ax.set_xticklabels(depths_ticks)
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center',
                va='bottom',
                fontsize=8.5,
            )

    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rmse_timeseries(outpath=None):
    import csv
    from datetime import datetime
    import matplotlib.dates as mdates

    models = {
        'U-Net (4-layer)': r'result\Unet structure\Unet_4L\eval_year_2021.csv',
        'Res U-Net (4-layer)': r'result\Unet structure\Res_Unet_4L\eval_year_2021.csv',
        'ResCBAM U-Net (4-layer)': r'result\Unet structure\ResCBAM_Unet_4L\eval_year_2021.csv',
    }

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for (label, path), color in zip(models.items(), colors):
        dates, rmses = [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dates.append(datetime.strptime(row['date'], '%Y-%m-%d'))
                rmses.append(float(row['rmse']))
        ax.plot(dates, rmses, label=label, color=color, linewidth=2)

    ax.set_xlim(datetime(2021, 1, 1), datetime(2021, 12, 31))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax.get_xticklabels(), fontsize=11)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title('Daily RMSE Comparison (2021)', fontsize=16)
    ax.legend()

    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def coastal_distance_rmse(pred_paths, model_labels=None, outpath=None):
    """Compute RMSE binned by distance from coastline for one or more predictions."""
    import os as _os

    if model_labels is None:
        model_labels = [_os.path.basename(p) for p in pred_paths]

    lines = ['=== Coastal Distance RMSE Analysis ===']
    seasons = {
        '01': 'winter',
        '02': 'winter',
        '12': 'winter',
        '03': 'spring',
        '04': 'spring',
        '05': 'spring',
        '06': 'summer',
        '07': 'summer',
        '08': 'summer',
        '09': 'autumn',
        '10': 'autumn',
        '11': 'autumn',
    }

    for pred_path, label in zip(pred_paths, model_labels):
        with xr.open_dataset(pred_path) as ds_pred:
            pred = np.squeeze(ds_pred['thetao'].isel(time=0).values)
            pred_time = ds_pred['time'].values[0]
            pred_depth = float(np.ravel(ds_pred['depth'].values)[0])

        date_str = _os.path.basename(pred_path).split('.')[0].split('_')[-1]
        year, month = date_str[:4], date_str[4:6]
        season = seasons.get(month, 'unknown')

        glory_dir = config_loader.ds_info[config_loader.target_ds]['path']
        glory_path = next(Path(glory_dir).glob(f'{year}/*{year}{month}.nc'))

        with xr.open_dataset(glory_path) as ds_glory:
            target = np.squeeze(
                ds_glory['thetao'].sel(time=pred_time, depth=pred_depth, method='nearest').values
            )

        ocean_mask = np.isfinite(target)
        dist_pixels = distance_transform_edt(ocean_mask)
        km_per_cell = 27.8
        dist_km = dist_pixels * km_per_cell

        bins = [(0, 100), (100, 300), (300, np.inf)]
        bin_labels = ['<=100 km', '100-300 km', '>300 km']

        lines.append(f'\n{label} | {date_str} ({season})')
        for (lo, hi), bin_label in zip(bins, bin_labels):
            bin_mask = (dist_km >= lo) & (dist_km < hi)
            valid = bin_mask & ocean_mask
            n = int(valid.sum())
            if n > 0:
                r = rmse(pred[valid], target[valid])
                m = mae(pred[valid], target[valid])
                b = bias(pred[valid], target[valid])
                lines.append(f'  {bin_label}: RMSE={r:.4f}, MAE={m:.4f}, Bias={b:.4f}')
            else:
                lines.append(f'  {bin_label}: no pixels')

    # Append to result.txt
    result_path = r'result\result.txt'
    _os.makedirs(_os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    # ssh_path = r'data\GOGL4SSH\1993\cmems_ssh_cnsea_199302.nc'
    # sst_path = r'data\OISST\1993\oisst_v21_sst_cnsea_199302.nc'
    # wind_path = r'data\CCMP\1993\ccmp_v31_cnsea_199302.nc'
    glory_path = r'data\GLORYS4V1_thetao_600\1993\cmems_cnsea_0.25deg_thetao_199301.nc'
    # ssd_path = r'data\SSD_5\1993\sea_searface_data_5var_199301.nc'
    with xr.open_dataset(glory_path) as ds:
        print(ds['depth'])

    # pred_path = r'result\Unet structure\ResCBAM_Unet_4L\L0\20260613_174739\pred_glory_20210101.nc'
    # out_path = r'paper\20210101_comparison_row.png'
    # compare_nc(pred_path, out_path)

    # outpath_d = r'paper\U-Net OM\figs\rmse_depth_comparison.png'
    # rmse_depth(outpath_d)

    # outpath_t = r'paper\U-Net OM\figs\rmse_timeseries.png'
    # plot_rmse_timeseries(outpath_t)

    # plot_ssd5_first_day()

    # unet4l_rmse = [
    #     np.mean([0.6094, 0.6097, 0.6098]),
    #     np.mean([0.6312, 0.6361, 0.6374]),
    #     np.mean([0.6666, 0.6715, 0.6738]),
    #     np.mean([0.7734, 0.7778, 0.7793]),
    #     np.mean([0.8467, 0.8517, 0.8628]),
    # ]
    # resunet4l_rmse = [
    #     np.mean([0.6063, 0.6078, 0.6165]),
    #     np.mean([0.6214, 0.6256, 0.6266]),
    #     np.mean([0.6712, 0.6702, 0.6862]),
    #     np.mean([0.7583, 0.7769, 0.7813]),
    #     np.mean([0.8469, 0.8627, 0.8667]),
    # ]
    # rescbam_rmse = [
    #     np.mean([0.5948, 0.5979, 0.6004]),
    #     np.mean([0.6096, 0.6133, 0.6154]),
    #     np.mean([0.6615, 0.6675, 0.6715]),
    #     np.mean([0.7510, 0.7528, 0.7528]),
    #     np.mean([0.8121, 0.8169, 0.8296]),
    # ]
    # print(rescbam_rmse)
