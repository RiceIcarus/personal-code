from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib.colors import Normalize


def _find_depth_prediction_files(root_dir, target_date='20210101'):
    root = Path(root_dir)
    pred_files = []

    for layer_dir in sorted(
        root.glob('L*'), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 99
    ):
        run_dirs = [p for p in layer_dir.iterdir() if p.is_dir()]
        if not run_dirs:
            continue

        pred_path = run_dirs[0] / f'pred_glory_{target_date}.nc'
        if pred_path.exists():
            pred_files.append(pred_path)

    return pred_files


def _plot_layers_3d(layers, title, stride, cmap, elev, azim):
    layers.sort(key=lambda item: item['depth'])
    vmin = min(np.nanmin(layer['data']) for layer in layers)
    vmax = max(np.nanmax(layer['data']) for layer in layers)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    for layer in layers:
        lon = layer['lon'][::stride]
        lat = layer['lat'][::stride]
        data = layer['data'][::stride, ::stride]
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        depth_grid = np.full_like(lon_grid, -layer['depth'], dtype=float)
        facecolors = cmap_obj(norm(data))
        facecolors[..., -1] = 0.6
        facecolors[np.isnan(data)] = (0.82, 0.86, 0.90, 0.2)

        ax.plot_surface(
            lon_grid,
            lat_grid,
            depth_grid,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08, aspect=24)
    cbar.set_label('Temperature (degC)', fontsize=11)

    ax.set_title(title, fontsize=15, pad=18)
    ax.set_xlabel('Longitude', labelpad=10)
    ax.set_ylabel('Latitude', labelpad=10)
    ax.set_zlabel('Depth (m)', labelpad=10)
    ax.view_init(elev=elev, azim=azim)

    ax.set_zticks([-layer['depth'] for layer in layers])
    ax.set_zticklabels([f'{layer["depth"]:.2f}' for layer in layers], fontsize=9)
    ax.set_box_aspect((1.35, 1.0, 0.55))
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.03, top=0.92)
    plt.show()


def plot_sea_depth_3d(
    root_dir=r'result\sea depth',
    target_date='20210101',
    var_name='thetao',
    stride=2,
    cmap='RdYlBu_r',
    elev=28,
    azim=-128,
):
    pred_files = _find_depth_prediction_files(root_dir, target_date=target_date)
    if not pred_files:
        raise FileNotFoundError(f'No prediction files found for {target_date} under {root_dir}')

    layers = []
    for pred_path in pred_files:
        with xr.open_dataset(pred_path) as ds:
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

            data = np.squeeze(ds[var_name].isel(time=0).values)
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            depth = float(np.ravel(ds['depth'].values)[0])

        layers.append(
            {
                'depth': depth,
                'lat': lat,
                'lon': lon,
                'data': data,
            }
        )

    _plot_layers_3d(
        layers,
        title=f'3D Predicted Sea Temperature Structure on {target_date}',
        stride=stride,
        cmap=cmap,
        elev=elev,
        azim=azim,
    )


def plot_glory_sea_depth_3d(
    data_path=r'data\GLORYS4V1_thetao_600\2021\cmems_cnsea_0.25deg_thetao_202101.nc',
    target_date='2021-01-01',
    var_name='thetao',
    stride=2,
    cmap='RdYlBu_r',
    elev=28,
    azim=-128,
):
    with xr.open_dataset(data_path) as ds:
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

        data = ds[var_name].sel(time=target_date).squeeze().values
        depths = ds['depth'].values
        lat = ds[lat_name].values
        lon = ds[lon_name].values

    layers = [
        {'depth': float(depth), 'lat': lat, 'lon': lon, 'data': layer_data}
        for depth, layer_data in zip(depths, data)
    ]
    _plot_layers_3d(
        layers,
        title=f'3D GLORYS Sea Temperature Structure on {target_date}',
        stride=stride,
        cmap=cmap,
        elev=elev,
        azim=azim,
    )


def plot_temperature_difference_3d(
    root_dir=r'result\sea depth',
    data_path=r'data\GLORYS4V1_thetao_600\2021\cmems_cnsea_0.25deg_thetao_202101.nc',
    target_date='20210101',
    var_name='thetao',
    stride=2,
    cmap='RdBu_r',
    elev=28,
    azim=-128,
):
    pred_files = _find_depth_prediction_files(root_dir, target_date=target_date)
    if not pred_files:
        raise FileNotFoundError(f'No prediction files found for {target_date} under {root_dir}')

    glory_date = f'{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}'
    with xr.open_dataset(data_path) as ds:
        glory_data = ds[var_name].sel(time=glory_date).squeeze().values
        glory_depths = ds['depth'].values

    layers = []
    for pred_path in pred_files:
        with xr.open_dataset(pred_path) as ds:
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

            prediction = np.squeeze(ds[var_name].isel(time=0).values)
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            depth = float(np.ravel(ds['depth'].values)[0])

        depth_index = int(np.abs(glory_depths - depth).argmin())
        layers.append(
            {
                'depth': depth,
                'lat': lat,
                'lon': lon,
                'data': prediction - glory_data[depth_index],
            }
        )

    max_abs_difference = max(np.nanmax(np.abs(layer['data'])) for layer in layers) * 0.8
    _plot_layers_3d(
        layers,
        title=f'3D Temperature Difference (Prediction - GLORYS) on {target_date}',
        stride=stride,
        cmap=cmap,
        elev=elev,
        azim=azim,
        vmin=-max_abs_difference,
        vmax=max_abs_difference,
        colorbar_label='Prediction - GLORYS (degC)',
    )


if __name__ == '__main__':
    plot_sea_depth_3d()
