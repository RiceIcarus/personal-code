import numpy as np
import xarray as xr
import calendar
import time
import os
from glob import glob

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader
import data_preprocess as dp

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')


def downsample_glory_25(inds):
    """Downsample GLORY data to 0.25° via interpolation and resampling"""
    res_up = 1 / 24
    lat_24 = np.arange(config_loader.lat_min + res_up / 2, config_loader.lat_max, res_up)
    lon_24 = np.arange(config_loader.lon_min + res_up / 2, config_loader.lon_max, res_up)

    results = []
    for var in inds.data_vars:
        inds[var] = inds[var].astype('float32')
    for t in range(inds.sizes['time']):
        ds_t = inds.isel(time=t)
        ds_t_up = ds_t.interp(
            latitude=lat_24, longitude=lon_24, method='nearest', assume_sorted=True
        )
        ds_t_coarse = ds_t_up.coarsen(latitude=6, longitude=6).reduce(
            dp.mean_with_nan, keep_attrs=True
        )
        results.append(ds_t_coarse)

    return xr.concat(results, dim='time')


def GLORY_download(year, month, path):
    import copernicusmarine

    # help(copernicusmarine.subset)
    # print(copernicusmarine.subset.__doc__)

    _, days = calendar.monthrange(year, month)

    copernicusmarine.subset(
        username='yzhang12345678910111213141516171819',
        password='1470_Copernicus',
        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
        variables=['uo', 'vo', 'so', 'thetao'],
        minimum_longitude=config_loader.lon_min,
        maximum_longitude=config_loader.lon_max,
        minimum_latitude=config_loader.lat_min,
        maximum_latitude=config_loader.lat_max,
        start_datetime=f'{year}-{month:02d}-01T00:00:00',
        end_datetime=f'{year}-{month:02d}-{days}T00:00:00',
        minimum_depth=0.49402499198913574,
        maximum_depth=5727.9169921875,
        output_directory=f'{path}/{year}',
    )


def GLORY_download_process():
    path_12 = 'data/raw/GLORYS12V1'
    path_4 = 'data/raw/GLORYS4V1'

    for year in range(2018, 2025):
        os.makedirs(os.path.join(path_4, str(year)), exist_ok=True)
        for month in range(1, 13):
            GLORY_download(year, month, path_12)

            time1 = time.time()
            ncpath_12 = glob(os.path.join(path_12, str(year), f'*_{year}-{month:02d}*.nc'))[0]
            newncname = f'cmems_mod_cnsea_phy_my_0.25deg_P1D-m_{year}{month:02d}.nc'
            ncpath_4 = os.path.join(path_4, str(year), newncname)

            with xr.open_dataset(ncpath_12) as ds:
                ds = downsample_glory_25(ds)
                dp.encode_nc(ds, ncpath_4, complevel=4)
            time2 = time.time()
            print(f'{ncpath_4} processing finished in {time2 - time1:.2f} seconds\n')


def GLORY_shrink():
    path_4 = 'data/z_raw/GLORYS4V1'
    path_4_s = 'data/GLORYS4V1_thetao_600'
    for year in range(1993, 2025):
        os.makedirs(os.path.join(path_4_s, str(year)), exist_ok=True)
        for month in range(1, 13):
            ncpath_4 = glob(os.path.join(path_4, str(year), f'*_{year}{month:02d}.nc'))[0]
            newncname = f'cmems_cnsea_0.25deg_thetao_{year}{month:02d}.nc'
            ncpath_4_s = os.path.join(path_4_s, str(year), newncname)

            with xr.open_dataset(ncpath_4) as ds:
                ds = ds.drop_vars(['so', 'uo', 'vo'])
                ds = ds.isel(depth=[0, 4, 7, 10, 13, 16, 18, 20, 22, 24, 26, 28, 30, 32])
                dp.encode_nc(ds, ncpath_4_s, chunk=True)
        print(f'{year} year processing finished')


if __name__ == '__main__':
    GLORY_shrink()
