import os
import numpy as np
import xarray as xr
from glob import glob
import time

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader
import data_preprocess as dp

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')


def CCMP_download(year, output_dir):
    import requests

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)\
            Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0'
    }
    url = (
        f'https://coastwatch.pfeg.noaa.gov/erddap/griddap/ccmp_31_LonPM180.nc?'
        f'uwnd[({year}-01-01T00:00:00Z):4:({year}-12-31T00:00:00Z)]'
        f'[({config_loader.lat_min}):1:({config_loader.lat_max})]'
        f'[({config_loader.lon_min}):1:({config_loader.lon_max})],'
        f'vwnd[({year}-01-01T00:00:00Z):4:({year}-12-31T00:00:00Z)]'
        f'[({config_loader.lat_min}):1:({config_loader.lat_max})]'
        f'[({config_loader.lon_min}):1:({config_loader.lon_max})]'
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'ccmp_v31_wind_cnsea_{year}.nc')
    temp_path = os.path.join(output_dir, f'temp_ccmp_v31_wind_cnsea_{year}.nc')

    print(f'downloading {year} year data...')
    response = requests.get(url, headers=HEADERS, stream=True)
    if response.status_code == 200:
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        with xr.open_dataset(temp_path) as ds:
            dp.encode_nc(ds, output_path, complevel=4)
        os.remove(temp_path)

        print(f'Complete: {output_path}\n')
    else:
        print(f'Failed {year}: HTTP {response.status_code}\n')

    time.sleep(18)


def CCMP_interp(nc_path):
    parts = ncpath.rsplit('_', 1)
    out_path = f'{parts[0]}_interp_{parts[1]}'

    with xr.open_dataset(nc_path) as ds:
        # 1) crop spatial domain
        ds = ds.sel(
            latitude=slice(config_loader.lat_min, config_loader.lat_max),
            longitude=slice(config_loader.lon_min, config_loader.lon_max),
        )

        # 2) build complete daily time axis for the year range
        years = ds.time.dt.year.values
        start = np.datetime64(f'{years[0]}-01-01')
        end = np.datetime64(f'{years[-1]}-12-31') + np.timedelta64(1, 'D')
        full_daily = np.arange(start, end, np.timedelta64(1, 'D'))

        # 3) reindex -> missing days become NaN -> linear interpolate
        ds = ds.reindex(time=full_daily)
        ds = ds.interpolate_na(dim='time', method='linear')

        # 4) edge NaN handling
        for var in ds.data_vars:
            if ds[var].isnull().sum():
                # forward fill and backward fill
                ds[var] = ds[var].ffill(dim='time').bfill(dim='time')

        dp.encode_nc(ds, out_path)
        print(f'{len(ds.time)} days written to: {out_path}')


def CCMP_slice_date(inpath, outpath):
    with xr.open_dataset(inpath) as ds:
        ds = ds.sel(
            latitude=slice(config_loader.lat_min, config_loader.lat_max),
            longitude=slice(config_loader.lon_min, config_loader.lon_max),
        )
        for time_label, month_ds in ds.resample(time='1ME'):
            year, month = str(time_label)[:7].split('-')

            newfilename = f'ccmp_v31_cnsea_{year}{month}.nc'
            newfilepath = os.path.join(outpath, str(year), newfilename)
            os.makedirs(os.path.dirname(newfilepath), exist_ok=True)

            dp.encode_nc(month_ds, newfilepath)
            print(f'{newfilepath} saved')


if __name__ == '__main__':
    ncpath = glob(os.path.join(r'data\raw\CCMP', '*.nc'))[0]
    CCMP_interp(ncpath)
