import os
import xarray as xr
import numpy as np
from glob import glob
import time

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader
import data_preprocess as dp

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')


def OISST_download(year, output_dir):
    import requests

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)\
            Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0'
    }
    url = (
        f'https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc?'
        f'sst[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)]'
        f'[(0.0):1:(0.0)]'
        f'[({config_loader.lat_min}):1:({config_loader.lat_max})]'
        f'[({config_loader.lon_min}):1:({config_loader.lon_max})]'
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'oisst-avhrr-v02r01_cnsea_{year}.nc')
    temp_path = os.path.join(output_dir, f'temp_oisst-avhrr-v02r01_cnsea_{year}.nc')

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


def OISST_download_all():
    outpath = r'data\raw\OISST'
    for year in range(1993, 2025):
        OISST_download(year, outpath)


def OISST_interp(ncpath):
    parts = ncpath.rsplit('_', 1)
    out_path = f'{parts[0]}_interp_{parts[1]}'

    with xr.open_dataset(ncpath) as ds:
        # Preserve the original intraday anchor, e.g. 12:00:00 for OISST daily files.
        first_time = ds['time'].values[0]
        day_anchor = first_time.astype('datetime64[D]')
        time_offset = first_time - day_anchor

        years = ds.time.dt.year.values
        start = np.datetime64(f'{years[0]}-01-01') + time_offset
        end = np.datetime64(f'{years[-1]}-12-31') + time_offset + np.timedelta64(1, 'D')
        full_daily = np.arange(start, end, np.timedelta64(1, 'D'))

        overlap = np.intersect1d(ds['time'].values, full_daily)
        if overlap.size == 0:
            raise ValueError(
                f'constructed daily time axis does not overlap source timestamps in {ncpath}. '
                f'first source time: {ds["time"].values[0]!s}, first target time: {full_daily[0]!s}'
            )

        land_masks = {}
        for var in ds.data_vars:
            if 'time' in ds[var].dims:
                land_masks[var] = ds[var].isnull().all(dim='time')

        # Reindex missing days to NaN, then interpolate only across the time gaps.
        ds = ds.reindex(time=full_daily)
        ds = ds.interpolate_na(dim='time', method='linear')

        # Fill edge gaps if the missing period touches the start or end of the series.
        for var in ds.data_vars:
            if 'time' in ds[var].dims and bool(ds[var].isnull().any()):
                ds[var] = ds[var].ffill(dim='time').bfill(dim='time')
            if var in land_masks:
                ds[var] = ds[var].where(~land_masks[var])

        dp.encode_nc(ds, out_path, complevel=4)
        print(f'{len(ds.time)} days written to: {out_path}')


def OISST_slice_date(ncpath, outdir):

    with xr.open_dataset(ncpath) as ds:
        ds = ds.isel(zlev=0, drop=True)
        ds = ds.sel(
            latitude=slice(config_loader.lat_min, config_loader.lat_max),
            longitude=slice(config_loader.lon_min, config_loader.lon_max),
        )

        for time_label, month_ds in ds.resample(time='1ME'):
            year, month = str(time_label)[:7].split('-')

            newfilename = f'oisst_v21_sst_cnsea_{year}{month}.nc'
            newfilepath = os.path.join(outpath, str(year), newfilename)
            os.makedirs(os.path.dirname(newfilepath), exist_ok=True)

            dp.encode_nc(month_ds, newfilepath, chunk=True)
            print(newfilepath + ' has finished processing')


if __name__ == '__main__':
    inpath = r'data\raw\OISST'
    outpath = r'data\OISST'

    # OISST_download_all()
    # dp.check_days(r'data\raw\OISST')
    # path = r'data\raw\OISST\oisst-avhrr-v02r01_cnsea_interp_2021.nc'
    # OISST_interp(path)
    for year in range(2021, 2026):
        path = glob(os.path.join(inpath, f'*interp_{year}.nc'))[0]
        OISST_slice_date(path, outpath)

    # ncpath = r'data\raw\OISST\oisst-avhrr-v02r01_cnsea_2021.nc'
    # with xr.open_dataset(ncpath) as ds:
    #     print(ds)
    #     print(ds['time'].values, '\n')
    #     print(ds['latitude'].values, '\n')
    #     print(ds['longitude'].values, '\n')
