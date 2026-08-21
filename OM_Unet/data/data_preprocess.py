import os
import xarray as xr
import numpy as np
from glob import glob
import calendar

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_loader

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')


def encode_nc(ds, outfile, complevel=1, shuffle=True, chunk=False):
    """encode and save the input dataset"""
    encoding = {}
    for coord in ds.coords:
        if coord != 'time':
            encoding[coord] = {
                'dtype': 'float32',
                'zlib': True,
                'complevel': complevel,
                'shuffle': shuffle,
            }
    for var in ds.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': complevel,
            'shuffle': shuffle,
        }

        dims, sizes = ds[var].dims, ds[var].sizes
        if chunk and dims == ('time', 'latitude', 'longitude'):
            encoding[var]['chunksizes'] = (1, sizes['latitude'], sizes['longitude'])
        elif chunk and dims == ('time', 'depth', 'latitude', 'longitude'):
            encoding[var]['chunksizes'] = (1, 1, sizes['latitude'], sizes['longitude'])

    ds.to_netcdf(outfile, encoding=encoding, engine='netcdf4')


def check_days(path):
    wrong_files = []
    ncpaths = glob(os.path.join(path, '**', '*.nc'), recursive=True)

    for ncpath in ncpaths:
        date = ncpath.split('_')[-1].replace('.nc', '')
        if len(date) == 4:
            whole_days = 366 if calendar.isleap(int(date)) else 365
            with xr.open_dataset(ncpath) as ds:
                real_days = ds.sizes['time']
                if real_days != whole_days:
                    wrong_files.append((ncpath, whole_days, real_days))
        elif len(date) == 6:
            whole_days = calendar.monthrange(int(date[:4]), int(date[-2:]))[1]
            with xr.open_dataset(ncpath) as ds:
                real_days = ds.sizes['time']
                if real_days != whole_days:
                    wrong_files.append((ncpath, whole_days, real_days))
        else:
            print('Wrong file name!')
            return False

    if wrong_files:
        for wrong_file in wrong_files:
            print(
                f'{wrong_file[0]}  is incomplete! '
                f'It should have {wrong_file[1]} days, but actually {wrong_file[2]}'
            )
    else:
        print('All files are complete!')
    return True


def mean_with_nan(arr, axis=('longitude', 'latitude')):
    """calculate the mean while ignoring NaN values or setting the result to NaN."""
    axis_size = np.prod([arr.shape[i] for i in axis])
    nan_ratio = np.isnan(arr).sum(axis=axis) / float(axis_size)
    mean_val = np.nanmean(arr, axis=axis)
    result = np.where(nan_ratio > 0.5, np.nan, mean_val)
    return result


def downsample_nc(inds, n=2):
    """Downsample the data to 0.25° using mean aggregation."""
    for var in inds.data_vars:
        inds[var] = inds[var].astype('float32')
    ds_coarsened = inds.coarsen(latitude=n, longitude=n, boundary='trim')
    ds_25 = ds_coarsened.reduce(mean_with_nan, keep_attrs=True)

    return ds_25


def rename_file(path):
    filepaths = glob(os.path.join(path, '**', '*.nc'), recursive=True)
    for filepath in filepaths:
        newfilepath = filepath.replace('thetao_0.25deg', '0.25deg_thetao')
        os.rename(filepath, newfilepath)


def concat_in_one():
    for year in range(config_loader.year_start, config_loader.year_end + 1):
        for month in range(1, 13):
            ssh_path = glob(os.path.join(r'data/GOGL4SSH', str(year), f'*{year}{month:02d}.nc'))[0]
            sst_path = glob(os.path.join(r'data/OISST', str(year), f'*{year}{month:02d}.nc'))[0]
            wind_path = glob(os.path.join(r'data/CCMP', str(year), f'*{year}{month:02d}.nc'))[0]

            with xr.open_dataset(ssh_path) as ds:
                sla_data = ds['sla'].values
                time = ds['time'].values
                lat = ds['latitude'].values
                lon = ds['longitude'].values

            with xr.open_dataset(sst_path) as ds:
                sst_data = ds['sst'].values

            with xr.open_dataset(wind_path) as ds:
                uwnd_data = ds['uwnd'].values
                vwnd_data = ds['vwnd'].values

            ds_ssd = xr.Dataset(
                data_vars={
                    'sla': (['time', 'latitude', 'longitude'], sla_data),
                    'sst': (['time', 'latitude', 'longitude'], sst_data),
                    'uwnd': (['time', 'latitude', 'longitude'], uwnd_data),
                    'vwnd': (['time', 'latitude', 'longitude'], vwnd_data),
                },
                coords={
                    'time': time,
                    'latitude': lat,
                    'longitude': lon,
                },
            )
            var_num = len(ds_ssd.data_vars)
            ssd_path = os.path.join(
                r'data/SSD_4', str(year), f'sea_searface_data_{var_num}var_{year}{month:02d}.nc'
            )
            os.makedirs(os.path.dirname(ssd_path), exist_ok=True)
            encode_nc(ds_ssd, ssd_path)

        print(f'{year} processing finished')


if __name__ == '__main__':
    concat_in_one()
