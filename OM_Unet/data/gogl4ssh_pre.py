import os
import xarray as xr
from glob import glob

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader
import data_preprocess as dp

import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')


def SSH_download(year, path):
    import copernicusmarine

    # help(copernicusmarine.subset)
    # print(copernicusmarine.subset.__doc__)

    copernicusmarine.subset(
        username='yzhang12345678910111213141516171819',
        password='1470_Copernicus',
        dataset_id='cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D',
        dataset_version='202411',
        variables=['sla'],
        minimum_longitude=config_loader.lon_min,
        maximum_longitude=config_loader.lon_max,
        minimum_latitude=config_loader.lat_min,
        maximum_latitude=config_loader.lat_max,
        start_datetime=f'{year}-01-01T00:00:00',
        end_datetime=f'{year}-12-31T00:00:00',
        coordinates_selection_method='strict-inside',
        netcdf_compression_level=4,
        output_directory=path,
    )


def SSH_download_all(path):
    for year in range(2024, 2025):
        os.makedirs(path, exist_ok=True)
        SSH_download(year, path)


def SSH_downsample_slice():
    inpath = r'data/raw/GOGL4SSH'
    outpath = r'data/GOGL4SSH'

    for year in range(1993, 2024):
        filepath = glob(os.path.join(inpath, f'*{year}-12-31.nc'))[0]
        with xr.open_dataset(filepath) as ds:
            ds = ds.sel(
                latitude=slice(config_loader.lat_min, config_loader.lat_max),
                longitude=slice(config_loader.lon_min, config_loader.lon_max),
            )
            ds = dp.downsample_nc(ds, 2)

            for time_label, month_ds in ds.resample(time='1ME'):
                year, month = str(time_label)[:7].split('-')
                newfilename = f'cmems_ssh_cnsea_{year}{month}.nc'
                newfilepath = os.path.join(outpath, str(year), newfilename)
                os.makedirs(os.path.dirname(newfilepath), exist_ok=True)
                dp.encode_nc(month_ds, newfilepath, chunk=True)

        print(f'{year} year processing finished')


if __name__ == '__main__':
    path = r'data\raw\GOGL4SSH'
    SSH_download_all(path)

    # SSH_downsample_slice()
    # ncpath = r'data\raw\GOGL4SSH\1993\dt_cnsea_allsat_phy_l4_199301.nc'
    # with xr.open_dataset(ncpath) as ds:
    #     print(ds)
