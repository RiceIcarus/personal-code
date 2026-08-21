import os
from glob import glob
import numpy as np
import xarray as xr
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config_loader
import data_preprocess as dp


R_EARTH = 6_371_000.0
RHO_AIR = 1.225
DRAG_COEFF = 1.3e-3


def calc_wind_stress_curl(ds):
    """
    Calculate wind stress curl on a regular lat/lon grid.

    Input variables must have dimensions: time, latitude, longitude.
    Output unit is approximately N m-3.
    """
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    u = ds['uwnd']
    v = ds['vwnd']
    wind_speed = np.sqrt(u**2 + v**2)

    rho_air = RHO_AIR
    drag_coeff = DRAG_COEFF
    tau_x = rho_air * drag_coeff * wind_speed * u
    tau_y = rho_air * drag_coeff * wind_speed * v

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    y = R_EARTH * lat_rad

    d_tau_x_dy = np.gradient(tau_x.values, y, axis=1)
    d_tau_y_dlon_rad = np.gradient(tau_y.values, lon_rad, axis=2)
    d_tau_y_dx = d_tau_y_dlon_rad / (R_EARTH * np.cos(lat_rad))[None, :, None]

    curl_data = d_tau_y_dx - d_tau_x_dy

    return curl_data


def add_wind_stress_curl(infile):
    with xr.open_dataset(infile) as ds:
        curl_data = calc_wind_stress_curl(ds)
        ds['wsc'] = xr.DataArray(
            curl_data,
            coords=ds['uwnd'].coords,
            dims=ds['uwnd'].dims,
            name='wind_stress_curl',
            attrs={
                'long_name': 'wind stress curl',
                'units': 'N m-3',
                'formula': 'd(tau_y)/dx - d(tau_x)/dy',
                'rho_air': RHO_AIR,
                'drag_coefficient': DRAG_COEFF,
            },
        )

    return ds


def add_all():
    for year in range(config_loader.year_start, config_loader.year_end + 1):
        for month in range(1, 13):
            infile = glob(os.path.join(r'data/SSD_4', str(year), f'*{year}{month:02d}.nc'))[0]
            ds = add_wind_stress_curl(infile)
            var_num = len(ds.data_vars)
            outfile = os.path.join(r'data/SSD_5', str(year), f'sea_searface_data_{var_num}var_{year}{month:02d}.nc')

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            dp.encode_nc(ds, outfile)

        print(f'{year} processing finished')


if __name__ == '__main__':
    add_all()
