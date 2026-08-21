from pathlib import Path
import numpy as np
import xarray as xr

import config_loader


def valid_pred_target(pred, target):
    """Return paired finite prediction and target values, ignoring NaN land points."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    mask = np.isfinite(pred) & np.isfinite(target)
    return pred[mask], target[mask]


def mae(pred, target):
    """Mean Absolute Error: mean(abs(pred - target)); lower means smaller average error."""
    pred_valid, target_valid = valid_pred_target(pred, target)
    if pred_valid.size == 0:
        return np.nan
    return float(np.mean(np.abs(pred_valid - target_valid)))


def rmse(pred, target):
    """Root Mean Square Error: sqrt(mean((pred - target)^2)); lower means fewer large errors."""
    pred_valid, target_valid = valid_pred_target(pred, target)
    if pred_valid.size == 0:
        return np.nan
    diff = pred_valid - target_valid
    return float(np.sqrt(np.mean(diff * diff)))


def bias(pred, target):
    """Mean Bias Error: mean(pred - target); positive is warmer, negative is colder."""
    pred_valid, target_valid = valid_pred_target(pred, target)
    if pred_valid.size == 0:
        return np.nan
    return float(np.mean(pred_valid - target_valid))


def corr(pred, target):
    """Pearson correlation: cov(pred, target) / (std(pred) * std(target)); closer to 1 means better."""
    pred_valid, target_valid = valid_pred_target(pred, target)
    if pred_valid.size <= 1:
        return np.nan
    return float(np.corrcoef(pred_valid, target_valid)[0, 1])


def get_evaluations(pred, target):
    return {
        'bias': bias(pred, target),
        'mae': mae(pred, target),
        'rmse': rmse(pred, target),
        'corr': corr(pred, target),
    }


def read_pred_target_data(pred_path):
    date = pred_path.split('_')[-1].split('.')[0]

    year, year_month = date[:4], date[:6]
    glory_file = next((Path(config_loader.ds_info['glory']['path']) / year).glob(f'*{year_month}.nc'))

    with xr.open_dataset(pred_path) as ds_pred, xr.open_dataset(glory_file) as ds_glory:
        pred = np.squeeze(ds_pred['thetao'].isel(time=0).values)
        pred_time = ds_pred['time'].values[0]
        pred_depth = float(np.ravel(ds_pred['depth'].values)[0])
        target = np.squeeze(ds_glory['thetao'].sel(time=pred_time, depth=pred_depth, method='nearest').values)

    evals = get_evaluations(pred, target)
    for key, value in evals.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    pred_path = r'result\20260602_234426\pred_thetao_20010101.nc'
    read_pred_target_data(pred_path)
