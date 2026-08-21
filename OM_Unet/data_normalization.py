import copy
import json
import os
import shutil
import numpy as np
import xarray as xr
from glob import glob


def build_data_context(config_source=None):
    """Build a minimal data context from config or a saved config snapshot."""
    if isinstance(config_source, dict):
        ds_info = config_source['ds_info']
        ds_split = config_source['ds_split']
        depth = int(config_source['depth'])
    else:
        ds_info = config_source.ds_info
        ds_split = config_source.ds_split
        depth = int(config_source.depth)

    return {'ds_info': ds_info, 'ds_split': ds_split, 'depth': depth}


def iter_split_nc_paths(ds_path, ds_split_range):
    start_raw, end_raw = str(ds_split_range[0]), str(ds_split_range[1])

    if not (len(start_raw) == len(end_raw) and len(start_raw) in (4, 6)):
        raise ValueError(f'Inconsistent ds_split range: {ds_split_range}')

    if len(start_raw) == 4:
        start_yyyymm, end_yyyymm = start_raw + '01', end_raw + '12'
    else:
        start_yyyymm, end_yyyymm = start_raw, end_raw

    if start_yyyymm > end_yyyymm:
        raise ValueError(f'Invalid ds_split order: {ds_split_range}')

    start_year, end_year = int(start_yyyymm[:4]), int(end_yyyymm[:4])

    for year in range(start_year, end_year + 1):
        for nc_path in sorted(glob(os.path.join(ds_path, str(year), '*.nc'))):
            file_yyyymm = nc_path.split('_')[-1].split('.')[0]
            if start_yyyymm <= file_yyyymm <= end_yyyymm:
                yield nc_path


class DatasetNormalization:
    """Manage normalization stats for one explicit data context."""

    def __init__(self, run_dir, config):
        self.run_dir = run_dir
        self.target_ds = config['target_ds'] if isinstance(config, dict) else config.target_ds
        self.meta = self._build_meta(build_data_context(config))
        self.json_name = self._get_stats_filename()
        self.stats_path = os.path.join(self.run_dir, self.json_name)
        self.meta_stats = self._load_or_build_stats()
        self.stats = self.meta_stats['stats']

    def _build_meta(self, meta):
        normalized_ds_info = {}
        for ds_name in sorted(meta['ds_info']):
            ds_detail = copy.deepcopy(meta['ds_info'][ds_name])
            ds_detail['vars'] = sorted(ds_detail['vars'])
            normalized_ds_info[ds_name] = ds_detail

        return {
            'ds_info': normalized_ds_info,
            'ds_split': {'train': meta['ds_split']['train']},
            'depth': meta['depth'],
        }

    def _get_stats_filename(self):
        vars_in = sum(
            len(detail['vars']) for name, detail in self.meta['ds_info'].items() if name != self.target_ds
        )
        vars_out = len(self.meta['ds_info'][self.target_ds]['vars'])
        return f'norm_stats_{vars_in}in_{vars_out}out_vars.json'

    def _iter_candidate_stats_paths(self):
        yield self.stats_path

        result_path = os.path.dirname(self.run_dir)
        previous_stats_paths = glob(os.path.join(result_path, '**', self.json_name), recursive=True)
        for stats_path in previous_stats_paths:
            if stats_path != self.stats_path:
                yield stats_path

    def _load_or_build_stats(self):
        for stats_path in self._iter_candidate_stats_paths():
            if not os.path.isfile(stats_path):
                continue

            with open(stats_path, 'r', encoding='utf-8') as f:
                previous_meta_stats = json.load(f)

            if self._build_meta(previous_meta_stats['meta']) == self.meta:
                if os.path.abspath(stats_path) != os.path.abspath(self.stats_path):
                    shutil.copyfile(stats_path, self.stats_path)
                print(f'Use matched normalization stats: {stats_path}')
                return previous_meta_stats

        stats = self._compute_stats()
        meta_stats = {'meta': self.meta, 'stats': stats}
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            json.dump(meta_stats, f, ensure_ascii=True, indent=2)
        print(f'No Matching normalization stats, new one saved to: {self.stats_path}')
        return meta_stats

    def _compute_stats(self):
        stats_accumulator = {
            var: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
            for _, ds_detail in self.meta['ds_info'].items()
            for var in ds_detail['vars']
        }

        for _, ds_detail in self.meta['ds_info'].items():
            for nc_path in iter_split_nc_paths(ds_detail['path'], self.meta['ds_split']['train']):
                with xr.open_dataset(nc_path) as ds:
                    for var in ds_detail['vars']:
                        data_array = ds[var]
                        if 'depth' in data_array.dims:
                            data_array = data_array.isel(depth=self.meta['depth'])
                        values = np.asarray(data_array.values, dtype=np.float64)
                        finite_mask = np.isfinite(values)
                        count = int(finite_mask.sum())
                        if count == 0:
                            continue

                        valid_values = values[finite_mask]
                        stats_accumulator[var]['sum'] += valid_values.sum(dtype=np.float64)
                        stats_accumulator[var]['sum_sq'] += np.square(valid_values).sum(dtype=np.float64)
                        stats_accumulator[var]['count'] += count

        stats = {}
        for key, record in stats_accumulator.items():
            count = record['count']
            if count == 0:
                continue

            mean = float(record['sum'] / count)
            variance = max(record['sum_sq'] / count - mean * mean, 0.0)
            std = max(float(np.sqrt(variance)), 1e-6)
            stats[key] = {'mean': mean, 'std': std}

        return stats

    def get_meta_stats(self):
        """Return a copy of the normalization metadata and stats used by this manager."""
        return copy.deepcopy(self.meta_stats)

    def _get_mean_std(self, tensor, var_name):
        if tensor.ndim < 3 or tensor.shape[-3] != 1:
            raise ValueError(f'Expected tensor with shape [...,1,H,W], got {tuple(tensor.shape)}.')
        stats = self.stats[var_name]
        return stats['mean'], stats['std']

    def normalize_tensor(self, tensor, var_name):
        """Normalize one variable tensor."""
        mean, std = self._get_mean_std(tensor, var_name)
        return (tensor - mean) / std

    def denormalize_tensor(self, tensor, var_name):
        """Denormalize one variable tensor."""
        mean, std = self._get_mean_std(tensor, var_name)
        return tensor * std + mean


def get_norm_manager(run_dir, config):
    """Return a normalization manager for the requested context."""
    return DatasetNormalization(run_dir, config)
