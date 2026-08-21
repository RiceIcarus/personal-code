import os
import random
from glob import glob

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Sampler

from data_normalization import build_data_context, iter_split_nc_paths

AUGMENTATION_NOISE_STD = 0.01
VALID_MODES = {'train', 'valid', 'test'}


class NCDataset(Dataset):
    """Read train or validation samples from aligned monthly NetCDF files."""

    def __init__(self, mode, norm_manager, config, augmentation_prob=0.0):
        if mode not in {'train', 'valid'}:
            raise ValueError("NCDataset only supports 'train' and 'valid' modes.")
        self.mode = mode
        self.data_context = build_data_context(config)
        self.ds_info = self.data_context['ds_info']
        self.target_ds = config.target_ds
        self.depth = self.data_context['depth']
        self.norm_manager = norm_manager
        self.augmentation_prob = augmentation_prob

        self.day_index = []
        self.file_day_ranges = []
        self._ncpaths = {}
        self._file_cache = {}

        self._index_files()

    def _index_files(self):
        """Index aligned monthly files and build the day-level sample map."""
        file_day_records = {}

        for ds_name, ds_detail in self.ds_info.items():
            ncpaths = list(iter_split_nc_paths(ds_detail['path'], self.data_context['ds_split'][self.mode]))
            self._ncpaths[ds_name] = ncpaths

            day_counts = []
            for nc_path in ncpaths:
                with xr.open_dataset(nc_path) as ds:
                    day_counts.append(ds.sizes['time'])
            file_day_records[ds_name] = day_counts

        reference_days = file_day_records[self.target_ds]
        for ds_name, day_counts in file_day_records.items():
            if ds_name != self.target_ds and day_counts != reference_days:
                print(day_counts)
                print(reference_days)
                raise ValueError(f'The num of days in {ds_name} file do not match with {self.target_ds}!')

        num_files = len(reference_days)
        self._file_cache = {name: [None] * num_files for name in self.ds_info}

        end_idx = 0
        for file_idx, n_days in enumerate(reference_days):
            start_idx = end_idx
            for day in range(n_days):
                self.day_index.append((file_idx, day))
            end_idx += n_days
            self.file_day_ranges.append((start_idx, end_idx))

        print(f'{self.mode} set loaded a total of {len(self.day_index)} days of data')

    def _read_variable(self, ds_name, var, file_idx, day_idx):
        ds_list = self._file_cache[ds_name]
        if ds_list[file_idx] is None:
            ds_list[file_idx] = xr.open_dataset(self._ncpaths[ds_name][file_idx])
        ds = ds_list[file_idx]
        indexer = {'time': day_idx}
        if ds_name == self.target_ds:
            indexer['depth'] = self.depth
        data = ds[var].isel(**indexer).values
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        return data

    def _augmentation(self, tensor):
        if self.mode == 'train' and random.random() < self.augmentation_prob:
            tensor += torch.randn_like(tensor) * AUGMENTATION_NOISE_STD
        return tensor

    def _load_inputs_tensor(self, file_idx, day_idx):
        input_tensors = []
        for ds_name, ds_detail in self.ds_info.items():
            if ds_name == self.target_ds:
                continue
            for var in ds_detail['vars']:
                data = self._read_variable(ds_name, var, file_idx, day_idx)
                tensor = torch.from_numpy(data).float()
                tensor = self.norm_manager.normalize_tensor(tensor, var)
                input_tensors.append(tensor)

        tensor = torch.cat(input_tensors, dim=0)
        tensor = self._augmentation(tensor)
        tensor = torch.nan_to_num(tensor)
        return tensor

    def _load_target_tensor(self, file_idx, day_idx):
        target_tensors = []
        for var in self.ds_info[self.target_ds]['vars']:
            data = self._read_variable(self.target_ds, var, file_idx, day_idx)
            tensor = torch.from_numpy(data).float()
            tensor = self.norm_manager.normalize_tensor(tensor, var)
            target_tensors.append(tensor)

        tensor = torch.cat(target_tensors, dim=0)
        return tensor

    def __len__(self):
        return len(self.day_index)

    def __getitem__(self, idx):
        """Get the data for the idx-th day."""
        file_idx, day_idx = self.day_index[idx]
        inputs_tensor = self._load_inputs_tensor(file_idx, day_idx)
        target_tensor = self._load_target_tensor(file_idx, day_idx)
        return inputs_tensor, target_tensor


class OneDayDataset(Dataset):
    """Load one normalized inference sample for a specific date."""

    def __init__(self, norm_manager, config, year, month, day):
        self.data_context = build_data_context(config)
        self.ds_info = self.data_context['ds_info']
        self.target_ds = config.target_ds
        self.norm_manager = norm_manager
        self.year, self.month, self.day = int(year), int(month), int(day)
        self._land_mask = None

    def _load_inputs(self):
        input_tensors = []
        for ds_name, ds_detail in self.ds_info.items():
            if ds_name == self.target_ds:
                continue
            file_path = glob(
                os.path.join(ds_detail['path'], str(self.year), f'*{self.year}{self.month:02d}.nc')
            )[0]
            for var in ds_detail['vars']:
                with xr.open_dataset(file_path) as ds:
                    data = ds[var].isel(time=self.day - 1).values
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                tensor = torch.from_numpy(data).float()
                tensor = self.norm_manager.normalize_tensor(tensor, var)
                input_tensors.append(tensor)

        tensor = torch.cat(input_tensors, dim=0)
        tensor = torch.nan_to_num(tensor)
        return tensor

    def get_land_mask(self):
        if self._land_mask is not None:
            return self._land_mask

        target_path = self.ds_info[self.target_ds]['path']
        reference_path = glob(os.path.join(target_path, '*', '*.nc'))[-1]
        reference_var = self.ds_info[self.target_ds]['vars'][0]
        with xr.open_dataset(reference_path) as ds_ref:
            lat = ds_ref['latitude'].values
            lon = ds_ref['longitude'].values
            depth_val = ds_ref['depth'].isel(depth=self.data_context['depth']).values
            land_mask = np.isfinite(
                ds_ref[reference_var].isel(time=0, depth=self.data_context['depth']).values
            )
            target_attrs = {
                target_var: dict(ds_ref[target_var].attrs)
                for target_var in self.ds_info[self.target_ds]['vars']
            }

        self._land_mask = {
            'depth': [depth_val],
            'latitude': lat,
            'longitude': lon,
            'land_mask': land_mask,
            'target_attrs': target_attrs,
        }
        return self._land_mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('SingleDayDataset contains only one sample.')
        return self._load_inputs()


class TrainSampler(Sampler):
    """Sampler that supports sequential, global, and block shuffling."""

    def __init__(self, dataset, shuffle_mode='sequential', block_size=0, seed=0):
        self.dataset = dataset
        self.block_size = int(block_size)
        self.seed = seed
        self.epoch = 0
        self.blocks = self._build_blocks()

        self.shuffle_mode = shuffle_mode
        valid_modes = {'sequential', 'global', 'block'}
        if self.shuffle_mode not in valid_modes:
            raise ValueError(f'Please select shuffle mode: {valid_modes}')

    def _build_blocks(self):
        if self.block_size < 0:
            raise ValueError('train_block_size must be >= 0')

        blocks = []
        for start_idx, end_idx in self.dataset.file_day_ranges:
            if self.block_size == 0:
                blocks.append(list(range(start_idx, end_idx)))
                continue

            for block_start in range(start_idx, end_idx, self.block_size):
                block_end = min(block_start + self.block_size, end_idx)
                blocks.append(list(range(block_start, block_end)))
        return blocks

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle_mode == 'sequential':
            return iter(range(len(self.dataset)))

        rng = random.Random(self.seed + self.epoch)

        if self.shuffle_mode == 'global':
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)
            return iter(indices)

        if self.shuffle_mode == 'block':
            blocks = [block[:] for block in self.blocks]
            rng.shuffle(blocks)

            indices = []
            for block in blocks:
                rng.shuffle(block)
                indices.extend(block)
            return iter(indices)

    def __len__(self):
        return len(self.dataset)


def get_dataloader(mode, norm_manager, config, is_cuda=False, year=None, month=None, day=None):
    if mode not in VALID_MODES:
        raise ValueError(f'Please select mode: {VALID_MODES}')

    batch_size = config.batch_size
    sampler = None
    if mode == 'train':
        dataset = NCDataset(mode, norm_manager, config, augmentation_prob=config.augmentation_prob)
        sampler = TrainSampler(dataset, shuffle_mode=config.shuffle_mode, seed=config.seed)
    if mode == 'valid':
        dataset = NCDataset(mode, norm_manager, config)
    if mode == 'test':
        if year is None or month is None or day is None:
            raise ValueError('test mode requires year, month, day.')
        dataset = OneDayDataset(norm_manager, config, year, month, day)
        batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=is_cuda,
        persistent_workers=bool(config.num_workers),
        prefetch_factor=max(2, int(batch_size / 8)) if config.num_workers else None,
    )
    return dataloader


if __name__ == '__main__':
    import config_loader

    s = OneDayDataset(None, config_loader, 2001, 1, 1)
    s.get_land_mask()
