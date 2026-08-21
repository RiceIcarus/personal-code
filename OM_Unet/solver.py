import csv
import calendar
import datetime
import logging
import os
import random
import time
from glob import glob
import numpy as np
import xarray as xr
import torch
from torch.backends import cudnn
from tqdm import tqdm

import config_loader
from data.data_preprocess import encode_nc
from data_normalization import get_norm_manager
from data_loader import get_dataloader
from evaluation import get_evaluations
from losses import masked_smooth_l1_loss
from network import build_model

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

logging.getLogger('torch._inductor').setLevel(logging.ERROR)


class Solver(object):
    def __init__(self, run_dir=None):
        """Initialize one training run or restore one historical run from run_dir."""
        self.train_loader = None
        self.valid_loader = None
        self.norm_manager = None
        self.criterion = masked_smooth_l1_loss
        self._best_model_loaded = False

        self.run_dir, self.is_new_run = config_loader.resolve_run_dir(run_dir)

        self.config = config_loader.load_run_config(self.run_dir, self.is_new_run)
        self.best_model_path = os.path.join(self.run_dir, f'best_model-{self.config.model_type}.pt')

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.is_cuda = bool(self.device.type == 'cuda')
        self.use_amp = bool(self.config.use_amp and self.is_cuda)
        self.use_compile = self.config.use_compile
        self.use_channels_last = bool(self.config.use_channels_last and self.is_cuda and self.use_compile)
        cudnn.benchmark = True

        self.model = build_model(
            config=self.config,
            device=self.device,
            use_compile=self.use_compile,
            use_channels_last=self.use_channels_last,
        )
        print(f'Run {self.config.model_type} with {self.device}.\n')

    def _build_dataloader(self, mode, **kwargs):
        if self.norm_manager is None:
            self.norm_manager = get_norm_manager(self.run_dir, self.config)
        loader_kwargs = {
            'mode': mode,
            'norm_manager': self.norm_manager,
            'config': self.config,
            'is_cuda': self.is_cuda,
        }
        loader_kwargs.update(kwargs)
        return get_dataloader(**loader_kwargs)

    def _load_best_model(self):
        if not self._best_model_loaded:
            checkpoint = torch.load(self.best_model_path)
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('_orig_mod.') for k in state_dict):
                state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
            model = getattr(self.model, '_orig_mod', self.model)
            model.load_state_dict(state_dict)
            self._best_model_loaded = True
            print(
                f'Best model loaded: {self.best_model_path} | '
                f'Best val loss: {round(checkpoint["val_loss"], 6)} at epoch {checkpoint["epoch"]}'
            )

    def _prepare_inputs(self, tensor):
        memory_format = (
            torch.channels_last if self.use_channels_last and tensor.ndim == 4 else torch.preserve_format
        )
        tensor = tensor.to(self.device, non_blocking=self.is_cuda, memory_format=memory_format)
        return tensor

    def _log_training_progress(self, epoch, train_loss, val_loss, best_val_loss, lr, epoch_time, log_writer):
        log_row = {
            'epoch': epoch,
            'train_loss': ' ' * 8 if train_loss is None else round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'best_val_loss': round(best_val_loss, 6),
            'lr': round(lr, 6),
            'epoch_time': round(epoch_time, 2),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        parts = [
            f'Epoch [{log_row["epoch"]}/{self.config.epochs}]' if key == 'epoch' else f'{key}: {value}'
            for key, value in log_row.items()
        ]
        print(' | '.join(parts[:-1]), '\n')
        log_writer.writerow(log_row)

    def _get_checkpoint_payload(self, epoch, train_loss, val_loss):
        return {
            'run_id': os.path.basename(self.run_dir),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config_loader.build_config_dict(self.config),
            'norm_stats': self.norm_manager.get_meta_stats(),
            'runtime': {
                'device': str(self.device),
                'use_amp': self.use_amp,
                'use_channels_last': self.use_channels_last,
                'use_compile': self.use_compile,
            },
            'model_state_dict': getattr(self.model, '_orig_mod', self.model).state_dict(),
        }

    def _validate_and_track_best(self, epoch, train_loss, best_val_loss, best_epoch):
        avg_val_loss = self.valid(epoch=epoch)

        if avg_val_loss < (best_val_loss - self.config.scheduler_threshold):
            best_val_loss = avg_val_loss
            best_epoch = epoch
            checkpoint = self._get_checkpoint_payload(epoch, train_loss, avg_val_loss)
            torch.save(checkpoint, self.best_model_path)
            print(
                f'Best model saved: {self.best_model_path} | '
                f'Best val loss: {round(avg_val_loss, 6)} at epoch {epoch}'
            )

        return avg_val_loss, best_val_loss, best_epoch

    def _train_setup(self):
        self._best_model_loaded = False

        # set seed
        if self.config.seed:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        # build dataloader
        if self.train_loader is None:
            self.train_loader = self._build_dataloader('train')
        if self.valid_loader is None:
            self.valid_loader = self._build_dataloader('valid')

    def train(self):
        if not self.is_new_run:
            print(f'Run directory already exists, train() is disabled for historical run: {self.run_dir}')
            return

        self._train_setup()

        # save logs
        train_log_path = os.path.join(self.run_dir, f'train_log-{self.config.model_type}.csv')
        log_fields = ['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'lr', 'epoch_time', 'timestamp']
        with open(train_log_path, 'w', newline='', encoding='utf-8') as log_f:
            log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
            log_writer.writeheader()
        print(f'Training log saved at: {train_log_path}\n')

        # build optimizer, scheduler and scaler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            threshold=self.config.scheduler_threshold,
            threshold_mode='abs',
            min_lr=1e-6,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        best_val_loss = float('inf')
        best_epoch = 0
        init_lr = optimizer.param_groups[0]['lr']

        # valid untrained model once
        val_start_time = time.time()
        init_val_loss, best_val_loss, best_epoch = self._validate_and_track_best(
            epoch=0, train_loss=None, best_val_loss=best_val_loss, best_epoch=best_epoch
        )
        init_val_time = time.time() - val_start_time

        with open(train_log_path, 'a', newline='', encoding='utf-8') as log_f:
            log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
            self._log_training_progress(
                epoch=0,
                train_loss=None,
                val_loss=init_val_loss,
                best_val_loss=best_val_loss,
                lr=init_lr,
                epoch_time=init_val_time,
                log_writer=log_writer,
            )

        # start training
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()

            if self.train_loader and hasattr(self.train_loader, 'sampler'):
                sampler = self.train_loader.sampler
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_len = 0

            desc = f'Epoch [{epoch + 1}/{self.config.epochs}] [Train]'
            train_bar = tqdm(self.train_loader, desc=desc, leave=False)
            for _, (inputs, target) in enumerate(train_bar):
                inputs = self._prepare_inputs(inputs)
                target = target.to(self.device, non_blocking=self.is_cuda)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    pred = self.model(inputs)
                    loss = self.criterion(pred, target)

                optimizer.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()  # backpropagation(scaler will downgraded when use_amp=False)
                scaler.unscale_(optimizer)  # unscale
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  #  gradient clipping
                scaler.step(optimizer)  # step optimizer
                scaler.update()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_len += inputs.size(0)
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = epoch_loss / epoch_len if epoch_len > 0 else float('nan')

            lr = optimizer.param_groups[0]['lr']
            avg_val_loss, best_val_loss, best_epoch = self._validate_and_track_best(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )
            scheduler.step(avg_val_loss)

            epoch_time = time.time() - epoch_start_time
            with open(train_log_path, 'a', newline='', encoding='utf-8') as log_f:
                log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
                self._log_training_progress(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    best_val_loss=best_val_loss,
                    lr=lr,
                    epoch_time=epoch_time,
                    log_writer=log_writer,
                )

            if lr <= self.config.initial_lr * self.config.scheduler_factor**self.config.lr_scheduler_times:
                print(
                    f'Early stop at epoch {epoch + 1} | '
                    f'Best val loss: {round(best_val_loss, 6)} at epoch {best_epoch}\n'
                )
                break

        print(f'Training complete! Complete training log has saved at: {train_log_path}\n')

    def _valid_collect_evaluations(self, pred, target, eval_count, eval_sums):
        target_vars = self.config.ds_info[self.config.target_ds]['vars']
        for channel_idx, target_var in enumerate(target_vars):
            pred_channel = self.norm_manager.denormalize_tensor(
                pred[:, channel_idx : channel_idx + 1], target_var
            )
            target_channel = self.norm_manager.denormalize_tensor(
                target[:, channel_idx : channel_idx + 1], target_var
            )
            evals = get_evaluations(pred_channel.cpu(), target_channel.cpu())
            for eval_name, eval_value in evals.items():
                if np.isfinite(eval_value):
                    eval_sums[target_var][eval_name] += eval_value
            eval_count[target_var] += 1

    def _valid_show_evaluations(self, eval_count, eval_sums):
        eval_log_path = os.path.join(self.run_dir, f'eval_log-{self.config.model_type}.txt')
        with open(eval_log_path, 'a', encoding='utf-8') as eval_f:
            for target_var, var_eval_count in eval_count.items():
                if var_eval_count == 0:
                    continue
                parts = [f'{target_var} evaluations']
                for eval_name, eval_sum in eval_sums[target_var].items():
                    parts.append(f'{eval_name}: {eval_sum / var_eval_count:.4f}')
                line = ' | '.join(parts)
                print(line, '\n')
                eval_f.write(line + '\n')
            eval_f.write('\n')

    def valid(self, epoch: int | None = None, show_evaluations=False):
        if epoch is None:
            self._load_best_model()
            desc = 'Validation'
        else:
            desc = f'Epoch [{epoch}/{self.config.epochs}] [Val]'

        if self.valid_loader is None:
            self.valid_loader = self._build_dataloader('valid')

        self.model.eval()
        val_loss = 0.0
        val_length = 0

        # evaluations relevants
        target_vars = self.config.ds_info[self.config.target_ds]['vars']
        eval_count = {target_var: 0 for target_var in target_vars}
        eval_sums = {
            target_var: {'bias': 0.0, 'mae': 0.0, 'rmse': 0.0, 'corr': 0.0} for target_var in target_vars
        }

        with torch.no_grad():
            val_bar = tqdm(self.valid_loader, desc=desc, leave=False)
            for _, (inputs, target) in enumerate(val_bar):
                inputs = self._prepare_inputs(inputs)
                target = target.to(self.device, non_blocking=self.is_cuda)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=False):
                    pred = self.model(inputs)
                    loss = self.criterion(pred, target)

                val_loss += loss.item() * inputs.size(0)
                val_length += inputs.size(0)
                val_bar.set_postfix(loss=loss.item())

                if show_evaluations:
                    self._valid_collect_evaluations(pred, target, eval_count, eval_sums)

        avg_val_loss = val_loss / val_length if val_length > 0 else float('nan')
        if epoch is None:
            print(f'\nValidation complete | Avg Val Loss: {avg_val_loss:.4f}')
        if show_evaluations:
            self._valid_show_evaluations(eval_count, eval_sums)

        return avg_val_loss

    def test(self, year: int, month: int, day: int):
        self._load_best_model()

        self.model.eval()
        test_loader = self._build_dataloader('test', year=year, month=month, day=day)
        tensor = next(iter(test_loader))

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=False):
                pred = self.model(self._prepare_inputs(tensor))

        pred = pred[0]
        target_vars = self.config.ds_info[self.config.target_ds]['vars']
        time_val = np.datetime64(f'{year}-{month:02d}-{day:02d}')
        land_mask = test_loader.dataset.get_land_mask()
        data_vars = {}
        for channel_idx, target_var in enumerate(target_vars):
            pred_channel = self.norm_manager.denormalize_tensor(
                pred[channel_idx : channel_idx + 1], target_var
            )
            pred_np = pred_channel.cpu().numpy()
            pred_np[0, ~land_mask['land_mask']] = np.nan

            data_vars[target_var] = (
                ['time', 'depth', 'latitude', 'longitude'],
                pred_np[0][np.newaxis, np.newaxis, :, :],
            )

        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': [time_val],
                'depth': land_mask['depth'],
                'latitude': land_mask['latitude'],
                'longitude': land_mask['longitude'],
            },
            attrs={'title': 'Sea Water Prediction'},
        )
        for target_var, attrs in land_mask['target_attrs'].items():
            ds_out[target_var].attrs = attrs

        nc_path = os.path.join(self.run_dir, f'pred_{self.config.target_ds}_{year}{month:02d}{day:02d}.nc')
        encode_nc(ds_out, nc_path)
        print(f'Prediction saved: {nc_path}')
        return nc_path

    def eval_year(self, year):
        """Compute daily evaluation metrics for a full year and save to CSV."""
        self._load_best_model()
        self.model.eval()
        target_var = self.config.ds_info[self.config.target_ds]['vars'][0]
        glory_dir = self.config.ds_info[self.config.target_ds]['path']

        csv_path = os.path.join(self.run_dir, f'eval_year_{year}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'bias', 'mae', 'rmse', 'corr'])

            for month in range(1, 13):
                _, days_in_month = calendar.monthrange(year, month)

                glory_path = glob(os.path.join(glory_dir, str(year), f'*{year}{month:02d}.nc'))[0]
                with xr.open_dataset(glory_path) as ds:
                    ref_data = ds[target_var].sel(depth=self.config.depth, method='nearest').values

                land_mask = None

                for day in range(1, days_in_month + 1):
                    test_loader = self._build_dataloader('test', year=year, month=month, day=day)
                    tensor = next(iter(test_loader))

                    with torch.no_grad():
                        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=False):
                            pred = self.model(self._prepare_inputs(tensor))

                    pred = pred[0]
                    pred_channel = self.norm_manager.denormalize_tensor(pred[0:1], target_var)
                    pred_np = pred_channel.cpu().numpy()

                    if land_mask is None:
                        land_mask = test_loader.dataset.get_land_mask()['land_mask']

                    ref = ref_data[day - 1]
                    ref[~land_mask] = np.nan
                    pred_np[0, ~land_mask] = np.nan

                    evals = get_evaluations(pred_np[0], ref)
                    date_str = f'{year}-{month:02d}-{day:02d}'
                    writer.writerow(
                        [
                            date_str,
                            f'{evals["bias"]:.6f}',
                            f'{evals["mae"]:.6f}',
                            f'{evals["rmse"]:.6f}',
                            f'{evals["corr"]:.6f}',
                        ]
                    )
                    print(f'{date_str} evaluation complete')
