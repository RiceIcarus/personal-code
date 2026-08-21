from glob import glob
import os
import torch
import numpy as np

import config_loader


def rename_file(path):
    filepaths = glob(os.path.join(path, '**', '*.csv'), recursive=True)
    for filepath in filepaths:
        newfilepath = filepath.replace('CSAtt', 'CBAM')
        os.rename(filepath, newfilepath)


def rename_pt(key, value, root_dir=config_loader.RESULT_PATH):
    files = glob(os.path.join(root_dir, '**', '*.pt'), recursive=True)
    for f in files:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config')
        if key in cfg:
            cfg[key] = value
            torch.save(ckpt, f)
            print(f'{f}:    {key} = {value}')


def remove_nc():
    run_dirs = glob(os.path.join(config_loader.RESULT_PATH, '**', '????????_??????'), recursive=True)
    for run_dir in run_dirs:
        nc_files = glob(os.path.join(run_dir, '*.nc'))
        for f in nc_files:
            os.remove(f)
            print(f'  removed: {f}')


if __name__ == '__main__':
    pass
