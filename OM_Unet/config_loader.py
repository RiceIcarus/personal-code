import copy
import os
from glob import glob
import shutil
import datetime
import tomli as tomllib


CONFIG_PATH = 'config.toml'
CONFIG_NAME = 'config.toml'
RESULT_PATH = 'result'


class ConfigView:
    """Provide top-level attribute access for a plain config dictionary."""

    def __init__(self, data):
        self._data = copy.deepcopy(data)

    def __getattr__(self, name):
        """Called when accessing a non-existent property in this class."""
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self):
        return copy.deepcopy(self._data)


def load_config(path=CONFIG_PATH):
    with open(path, 'rb') as f:
        config_dict = tomllib.load(f)
    return ConfigView(config_dict)


_DEFAULT_CONFIG = load_config()


def __getattr__(name):
    return getattr(_DEFAULT_CONFIG, name)


def load_config_from_dict(data):
    return ConfigView(data)


def build_config_dict(config_obj):
    if hasattr(config_obj, 'to_dict'):
        return config_obj.to_dict()
    if isinstance(config_obj, dict):
        return copy.deepcopy(config_obj)
    return copy.deepcopy(vars(config_obj))


def resolve_run_dir(run_dir=None):
    if run_dir:
        if os.path.isdir(run_dir):
            print(f'Loaded historical run status: {run_dir}')
            return run_dir, False
        else:
            os.makedirs(run_dir, exist_ok=True)
            print(f'run_dir do not exist, create a new run with the given name: {run_dir}')
            return run_dir, True
    else:
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(RESULT_PATH, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f'Create a new run with the current timestamp: {run_dir}')
        return run_dir, True


def load_run_config(run_dir, is_new_run):
    config_path = os.path.join(run_dir, CONFIG_PATH)
    if is_new_run:
        shutil.copyfile(CONFIG_PATH, config_path)
        print(f'Training config saved at: {config_path}')
    return load_config(config_path)


def _format_toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, str):
        return repr(value)
    else:
        return str(value)


def add_config_field(key, value, root_dir=RESULT_PATH, line_no=0):
    updated_paths = []
    config_paths = glob(os.path.join(root_dir, '**', CONFIG_NAME), recursive=True)
    for config_path in config_paths:
        with open(config_path, 'rb') as f:
            config_dict = tomllib.load(f)
            if key in config_dict:
                continue

        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        insert_line = f'{key} = {_format_toml_value(value)}\n'
        insert_idx = max(0, min(len(lines), line_no - 1))
        lines.insert(insert_idx, insert_line)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        updated_paths.append(config_path)
        print(config_path)

    print(f'Updated {len(updated_paths)} config files.')
    return updated_paths


def update_config_field(key, value, root_dir=RESULT_PATH):
    updated_paths = []
    # new_line = f'{key} = {_format_toml_value(value)}'

    config_paths = glob(os.path.join(root_dir, '**', CONFIG_NAME), recursive=True)
    for config_path in config_paths:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        in_top_level = True
        updated = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                in_top_level = False
            if in_top_level and stripped.startswith(f'{key} ='):
                new_lines.append(stripped.replace('label_name', 'target_ds') + '\n')
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            continue

        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        updated_paths.append(config_path)
        print(config_path)

    print(f'Updated {len(updated_paths)} config files.')
    return updated_paths


def remove_config_field(key, root_dir=RESULT_PATH):
    updated_paths = []
    config_paths = glob(os.path.join(root_dir, '**', CONFIG_NAME), recursive=True)
    for config_path in config_paths:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        in_top_level = True
        removed = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                in_top_level = False
            if in_top_level and stripped.startswith(f'{key} ='):
                removed = True
            else:
                new_lines.append(line)

        if not removed:
            continue

        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        updated_paths.append(config_path)
        print(config_path)

    print(f'Updated {len(updated_paths)} config files.')
    return updated_paths


if __name__ == '__main__':
    update_config_field('label_name', 'UNet_3L', root_dir=r'result\Unet layer depth\3L\20260609_102034')
