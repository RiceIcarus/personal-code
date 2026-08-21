# 下载链接
# https://pan.baidu.com/s/1ypna94aDuJDo2uGOWRW1Aw#list/path=%2F&parentPath=%2F?pwd=roq0
# 提取码：roq0

import json
import os
import numpy as np
import h5py
import glob
import pandas as pd
from pathlib import Path
import trimesh

# 设置路径
scannet_path = 'scannet/scans'
scannet_txt_path = 'scannet_txt/scans'  # scannet路径
scannet_h5_path = 'scannet_h5/scans'  # 输出文件夹
tsv_path = 'scannet/scannetv2-labels.combined.tsv'
num_points = 1024  # 采集点数目

# 40类标签（顺序固定）
CLASSES = ['towel', 'bed', 'lamp', 'television', 'sofa', 'wall', 'chair', 'otherprop', 'clothes', 'mirror', 'books',
           'door', 'picture', 'shower curtain', 'dresser', 'curtain', 'ceiling', 'floor mat', 'counter', 'desk',
           'otherstructure', 'sink', 'paper', 'floor', 'whiteboard', 'bookshelf', 'night stand', 'toilet', 'window',
           'blinds', 'otherfurniture', 'person', 'box', 'table', 'cabinet', 'refridgerator', 'shelves', 'pillow',
           'bathtub', 'bag']
CLASS2ID = {cls: idx for idx, cls in enumerate(CLASSES)}


# 返回每个顶点所属的 segment id，长度 = 顶点数
def load_segments(segs_json):
    with open(segs_json) as f:
        return np.array(json.load(f)["segIndices"], dtype=np.int32)


#  返回 [(inst_id, label, [seg_ids]) ...]
def load_instances(agg_json):
    with open(agg_json) as f:
        groups = json.load(f)["segGroups"]
    return [(g["id"], g["label"], g["segments"]) for g in groups]


def export_instance_xyz(file_path, out_path):
    for scene in os.listdir(file_path):
        out_dir = os.path.join(out_path, scene)
        ply_path = os.path.join(file_path, scene, f'{scene}_vh_clean_2.ply')
        segs_path = os.path.join(file_path, scene, f'{scene}_vh_clean_2.0.010000.segs.json')
        agg_path = os.path.join(file_path, scene, f'{scene}.aggregation.json')

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 读网格顶点
        mesh = trimesh.load(str(ply_path), process=False)
        xyz = mesh.vertices  # (V,3) numpy.ndarray

        # 2) 读 segment 与 instance
        seg_ids = load_segments(segs_path)
        instances = load_instances(agg_path)

        # 3) 逐实例导出
        for inst_id, label, segs in instances:
            mask = np.isin(seg_ids, segs)
            if not mask.any():
                continue
            points = xyz[mask]

            safe_label = label.replace('/', '_')
            fname = f"{scene}_{safe_label}_{inst_id:03d}.txt"
            np.savetxt(out_dir / fname, points, fmt='%.6f')
            print(f"Saved {fname}  {len(points)} points")


def sample_point(data, num_points):  # 将点云采样为统一尺寸
    xyz = data[:, :3]  # 只用坐标数据
    N = xyz.shape[0]  # 总点数
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    return xyz[idx]


def normalize_point(data):  # 点云归一化
    data -= data.mean(axis=0)  # 去中心化
    scale = np.percentile(np.linalg.norm(data, axis=1), 95)  # 用95%分位数缩放
    data /= (scale + 1e-8)
    return data


def scannet_classification():
    all_points = []
    all_labels = []
    tsv = pd.read_csv(tsv_path, sep='\t')
    raw2cls = dict(zip(tsv['raw_category'], tsv['nyu40class']))

    for scene in os.listdir(scannet_txt_path):
        txt_files = glob.glob(os.path.join(scannet_txt_path, scene, '*.txt'))
        for sample in txt_files:
            labelname = sample.split('_')[-2]
            labelname_40 = raw2cls[labelname]
            label = [CLASS2ID[labelname_40]]
            data = np.loadtxt(sample)  # 原始点云
            pointcloud = sample_point(data, num_points)  # 采样后点云(1024, 3)
            pointcloud = normalize_point(pointcloud)  # 归一化后点云

            all_points.append(pointcloud)
            all_labels.append(label)

    all_points = np.array(all_points, dtype=np.float32)  # (N, 1024, 3)
    all_labels = np.array(all_labels, dtype=np.int64)  # (N,)
    return all_points, all_labels


def save_h5():
    all_points, all_labels = scannet_classification()
    chunk_size = 4096  # 每个h5文件存储点云数目
    num_chunks = int(np.ceil(all_points.shape[0] / chunk_size))

    for k in range(num_chunks):
        start = k * chunk_size
        end = min(start + chunk_size, all_points.shape[0])

        points_chunk = all_points[start:end]  # (N, 1024, 3)
        labels_chunk = all_labels[start:end]  # (N, 1)

        fpath = os.path.join(scannet_h5_path, f'scannet_40_1024_{k:02d}.h5')
        with h5py.File(fpath, 'w') as f:
            f.create_dataset('data', data=points_chunk, dtype=np.float32)
            f.create_dataset('labels', data=labels_chunk, dtype=np.int64)
            print(f'saved file {k}')


if __name__ == "__main__":
    export_instance_xyz(scannet_path, scannet_h5_path)
