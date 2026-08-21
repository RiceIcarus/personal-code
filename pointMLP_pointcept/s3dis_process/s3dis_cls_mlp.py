# 下载链接
# https://stanford.redivis.com/datasets/9q3m-9w5pa1a2h

import os
import numpy as np
import h5py
from glob import glob

# 设置路径
S3dis_path = 'Stanford3dDataset_v1.2'  # S3dis路径
out_path = 's3dis_classification_14_1024'  # 输出文件夹
num_points = 1024  # 采集点数目

# 13+1类标签（顺序固定）（多了一个stairs）
CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter', 'stairs']
CLASS2ID = {cls: idx for idx, cls in enumerate(CLASSES)}


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


def s3dis_classification():
    for area in os.listdir(S3dis_path):
        all_points = []
        all_labels = []

        txt_files = glob(os.path.join(S3dis_path, area, '*', 'Annotations', '*.txt'))
        for sample in txt_files:
            labelname = sample.split('\\')[-1].split('_')[0]
            label = [CLASS2ID[labelname]]
            data = np.loadtxt(sample)  # 原始点云
            pointcloud = sample_point(data, num_points)  # 采样后点云(1024, 3)
            pointcloud = normalize_point(pointcloud)  # 归一化后点云

            all_points.append(pointcloud)
            all_labels.append(label)

        all_points = np.array(all_points, dtype=np.float32)  # (N, 1024, 3)
        all_labels = np.array(all_labels, dtype=np.int64)  # (N,)

        # 保存为 h5
        h5_path = os.path.join(out_path, f's3dis_cla_{area}.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=all_points)
            f.create_dataset('label', data=all_labels)

        print(f"Saved {len(all_labels)} samples to {h5_path}")


if __name__ == '__main__':
    s3dis_classification()
