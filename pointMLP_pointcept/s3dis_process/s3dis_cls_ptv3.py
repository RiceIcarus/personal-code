# 下载链接
# https://stanford.redivis.com/datasets/9q3m-9w5pa1a2h

import os
import numpy as np
import open3d as o3d
import glob
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple

# 设置路径
s3dis_path = 'Stanford3dDataset_v1.2'  # S3dis路径
out_path = 's3dis_txt'  # 输出文件夹

# 13+1类标签（顺序固定）（多了一个stairs）
CLASSES = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'stairs',
           'table', 'wall', 'window']
CLASS2ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def create_labelfolder(out_path, CLASSES):
    for c in CLASSES:
        os.makedirs(os.path.join(out_path, c), exist_ok=True)


# 计算法向量
def compute_normals(xyz: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=25))
    pcd.orient_normals_consistent_tangent_plane(100)
    return np.asarray(pcd.normals)


# 点云归一化
def normalize_point(xyz: np.ndarray) -> np.ndarray:
    xyz -= xyz.mean(axis=0)  # 去中心化
    scale = np.percentile(np.linalg.norm(xyz, axis=1), 100)
    xyz /= (scale + 1e-8)
    return xyz


def s3dis_cla_ptv3_txt(args: Tuple[int, str]) -> Tuple[int, str]:
    try:

        num, file_path = args
        labelname = file_path.split('\\')[-1].split('_')[0]

        data = np.loadtxt(file_path)  # 原始点云
        xyz = data[:, :3].astype(np.float32)
        normals = compute_normals(xyz)
        xyz = normalize_point(xyz)  # 归一化后点云
        xyznormals = np.hstack((xyz, normals))

        fname = f"{labelname}_{num:04d}.txt"
        out_file = os.path.join(out_path, labelname, fname)
        np.savetxt(out_file, xyznormals, fmt='%.6f', delimiter=",")
        # print(fname + ' saved')
    except Exception as e:
        print(f'ERROR in {args}: {e}')
        raise


def main():
    create_labelfolder(out_path, CLASSES)
    # labellist = np.ones(len(CLASSES), dtype=int)
    labellist =[91,108,494,322,1184,3198,200,450,235,46,16,378,1300,137]
    labellist = [x + 1 for x in labellist]
    for i in range(5, 6):
        area = os.listdir(s3dis_path)[i]
        tasks = []
        txt_files = glob.glob(os.path.join(s3dis_path, area, '*', 'Annotations', '*.txt'))
        for sample in txt_files:
            labelname = sample.split('\\')[-1].split('_')[0]
            labelid = CLASS2ID[labelname]
            num = labellist[labelid]
            tasks.append((num, sample))
            labellist[labelid] += 1
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in tqdm(pool.imap_unordered(s3dis_cla_ptv3_txt, tasks), total=len(tasks), desc='Processing'):
                pass
        print(area + ' saved')
    print(labellist)


# 写filelist，train，test和shapename的txt文件
def train_test_txt(out_path):
    f_all = open(os.path.join(out_path, 'filelist.txt'), 'w')
    f_train = open(os.path.join(out_path, 's3dis_train.txt'), 'w')
    f_test = open(os.path.join(out_path, 's3dis_test.txt'), 'w')
    for cla in os.listdir(out_path):
        if os.path.isdir(os.path.join(out_path, cla)):
            txt_files = glob.glob(os.path.join(out_path, cla, '*.txt'))
            for txt_file in txt_files:
                f_all.write(cla + '/' + txt_file.split('\\')[-1] + '\n')
            trainl = round(len(txt_files) * 0.8)
            for i in range(trainl):
                f_train.write(txt_files[i].split('\\')[-1].split('.')[0] + '\n')
            for i in range(trainl, len(txt_files)):
                f_test.write(txt_files[i].split('\\')[-1].split('.')[0] + '\n')
    f_all.close()
    f_train.close()
    f_test.close()

    with open(os.path.join(out_path, 's3dis_shape_names.txt'), 'w') as f:
        for cls in CLASSES:
            f.write(cls + '\n')


if __name__ == '__main__':
    # main()
    train_test_txt(out_path)
