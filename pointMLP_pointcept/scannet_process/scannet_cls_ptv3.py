# 下载链接
# https://pan.baidu.com/s/1ypna94aDuJDo2uGOWRW1Aw#list/path=%2F&parentPath=%2F?pwd=roq0
# 提取码：roq0

import json
import os
import numpy as np
import open3d as o3d
import pandas as pd
import glob

scannet_path = 'scannet/scans'
out_path = 'scannet_txt'
tsv_path = 'scannet/scannetv2-labels.combined.tsv'

# 40类标签（顺序固定）
CLASSES = ['bag', 'bathtub', 'bed', 'blinds', 'books', 'bookshelf', 'box', 'cabinet', 'ceiling', 'chair', 'clothes',
           'counter', 'curtain', 'desk', 'door', 'dresser', 'floor', 'floor mat', 'lamp', 'mirror', 'night stand',
           'otherfurniture', 'otherprop', 'otherstructure', 'paper', 'person', 'picture', 'pillow', 'refridgerator',
           'shelves', 'shower curtain', 'sink', 'sofa', 'table', 'television', 'toilet', 'towel', 'wall', 'whiteboard',
           'window']
CLASS2ID = {cls: idx for idx, cls in enumerate(CLASSES)}


# 计算法向量
def compute_normals(input_ply, output_ply):
    pcd = o3d.io.read_point_cloud(input_ply)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud(output_ply, pcd)


# 存储法向量
def save_normals(scannet_ROOT):
    for scene in os.listdir(scannet_ROOT):
        input_ply = os.path.join(scannet_ROOT, scene, f'{scene}_vh_clean_2.ply')
        output_ply = os.path.join(scannet_ROOT, scene, f'{scene}_vh_clean_2_normals.ply')
        compute_normals(input_ply, output_ply)
        print(scene + ' normals saved')


def create_labelfolder(out_path, CLASSES):
    for c in CLASSES:
        os.makedirs(os.path.join(out_path, c.replace(' ','')), exist_ok=True)


def load_segments(segs_json):
    """返回每个顶点所属的 segment id，长度 = 顶点数"""
    with open(segs_json) as f:
        return np.array(json.load(f)["segIndices"], dtype=np.int32)


def load_instances(agg_json):
    """返回 [(inst_id, label, [seg_ids]) ...]"""
    with open(agg_json) as f:
        groups = json.load(f)["segGroups"]
    return [(g["id"], g["label"], g["segments"]) for g in groups]


def normalize_points(points):  # 点云归一化
    points -= points.mean(axis=0)  # 去中心化
    scale = np.percentile(np.linalg.norm(points, axis=1), 100)  # 用95%分位数缩放
    points /= (scale + 1e-8)
    return points


def save(out_path, points, label, labellist):
    tsv = pd.read_csv(tsv_path, sep='\t')
    raw2cls = dict(zip(tsv['raw_category'], tsv['nyu40class']))
    label_40 = raw2cls[label]
    label_id = CLASS2ID[label_40]
    fname = f"{label_40}_{labellist[label_id]:04d}.txt"
    labellist[label_id] += 1
    np.savetxt(os.path.join(out_path, str(label_40).replace(' ',''), fname.replace(' ','')), points, fmt='%.6f', delimiter=",")
    # print(f"Saved {fname} {len(points)} points")
    return labellist


def export_instance_xyz_normals(file_path, out_path):
    create_labelfolder(out_path, CLASSES)
    labellist = np.ones(len(CLASSES), dtype=int)
    for scene in os.listdir(file_path):
        ply_path = os.path.join(file_path, scene, f'{scene}_vh_clean_2_normals.ply')
        segs_path = os.path.join(file_path, scene, f'{scene}_vh_clean_2.0.010000.segs.json')
        agg_path = os.path.join(file_path, scene, f'{scene}.aggregation.json')

        normals_pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(normals_pcd.points)
        points = normalize_points(points)
        normals = np.asarray(normals_pcd.normals)
        pointsnormals = np.hstack((points, normals))

        # 2) 读 segment 与 instance
        seg_ids = load_segments(segs_path)
        instances = load_instances(agg_path)

        # 3) 逐实例导出
        for inst_id, label, segs in instances:
            mask = np.isin(seg_ids, segs)
            if not mask.any():
                continue
            points = pointsnormals[mask]
            label = label.replace('/', '_')
            labellist = save(out_path, points, label, labellist)
        print(scene + ' saved')


# 写filelist，train，test和shapename的txt文件
def train_test_txt(out_path):
    f_all = open(os.path.join(out_path, 'filelist.txt'), 'w')
    f_train = open(os.path.join(out_path, 'scannet_train.txt'), 'w')
    f_test = open(os.path.join(out_path, 'scannet_test.txt'), 'w')
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

    with open(os.path.join(out_path, 'scannet_shape_names.txt'), 'w') as f:
        for cls in CLASSES:
            f.write(cls + '\n')


if __name__ == "__main__":
    # save_normals(scannet_path)
    # export_instance_xyz_normals(scannet_path, out_path)
    train_test_txt(out_path)
