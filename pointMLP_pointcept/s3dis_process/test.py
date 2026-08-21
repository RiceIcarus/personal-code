import os
import glob
import numpy as np

import heapq

import multiprocessing as mp
s3dis_path = 'Stanford3dDataset_v1.2'  # S3dis路径

labellist =[91,108,494,322,1184,3198,200,450,235,46,16,378,1300,137]
labellist = [x + 1 for x in labellist]
print(labellist)