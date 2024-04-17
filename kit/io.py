import multiprocessing

import numpy as np
import pandas as pd

from tqdm import tqdm
from pyntcloud import PyntCloud

import torch

def read_point_cloud(filepath):
    pc = PyntCloud.from_file(filepath)
    pc = np.array(pc.points)[:, :3]
    return pc

def read_point_clouds(file_path_list):
    print('loading point clouds...')
    with multiprocessing.Pool(4) as p:
        pcs = list(tqdm(p.imap(read_point_cloud, file_path_list, 32), total=len(file_path_list)))
    return pcs

def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)

