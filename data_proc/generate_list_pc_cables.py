from progressbar import progressbar
import glob
import pickle
import os
import numpy as np

path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p/normalized_2048/*pkl'
output_dir = 'prod'

files = glob.glob(path)

for file in progressbar(files):
    with open(file, 'rb') as f:
        # pc = torch.load(f)
        pc = pickle.load(f)

    set_labels = set(pc[:, 3].astype(int))
    if 14 in set_labels and 15 not in set_labels:
        file_name = file.split('/')[-1]
        print(file_name)
        with open(os.path.join(output_dir, 'pc_with_cables.txt'), 'a') as fid:
            fid.write(file_name)
