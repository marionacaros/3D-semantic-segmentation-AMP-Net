import glob
from utils.utils import fps
import pickle
import numpy as np
import os
from progressbar import progressbar

in_path='/dades/LIDAR/towers_detection/datasets/towers_100x100/*pkl'
out_path='/dades/LIDAR/towers_detection/datasets'
files = glob.glob(in_path)

for point_file in progressbar(files):

    with open(point_file, 'rb') as f:
        pc = pickle.load(f).astype(np.float32)  # [2048, 11]

    fileName = point_file.split('/')[-1].split('.')[0]

    # remove noise
    pc = pc[np.where(pc[:, 3] != 30)]
    pc = pc[np.where(pc[:, 3] != 7)]

    if pc.shape[0] > 8192:
        pc = fps(pc, 8192)
        with open(os.path.join(out_path + '/towers_100x100_fps_8192', fileName) + '.pkl', 'wb') as f:
            pickle.dump(pc, f)

    if pc.shape[0] > 4096:
        pc = fps(pc, 4096)
        with open(os.path.join(out_path + '/towers_100x100_fps_4096', fileName) + '.pkl', 'wb') as f:
            pickle.dump(pc, f)
    else:
        with open(os.path.join(out_path + '/towers_100x100_fps_4096', fileName) + '.pkl', 'wb') as f:
            pickle.dump(pc, f)