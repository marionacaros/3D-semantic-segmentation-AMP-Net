import argparse
import hashlib
import logging
import random
from utils.utils import *
import pickle
import laspy
import time
import multiprocessing

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def remove_ground_and_outliers(file, max_z=100.0, max_intensity=5000, n_points=1024,
                               dataset='', TH_1=3, TH_2=8):
    """
    1- Remove certain labeled points (by Terrasolid) to reduce noise and number of points
    2- Add NIR from dictionary
    3- Remove outliers defined as points > max_z and points < 0
    4- Normalize data
    5- Remove terrain points (up to n_points points in point cloud)

    It stores pickle files with preprocessed data
    """
    counters = {
        'total_count': 0,
        'need_ground': 0,
        'keep_ground': 0,
        'count_sample3': 0,
        'count_sample8': 0,
        'sample_all': 0,
        'missed': 0
    }
    name='pc_'
    fileName = file.split('/')[-1].split('.')[0]
    data_f = laspy.read(file)

    # Remove all categories of ground points
    data_f.points = data_f.points[np.where(data_f.classification != 2)]
    data_f.points = data_f.points[np.where(data_f.classification != 7)]
    data_f.points = data_f.points[np.where(data_f.classification != 8)]
    data_f.points = data_f.points[np.where(data_f.classification != 13)]
    data_f.points = data_f.points[np.where(data_f.classification != 24)]

    # Remove sensor noise
    data_f.points = data_f.points[np.where(data_f.classification != 30)]
    try:
        # Remove outliers (points above max_z)
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround <= max_z)]
        # Remove points z < 0
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround >= 0)]

        # check file is not empty
        if len(data_f.x) > 0:

            if dataset != 'BDN':
                # get NIR
                nir_arr = []
                with open(file.replace(".las", "") + '_NIR.pkl', 'rb') as f:
                    nir_dict = pickle.load(f)

                for x, y, z in zip(data_f.x, data_f.y, data_f.z):
                    mystring = str(int(x)) + '_' + str(int(y)) + '_' + str(int(z))
                    hash_object = hashlib.md5(mystring.encode())
                    nir_arr.append(nir_dict[hash_object.hexdigest()])

                # NDVI
                nir_arr = np.array(nir_arr)
                ndvi_arr = (nir_arr - data_f.red) / (nir_arr + data_f.red)  # range [-1, 1]
            else:
                nir_arr = np.zeros(len(data_f.x))
                ndvi_arr = np.zeros(len(data_f.x))

            pc = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround,
                            data_f.classification,  # 3
                            data_f.intensity / max_intensity,  # 4
                            data_f.red / 65536.0,  # 5
                            data_f.green / 65536.0,  # 6
                            data_f.blue / 65536.0,  # 7
                            nir_arr / 65535.0,  # 8
                            ndvi_arr,  # 9
                            data_f.x,  # 10
                            data_f.y,  # 11
                            data_f.z))  # 12

            # ----------------------------------------- NORMALIZATION -----------------------------------------
            pc = pc.transpose()

            if not (pc[:, 0].max() - pc[:, 0].min() == 0 or pc[:, 1].max() - pc[:, 1].min() == 0):
                # normalize axes between -1 and 1
                pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
                pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
                pc[:, 2] = pc[:, 2] / max_z  # (HAG)

                # Remove points z < 0
                pc = pc[pc[:, 2] >= 0]
                # make sure intensity and NIR is in range (0,1)
                pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)
                pc[:, 8] = np.clip(pc[:, 8], 0.0, 1.0)
                # normalize NDVI
                pc[:, 9] = (pc[:, 9] + 1) / 2
                pc[:, 9] = np.clip(pc[:, 9], 0.0, 1.0)

                # store files with n_points as minimum
                if pc.shape[0] >= n_points:
                    counters['total_count'] += 1

                    # rename
                    unique, counts = np.unique(pc[:, 3].astype(int), return_counts=True)
                    dic_counts = dict(zip(unique, counts))

                    if 15 in dic_counts.keys():
                        if dic_counts[15] > 10:
                            name = 'tower_'
                    elif 14 in dic_counts.keys():
                        if dic_counts[14] > 10:
                            name = 'powerline_'

                    fileName = fileName.split('_')
                    fileName = fileName[1] + '_' + fileName[2] + '_' + fileName[3]

                    # store file
                    with open(os.path.join(out_path, name + fileName) + '.pkl', 'wb') as f:
                        pickle.dump(pc, f)

                else:
                    counters['missed'] += 1

    except Exception as e:
        print(f'Error {e} in file {fileName}')

# print(f'count keep ground: ', counters['keep_ground'])
# print(f'count not enough ground points: ', counters['need_ground'])
# print(f'total_count:', counters['total_count'])
# print(' ----- Constrained Sampling ------')
# print(f'counter sampled below {TH_1} m: ', counters['count_sample3'])
# print(f'counter sampled below {TH_2} m: ', counters['count_sample8'])
# print(f'counter sampled all pc: ', counters['sample_all'])
# print(f'counter total sampled: ', counters['count_sample3'] + counters['count_sample8'] + counters['sample_all'])
# print(f'counter less than n_points: ', counters['missed'])


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(remove_ground_and_outliers, files_list, 1),
                         max_value=len(files_list)):  # redirect_stdout=True)
        pass
    p.close()
    p.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/datasets/towers_100x100_cls',
                        help='output folder where processed files are stored')
    parser.add_argument('--in_path', default='/dades/LIDAR/towers_detection/LAS_data_windows_100x100/')
    parser.add_argument('--datasets', type=list, default=['RIBERA', 'CAT3'], help='list of datasets names')
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--max_z', type=float, default=100.0)

    args = parser.parse_args()
    start_time = time.time()

    out_path = args.out_path

    for dataset_name in args.datasets:
        path = args.in_path + dataset_name

        # for input_path in paths:
        logging.info(f'Dataset: {dataset_name}')
        logging.info(f'Input path: {args.in_path}')

        # !!!!!!!!! IMPORTANT !!!!!!!!!
        # First execute pdal on all files to get HeightAboveGround

        # ------ Remove ground, noise, outliers and normalize ------
        logging.info(f"1. Remove points of ground, noise and outliers, normalize"
                     f"and add constrained sampling flag ")

        # out_path = os.path.join(out_path, 'normalized_' + str(n_points))
        logging.info(f'output path: {args.out_path}')
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)

        files = glob.glob(os.path.join(path, '*.las'))
        # Multiprocessing
        parallel_proc(files, num_cpus=10)

        print("--- Remove ground and noise time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
        rm_ground_time = time.time()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
