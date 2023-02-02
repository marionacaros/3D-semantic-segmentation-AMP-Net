import argparse
from utils.utils import *
import logging
import time
from alive_progress import alive_bar
import hashlib
import pickle

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def split_dataset_windows(DATASET, LAS_PATH, W_SIZE=[40, 40]):
    global save_path
    start_time = time.time()
    print(f"Dataset: {DATASET}")
    logging.info('Loading LAS files')
    files = glob.glob(os.path.join(LAS_PATH, '*.las'))

    dir_name = 'point_clouds_' + str(W_SIZE[0]) + 'x' + str(W_SIZE[1])

    # counters
    i_towers = 0
    i_lines = 0

    if not os.path.exists(os.path.join(save_path, dir_name)):
        os.makedirs(os.path.join(save_path, dir_name))

    with alive_bar(len(files), bar='filling', spinner='waves') as bar:
        # loop over blocks
        for f in files:
            i_w = 0
            label = 'pc_'
            name_f = f.split('/')[-1].split('.')[0]
            las_pc = laspy.read(f)
            if DATASET == 'CAT3' or DATASET == 'RIBERA':
                nir = las_pc.nir
                red = las_pc.red
                green = las_pc.green
                blue = las_pc.blue
            elif DATASET == 'BDN':
                nir = np.zeros(len(las_pc))
                red = np.zeros(len(las_pc))
                green = np.zeros(len(las_pc))
                blue = np.zeros(len(las_pc))

            pc = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                            las_pc.intensity,
                            red, green, blue,
                            nir))
            # get coords
            x_min, y_min, z_min = pc[0].min(), pc[1].min(), pc[2].min()
            x_max, y_max, z_max = pc[0].max(), pc[1].max(), pc[2].max()

            # split point cloud
            for y in range(round(y_min), round(y_max), W_SIZE[1]):
                bool_w_y = np.logical_and(pc[1] < (y + W_SIZE[1]), pc[1] > y)

                for x in range(round(x_min), round(x_max), W_SIZE[0]):
                    bool_w_x = np.logical_and(pc[0] < (x + W_SIZE[0]), pc[0] > x)
                    bool_w = np.logical_and(bool_w_x, bool_w_y)
                    i_w += 1

                    if any(bool_w):
                        if pc[:, bool_w].shape[1] > 0:
                            set_lables = set(pc[3])
                            if 15 in set_lables:
                                label = 'tower_'
                                i_towers += 1

                            if 14 in set_lables:
                                label = 'tower_'
                                i_lines += 1

                            # store las file
                            file = label + DATASET + '_' + name_f + '_w' + str(i_w)
                            store_las_file_from_pc(pc[:, bool_w], file, save_path, DATASET)
                            i_w += 1
                            label = 'pc_'

            # print(f'Stored windows of block {name_f}: {i_w}')
            bar()

    print(f'Point clouds with towers: {i_towers}')
    print(f'Point clouds with lines: {i_lines}')

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
    # ------------------------------------------------------------------------------------------------------------


def read_las_files(path):
    """

    :param path: path containing LAS files
    :return: dict with [x,y,z,class]
    """
    dict_pc = {}
    files = glob.glob(os.path.join(path, '*.las'))
    with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
        for f in files:
            fileName = f.split('/')[-1].split('.')[0]
            las_pc = laspy.read(f)
            dict_pc[fileName] = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification))
            bar()

    return dict_pc


def store_las_file_from_pc(pc, fileName, path_las_dir, dataset):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.4")  # we need this format for processing with PDAL
    # header.add_extra_dim(laspy.ExtraBytesParams(name="nir_extra", type=np.int32))
    header.offsets = np.array([0, 0, 0])  # np.min(pc, axis=0)
    header.scales = np.array([1, 1, 1])

    # 2. Create a Las
    las = laspy.LasData(header)
    las.x = pc[0].astype(np.int32)
    las.y = pc[1].astype(np.int32)
    las.z = pc[2].astype(np.int32)
    p_class = pc[3].astype(np.int8)
    las.intensity = pc[4].astype(np.int16)
    las.red = pc[5].astype(np.int16)
    las.green = pc[6].astype(np.int16)
    las.blue = pc[7].astype(np.int16)
    # las.return_number = pc[5].astype(np.int8)
    # las.number_of_returns = pc[6].astype(np.int8)

    # Classification unsigned char 1 byte (max is 31)
    p_class[p_class == 135] = 30
    p_class[p_class == 106] = 31
    las.classification = p_class

    if not os.path.exists(path_las_dir):
        os.makedirs(path_las_dir)
    las.write(os.path.join(path_las_dir, fileName + ".las"))

    if dataset != 'BDN':  # BDN data do not have NIR
        # Store NIR with hash ID
        nir = {}
        for i in range(pc.shape[1]):
            mystring = str(int(pc[0, i])) + '_' + str(int(pc[1, i])) + '_' + str(int(pc[2, i]))
            hash_object = hashlib.md5(mystring.encode())
            nir[hash_object.hexdigest()] = int(pc[8, i])

        with open(os.path.join(path_las_dir, fileName + '_NIR.pkl'), 'wb') as f:
            pickle.dump(nir, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/LAS_data_windows_200x200',
                        help='output folder where processed files are stored')
    parser.add_argument('--min_p', type=int, default=10, help='minimum number of points in object')
    parser.add_argument('--sel_class', type=int, default=15, help='selected class')
    parser.add_argument('--datasets', type=list, default=['CAT3', 'RIBERA'], help='list of datasets names')
    parser.add_argument('--LAS_files_path', type=str)
    parser.add_argument('--w_size', default=[200, 200])
    parser.add_argument('--data_augm', default=10)

    args = parser.parse_args()

    SEL_CLASS = args.sel_class
    # 15 corresponds to power transmission tower
    # 18 corresponds to other towers
    LAS_files_path = args.LAS_files_path

    # Our Datasets
    for DATASET_NAME in args.datasets:
        # paths
        if DATASET_NAME == 'BDN':
            LAS_files_path = '/mnt/Lidar_K/PROJECTES/0025310000_VOLTA_MachineLearning_Badalona_FBK_5anys/Lliurament_211203_Mariona/LASCLAS_AMB_FOREST-URBAN/FOREST'
        elif DATASET_NAME == 'CAT3':
            LAS_files_path = '/mnt/Lidar_M/DEMO_Productes_LIDARCAT3/LAS_def'
        elif DATASET_NAME == 'RIBERA':
            LAS_files_path = '/mnt/Lidar_O/DeepLIDAR/VolVegetacioRibera_ClassTorres-Linies/LAS'

        save_path = os.path.join(args.out_path, DATASET_NAME)

        split_dataset_windows(DATASET_NAME, LAS_files_path, args.w_size)
