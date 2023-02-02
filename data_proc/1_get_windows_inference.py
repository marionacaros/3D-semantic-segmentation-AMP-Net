import argparse
from utils.utils import *
import logging
import time
from alive_progress import alive_bar
import hashlib
import pickle

logging.basicConfig(format='[%(asctime)s %(levelname)s %(filename)20s()] %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m %H:%M:%S')


def store_las_file_from_pc(pc, fileName, path_las_dir, dataset, labels2remove=[135, 106]):
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
    for label in labels2remove:
        p_class[p_class == int(label)] = 30

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


def load_data_and_split(w_size, in_data_path, dataset, out_path, noise_labels):
    """
    Load LAS files with all its features from in_data_path
    each las pointcloud is stores as an array of features
    if dataset == 'BDN', NIR & RGB values are set to zero
    LAS point clouds are split into smaller point clouds through split_pointcloud()

    :param w_size: output size of point clouds (called windows)
    :param in_data_path: input data path
    :param dataset: name of dataset
    :param out_path: output data path
    :param noise_labels: values of noise labels
    """

    logging.info('Loading LAS files...')
    files = glob.glob(os.path.join(in_data_path, '*.las'))
    dir_name = 'files_' + str(w_size[0]) + 'x' + str(w_size[1])

    # output directory where processed files are stored
    save_path = os.path.join(out_path, args.dataset_name)
    if not os.path.exists(os.path.join(save_path, dir_name)):
        os.makedirs(os.path.join(save_path, dir_name))

    with alive_bar(len(files), bar='filling', spinner='waves') as bar:
        for f in files:
            name_f = f.split('/')[-1].split('.')[0]

            las_pc = laspy.read(f)
            nir = las_pc.nir
            red = las_pc.red
            green = las_pc.green
            blue = las_pc.blue
            if dataset == 'BDN':
                nir = np.zeros(len(las_pc))
                red = np.zeros(len(las_pc))
                green = np.zeros(len(las_pc))
                blue = np.zeros(len(las_pc))

            block_pc = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                                  las_pc.intensity,
                                  red, green, blue,
                                  nir))
            # las_pc.return_number,
            # las_pc.number_of_returns,

            # each LAS file is split here
            split_pointcloud(block_pc, f_name=name_f, dir=dir_name, path=save_path, w_size=w_size,
                             dataset=dataset, noise_labels=noise_labels)
            bar()


def split_pointcloud(pointcloud, f_name, dir='files_40x40', path='', w_size=[40, 40], dataset='', noise_labels=[135]):
    """ Split point cloud with a sliding window to smaller point clouds of size w_size.

        :param pointcloud: array of .LAS point cloud blocks (usually of 1km x 1km)
        :param f_name:
        :param dir:
        :param path:
        :param w_size: size of window
        :param dataset: name of dataset
        :param noise_labels: value of noise labels

        :return pc_w
    """
    i_w = 0
    x_min, y_min, z_min = pointcloud[0].min(), pointcloud[1].min(), pointcloud[2].min()
    x_max, y_max, z_max = pointcloud[0].max(), pointcloud[1].max(), pointcloud[2].max()

    for y in range(round(y_min), round(y_max), round(w_size[1] / 2)):
        bool_w_y = np.logical_and(pointcloud[1] < (y + w_size[1]), pointcloud[1] > y)

        for x in range(round(x_min), round(x_max), round(w_size[0] / 2)):
            bool_w_x = np.logical_and(pointcloud[0] < (x + w_size[0]), pointcloud[0] > x)
            bool_w = np.logical_and(bool_w_x, bool_w_y)
            i_w += 1

            if any(bool_w):
                if pointcloud[:, bool_w].shape[1] > 0:
                    # store las file
                    file = 'pc_' + dataset + '_' + f_name + '_w' + str(i_w)
                    store_las_file_from_pc(pointcloud[:, bool_w], file, os.path.join(path, dir), dataset, noise_labels)
                    i_w += 1
                    # Store point cloud of window in pickle
                    # stored_f = os.path.join(path, dir, 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w) + '.pkl')
                    # with open(stored_f, 'wb') as f:
                    #     pickle.dump(pointcloud[:, bool_w], f)

    print(f'Stored windows of block {f_name}: {i_w}')


if __name__ == '__main__':

    logging.info("Make sure output directory has write permissions")
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--w_size', type=int, default=40, help='width of squared window')
    parser.add_argument('--dataset_name', type=str, default='test_inference',
                        help='common name for all files, a directory with this name will be created to store output '
                             'files')
    parser.add_argument('--LAS_files_path', type=str)
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/LAS_data_windows',
                        help='output directory where processed files are stored')
    parser.add_argument('--noise_labels', type=list, default=[135, 106], help='Assigned values to label noise')

    args = parser.parse_args()
    logging.info(f"Output directory: {args.out_path}")
    logging.info(f'Split LAS files into squared windows of {args.w_size} meters')
    logging.info(f"Dataset name: {args.dataset_name}")

    load_data_and_split(w_size=[args.w_size, args.w_size],
                        in_data_path=args.LAS_files_path,
                        dataset=args.dataset_name,
                        out_path=args.out_path,
                        noise_labels=args.noise_labels)

    logging.info("TIME: %s h" % (round((time.time() - start_time) / 3600, 3)))
