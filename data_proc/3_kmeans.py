import logging
from utils import *
import pickle
import time
from k_means_constrained import KMeansConstrained
import itertools
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import random
import multiprocessing

i_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_80x80_10p/normalized_2048'
o_path = '/dades/LIDAR/towers_detection/datasets/kmeans_80x80_w16/'
N_POINTS = 2048
NUM_CPUS = 20


def split_kmeans(file_path, n_points=N_POINTS, max_clusters=16, plot=False, writer_tensorboard=None):
    """ split point cloud in windows of fixed size (n_points) with k-means
    Fill empty windows with duplicate points of previous windows
    Number of points must be multiple of n_points, so points left over are removed

    :param file_path:
    :param in_pc: input point cloud [total_points, dims]
    :param n_points: number of points for each window
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filename: ''

    :return pc_w: tensor containing point cloud in windows of fixed size [2048, 11, w_len]
    """

    filename = file_path.split('/')[-1].split('.')[0]

    with open(file_path, 'rb') as f:
        in_pc = torch.Tensor(pickle.load(f))

    # if point cloud is larger than n_points we cluster them with k-means
    if in_pc.shape[0] >= 2 * n_points:

        # tensor for points per window
        pc_w = torch.FloatTensor()

        # K-means clustering
        k_clusters = int(np.floor(in_pc.shape[0] / n_points))
        if k_clusters > max_clusters:
            k_clusters = max_clusters
            ix = random.sample(range(in_pc.shape[0]), n_points * max_clusters)
            in_pc = in_pc[ix, :]

        if in_pc.shape[0] % n_points != 0:
            # Number of points must be multiple of n_points, so points left over are removed
            in_pc = in_pc[:n_points * (in_pc.shape[0] // n_points), :]

        clf = KMeansConstrained(n_clusters=k_clusters, size_min=n_points, size_max=n_points,
                                n_init=5, max_iter=150, tol=1e-4,
                                verbose=False, random_state=None, copy_x=True)
        i_f = [0, 1, 9]  # x,y, NDVI
        i_cluster = clf.fit_predict(in_pc[:, i_f].numpy())  # array of ints -> indices to each of the windows
        print('\n', max(i_cluster))

        # get tuple cluster points
        tuple_cluster_points = list(zip(i_cluster, in_pc))
        cluster_lists = [list(item[1]) for item in
                         itertools.groupby(sorted(tuple_cluster_points, key=lambda x: x[0]), key=lambda x: x[0])]
        if plot and k_clusters > 1:
            plot_3d_sequence_tensorboard(in_pc, writer_tensorboard, filename, i_w=0, title='original',
                                         n_clusters=k_clusters)

        for cluster in cluster_lists:
            pc_features_cluster = torch.stack([feat for (i_c, feat) in cluster])  # [2048, 11]
            pc_w = torch.cat((pc_w, pc_features_cluster.unsqueeze(2)), 2)  # [2048, 11, 1]

            if plot and k_clusters > 1:
                plot_3d_sequence_tensorboard(pc_features_cluster, writer_tensorboard, filename,
                                             i_w=cluster[0][0] + 1,
                                             title='cluster ' + str(cluster[0][0]), n_clusters=k_clusters)
    else:
        if in_pc.shape[0] != n_points:
            ix = random.sample(range(in_pc.shape[0]), n_points)
            in_pc = in_pc[ix, :, ]
        pc_w = in_pc.unsqueeze(2)

    torch.save(pc_w, o_path + 'kmeans_' + filename + '.pt')


def parallel_kmeans(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(split_kmeans, files_list, 1), max_value=len(files_list), redirect_stdout=True): #
        pass

    p.close()
    p.join()


if __name__ == '__main__':
    logging.info(f"3. K-means clustering")
    start_time = time.time()

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/kmeans/'
    writer = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'kmeans')
    files = glob.glob(os.path.join(i_path, 'tower_*pkl'))

    # Sort list of files in directory by size
    files = sorted(files, key=lambda x: os.stat(x).st_size, reverse=False)

    if not os.path.exists(o_path):
        # Create a new directory because it does not exist
        os.makedirs(o_path)

    # run k-means in parallel
    parallel_kmeans(files, NUM_CPUS)

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
