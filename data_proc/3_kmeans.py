import logging
from utils import *
import pickle
import time
from k_means_constrained import KMeansConstrained
import itertools
from utils.utils import plot_3d_sequence_tensorboard
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter


def split_kmeans(in_pc, n_points, plot=False, writer_tensorboard=None, filename=''):
    """ split point cloud in windows of fixed size (n_points) with k-means
    Fill empty windows with duplicate points of previous windows
    Number of points must be multiple of n_points, so points left over are removed

    :param in_pc: input point cloud [total_points, dims]
    :param n_points: number of points for each window
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filename: ''

    :return pc_w: tensor containing point cloud in windows of fixed size [2048, 11, w_len]
    """

    # if point cloud is larger than n_points we cluster them with k-means
    if in_pc.shape[0] > n_points:

        # tensor for points per window
        pc_w = torch.FloatTensor()

        if in_pc.shape[0] % n_points != 0:
            # Number of points must be multiple of n_points, so points left over are removed
            in_pc = in_pc[:n_points * (in_pc.shape[0] // n_points), :]
        # K-means clustering
        k_clusters = int(np.floor(in_pc.shape[0] / n_points))
        clf = KMeansConstrained(n_clusters=k_clusters, size_min=n_points, size_max=n_points)
        i_f = [0, 1, 9]  # x,y, NDVI
        i_cluster = clf.fit_predict(in_pc[:, i_f].numpy())  # array of ints -> indices to each of the windows

        # get tuple cluster points
        tuple_cluster_points = list(zip(i_cluster, in_pc))
        cluster_lists = [list(item[1]) for item in
                         itertools.groupby(sorted(tuple_cluster_points, key=lambda x: x[0]), key=lambda x: x[0])]
        if plot:
            plot_3d_sequence_tensorboard(in_pc, writer_tensorboard, filename, i_w=0, title='original',
                                         n_clusters=k_clusters)

        for cluster in cluster_lists:
            pc_features_cluster = torch.stack([feat for (i_c, feat) in cluster])  # [2048, 11]
            pc_w = torch.cat((pc_w, pc_features_cluster.unsqueeze(2)), 2)  # [2048, 11, 1]

            if plot:
                plot_3d_sequence_tensorboard(pc_features_cluster, writer_tensorboard, filename,
                                             i_w=cluster[0][0] + 1,
                                             title='cluster ' + str(cluster[0][0]), n_clusters=k_clusters)
    else:
        pc_w = in_pc.unsqueeze(2)

    return pc_w


if __name__ == '__main__':
    logging.info(f"3. K-means clustering")
    i_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p/normalized_2048'
    N_POINTS = 2048
    start_time = time.time()

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/rnn/'
    writer = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'rnn' )
    files = glob.glob(os.path.join(i_path, 'tower*pkl'))

    for file in progressbar(files):
        fileName = file.split('/')[-1]
        with open(file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)

        pc_w = split_kmeans(torch.Tensor(pc), n_points=N_POINTS, plot=True, writer_tensorboard=writer,
                            filename=fileName)
        o_path = '/dades/LIDAR/towers_detection/datasets/new_kmeans/'
        torch.save(pc_w, o_path + 'kmeans_' + fileName)

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
