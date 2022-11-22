import json
import os
import csv
import torch.utils.data as data
import torch
import glob
import numpy as np
import pickle
import random
from k_means_constrained import KMeansConstrained
import itertools


class LidarKmeansDataset4Train(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 number_of_windows=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):
        # 0 -> background
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.n_windows = number_of_windows
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        for file in self.files:
            if 'pc_' in file:
                self.classes_mapping[file] = 0
            elif 'tower_' in file:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def __getitem__(self, index):
        """
        If task is classification and no path_kmeans is given, it returns a raw point cloud (pc), labels and filename
        If task is classification and path_kmeans is given, it returns a clustered point cloud into windows (pc_w),
        labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]

        # load data clustered in windows with k-means
        pc = torch.load(filename, map_location=torch.device('cpu'))
        # pc_w size [2048, dims, w_len]
        labels = self.get_labels(pc, self.classes_mapping[self.files[index]], self.task)

        return pc, labels, filename

    @staticmethod
    def get_labels(pointcloud,
                   point_cloud_class,
                   task='classification'):
        """
        Get labels for classification or segmentation

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> low vegetation
        3 -> medium vegetation
        4 -> high vegetation

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        if task == 'segmentation':
            segment_labels = pointcloud[:, 3, :]  # [2048, 5]
            segment_labels[segment_labels == 15] = 100
            # segment_labels[segment_labels == 14] = 200
            segment_labels[segment_labels == 3] = 200
            segment_labels[segment_labels == 4] = 300
            segment_labels[segment_labels == 5] = 400
            segment_labels[segment_labels < 100] = 0
            segment_labels = (segment_labels/100)

            # segment_labels[segment_labels == 15] = 1
            # segment_labels[segment_labels != 15] = 0

            labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        elif task == 'classification':
            labels = [point_cloud_class]
            labels = labels * pointcloud.shape[2]

        return labels


class LidarKmeansDataset4Test(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 number_of_windows=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):
        # 0 -> background
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.n_windows = number_of_windows
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        for file in self.files:
            if 'pc_' in file:
                self.classes_mapping[file] = 0
            elif 'tower_' in file:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def __getitem__(self, index):
        """
        If task is classification and no path_kmeans is given, it returns a raw point cloud (pc), labels and filename
        If task is classification and path_kmeans is given, it returns a clustered point cloud into windows (pc_w),
        labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]

        # load data clustered in windows with k-means
        with open(filename, 'rb') as f:
            pc = torch.from_numpy(pickle.load(f)).type(torch.FloatTensor)  # [2048, 11]

        cluster_lists = self.kmeans_clustering(pc)

        # pc_w size [2048, dims, w_len]
        labels = self.get_labels(cluster_lists, self.classes_mapping[self.files[index]], self.task)

        return cluster_lists, labels, filename

    @staticmethod
    def kmeans_clustering(in_pc, n_points=2048):
        # in_pc [n_p, dim]
        MAX_CLUSTERS = 5
        cluster_lists=[]

        # if point cloud is larger than n_points we cluster them with k-means
        if in_pc.shape[0] > 2*n_points:

            # K-means clustering
            k_clusters = int(np.floor(in_pc.shape[0] / n_points))
            # leftover = in_pc.shape[0] % n_points
            # if k_clusters <= 5 and leftover != 0:
            #     # Number of points must be multiple of n_points
            #     replicate_idx = random.sample(range(0, in_pc.shape[0], 1), n_points - leftover)
            #     replicate_points = in_pc[replicate_idx]
            #     replicate_points[:, 3] = -1
            #     in_pc = torch.cat((in_pc, replicate_points))

            if k_clusters > 5:
                k_clusters = 5

            if k_clusters * n_points > in_pc.shape[0]:
                print('debug error')

            clf = KMeansConstrained(n_clusters=k_clusters, size_min=n_points)
            i_f = [0, 1, 9]  # x,y, NDVI
            i_cluster = clf.fit_predict(in_pc[:, i_f].numpy())  # array of ints -> indices to each of the windows

            # get tuple cluster points
            tuple_cluster_points = list(zip(i_cluster, in_pc))
            cluster_list_tuples = [list(item[1]) for item in
                             itertools.groupby(sorted(tuple_cluster_points, key=lambda x: x[0]), key=lambda x: x[0])]

            for cluster in cluster_list_tuples:
                pc_features_cluster = torch.stack([feat for (i_c, feat) in cluster])  # [2048, 11]
                cluster_lists.append(pc_features_cluster)
            #     pc_w = torch.cat((pc_w, pc_features_cluster.unsqueeze(2)), 2)  # [2048, 11, 1]

        else:
            cluster_lists.append(in_pc)

        return cluster_lists


    @staticmethod
    def get_labels(cluster_lists,
                   point_cloud_class,
                   task='classification'):
        """
        Get labels for classification or segmentation

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> cables
        3 -> low vegetation
        4 -> medium vegetation
        5 -> high vegetation

        """

        segment_labels_list = []

        if task == 'segmentation':
            for pointcloud in cluster_lists:

                segment_labels = pointcloud[:, 3]
                segment_labels[segment_labels == 15] = 100
                segment_labels[segment_labels == 14] = 200
                segment_labels[segment_labels == 3] = 300
                segment_labels[segment_labels == 4] = 400
                segment_labels[segment_labels == 5] = 500
                segment_labels[segment_labels < 100] = 0
                segment_labels = (segment_labels/100)

                # segment_labels[segment_labels == 15] = 1
                # segment_labels[segment_labels != 15] = 0

                labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
                segment_labels_list.append(labels)

        # elif task == 'classification':
        #     labels = [point_cloud_class]
        #     labels = labels * pointcloud.shape[2]

        return segment_labels_list


class LidarDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 number_of_windows=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.n_windows = number_of_windows
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        for file in self.files:
            if 'pc_' in file:
                self.classes_mapping[file] = 0
            elif 'tower_' in file:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def __getitem__(self, index):
        """
        If task is classification, it returns a raw point cloud (pc), labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling)
        # pc size [2048,11]

        labels = self.get_labels(pc, self.classes_mapping[self.files[index]], self.task)

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):

        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [2048, 11]
        # pc = pc[:,:10]
        np.random.shuffle(pc)

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, -1] == 1]  # should be label of position 11

        # sample points if fixed_num_points (random sampling, no RNN)
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed (no RNN)
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels(pointcloud,
                   point_cloud_class,
                   task='classification'):
        """
        Get labels for classification or segmentation

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> low vegetation
        3 -> medium vegetation
        4 -> high vegetation

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        if task == 'segmentation':
            segment_labels = pointcloud[:, 3]
            segment_labels[segment_labels == 15] = 100
            # segment_labels[segment_labels == 14] = 200
            segment_labels[segment_labels == 3] = 200
            segment_labels[segment_labels == 4] = 300
            segment_labels[segment_labels == 5] = 400
            segment_labels[segment_labels < 100] = 0
            segment_labels = (segment_labels/100)

            # segment_labels[segment_labels == 15] = 1
            # segment_labels[segment_labels != 15] = 0

            labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        elif task == 'classification':
            labels = [point_cloud_class]
            labels = labels * pointcloud.shape[2]

        return labels


class BdnDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder, task='classification', number_of_points=None):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        self.task = task

        category_file = os.path.join(self.dataset_folder, 'BdnDataFolders.txt')  # reduced_dataFolders
        self.folders_to_classes_mapping = {}
        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[0]] = k

        self.paths_files = glob.glob(os.path.join(self.dataset_folder, 'bdn*/*'))
        self.files = [(f.split('/')[-2], f.split('/')[-1]) for f in self.paths_files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_cloud_class = self.folders_to_classes_mapping[folder]

        return self.prepare_data(self.paths_files[index],
                                 self.number_of_points,
                                 point_cloud_class=point_cloud_class,
                                 fileName=file)

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     fileName=''):
        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # get only x,y,z and transpose
        ###pc = pc[:, :3]
        # print(point_cloud.shape)  # (1000, 3)
        if number_of_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]
        # normalize x,y
        # min_z = 4
        # max_z = 28
        # pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
        # pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
        # pc[:, 2] = (pc[:, 2] - min_z) / (max_z - min_z)

        pc = torch.from_numpy(pc)
        point_cloud_class = torch.tensor(point_cloud_class)

        return pc, point_cloud_class, fileName


class ShapeNetDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    NUM_SEGMENTATION_CLASSES = 50

    POINT_DIMENSION = 3

    PER_CLASS_NUM_SEGMENTATION_CLASSES = {
        'Airplane': 4,
        'Bag': 2,
        'Cap': 2,
        'Car': 4,
        'Chair': 4,
        'Earphone': 3,
        'Guitar': 3,
        'Knife': 2,
        'Lamp': 4,
        'Laptop': 2,
        'Motorbike': 6,
        'Mug': 2,
        'Pistol': 3,
        'Rocket': 3,
        'Skateboard': 3,
        'Table': 3,
    }

    def __init__(self,
                 dataset_folder,
                 number_of_points=2500,
                 task='classification',
                 train=True):
        self.dataset_folder = dataset_folder
        self.number_of_points = number_of_points
        assert task in ['classification', 'segmentation']
        self.task = task
        self.train = train

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')
        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]

        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):
        folder, file = self.files[index]
        point_file = os.path.join(self.dataset_folder,
                                  folder,
                                  'points',
                                  '%s.pts' % file)
        segmentation_label_file = os.path.join(self.dataset_folder,
                                               folder,
                                               'points_label',
                                               '%s.seg' % file)
        point_cloud_class = self.folders_to_classes_mapping[folder]
        if self.task == 'classification':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class)
        elif self.task == 'segmentation':
            return self.prepare_data(point_file,
                                     self.number_of_points,
                                     point_cloud_class=point_cloud_class,
                                     segmentation_label_file=segmentation_label_file,
                                     segmentation_classes_offset=self.segmentation_classes_offset[folder])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
        point_cloud = np.loadtxt(point_file).astype(np.float32)
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file).astype(np.int64)
            if number_of_points:
                segmentation_classes = segmentation_classes[sampling_indices]
            segmentation_classes = segmentation_classes + segmentation_classes_offset - 1
            segmentation_classes = torch.from_numpy(segmentation_classes)
            return point_cloud, segmentation_classes
        elif point_cloud_class is not None:
            point_cloud_class = torch.tensor(point_cloud_class)
            return point_cloud, point_cloud_class
        else:
            return point_cloud
