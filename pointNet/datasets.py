import os
import torch.utils.data as data
import torch
import numpy as np
import pickle
from utils.utils import fps


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
        pc = np.concatenate((pc[:, :3], pc[:, 4].unsqueeze(1), pc[:, 6:8], pc[:, 9].unsqueeze(1)), axis=1)
        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):

        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [2048, 11]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 10] == 1]  # should be flag of position 10

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

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)

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
            segment_labels[segment_labels == 14] = 200
            segment_labels[segment_labels == 3] = 300  # low veg
            segment_labels[segment_labels == 4] = 300  # med veg
            segment_labels[segment_labels == 5] = 400
            segment_labels[segment_labels < 100] = 0
            segment_labels = (segment_labels / 100)

            # segment_labels[segment_labels == 15] = 1
            # segment_labels[segment_labels != 15] = 0

            labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        elif task == 'classification':
            labels = point_cloud_class  # for training data
            # if not point_cloud_class
            # labels = set(pointcloud[:, 3].numpy().astype(int))
            # if 15 in labels:
            #     labels = 1
            # else:
            #     labels = 0

        return labels


class LidarDatasetExpanded(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

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
                               fixed_num_points=self.fixed_num_points)
        # pc size [2048,11]

        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:
            labels = self.get_labels_cls(pc)

        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)
        # Normalize
        pc = self.pc_normalize_neg_one(pc)
        # .unsqueeze(1), pc[:, 6:8], pc[:, 9].unsqueeze(1)), axis=1)
        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):

        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [2048, 11]

        # remove noise
        pc = pc[np.where(pc[:, 3] != 30)]
        pc = pc[np.where(pc[:, 3] != 7)]
        pc = pc[np.where(pc[:, 3] != 2)]
        pc = pc[np.where(pc[:, 3] != 8)]
        pc = pc[np.where(pc[:, 3] != 13)]
        pc = pc[np.where(pc[:, 3] != 14)]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 10] == 1]  # should be flag of position 10

        # sample points if fixed_num_points (random sampling, no RNN)
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]
            # FPS -> too slow
            # pc = fps(pc, number_of_points)

        try:
            # duplicate points if needed
            if fixed_num_points and pc.shape[0] < number_of_points:
                points_needed = number_of_points - pc.shape[0]
                rdm_list = np.random.randint(0, pc.shape[0], points_needed)
                extra_points = pc[rdm_list, :]
                pc = np.concatenate([pc, extra_points], axis=0)
        except Exception as e:
            print(e)
            print(f'\n {point_file}\n')

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = pc[:, 0] * 2 - 1
        pc[:, 1] = pc[:, 1] * 2 - 1
        # centroid = np.mean(pc, axis=0)
        # pc = pc - centroid
        # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        # pc = pc / m
        return pc
    @staticmethod
    def get_labels_cls(pointcloud):
        """ Get labels for classification or segmentation

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        """
        labels = set(pointcloud[:, 3].numpy().astype(int))
        if 15 in labels or 14 in labels:
            labels = 1
        else:
            labels = 0
        return labels

    @staticmethod
    def get_labels_segmen(pointcloud):
        """

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> lines
        3 -> low-med vegetation
        4 -> high vegetation
        5 -> other towers

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 300  # med veg
        segment_labels[segment_labels == 5] = 400
        # segment_labels[segment_labels == 18] = 500
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        return labels


class LidarKmeansDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2  # we use 2 dimensions (x,y) to learn T-Net transformation

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 sort_kmeans=False,
                 get_centroids=True):

        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.files = [f.split('.')[0] for f in files]
        self.sort_kmeans = sort_kmeans
        self.get_centroids = get_centroids
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, 'kmeans_'+ f +'.pt') for f in self.files]

    def __len__(self):
        return len(self.paths_files)

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
        # pc size [2048, dims, w_len]

        # find the indices where pc[:, 3,:] == 30
        indices = np.where(pc[:, 3, :] == 30)
        pc = np.delete(pc, indices[0], axis=0)
        indices = np.where(pc[:, 3, :] == 7)
        pc = np.delete(pc, indices[0], axis=0)
        indices = np.where(pc[:, 3, :] == 2)
        pc = np.delete(pc, indices[0], axis=0)
        indices = np.where(pc[:, 3, :] == 8)
        pc = np.delete(pc, indices[0], axis=0)
        indices = np.where(pc[:, 3, :] == 13)
        pc = np.delete(pc, indices[0], axis=0)
        indices = np.where(pc[:, 3, :] == 14)
        pc = np.delete(pc, indices[0], axis=0)


        # Targets
        if self.task == 'classification':
            labels_cls = self.get_labels_cls(pc)

        labels_segmen = self.get_labels_segmen(pc)
        # Drop not used features data
        pc = np.concatenate((pc[:, :3, :], pc[:, 4:10, :]), axis=1)
        # Normalize
        pc = self.pc_normalize_neg_one(pc)

        # Get cluster centroids
        if self.get_centroids:
            centroids = self.get_cluster_centroid(pc)

        if self.task == 'segmentation':
            return pc, labels_segmen, filename, centroids
        else:
            return pc, labels_cls, filename, centroids, labels_segmen

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim, seq]
        """
        pc[:, 0, :] = pc[:, 0, :] * 2 - 1
        pc[:, 1, :] = pc[:, 1, :] * 2 - 1
        # centroid = np.mean(pc, axis=0)
        # pc = pc - centroid
        # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        # pc = pc / m
        return pc

    @staticmethod
    def sort(pc):
        """
        sort clusters from max Y min X to min Y max X
        :param pc:
        :return:
        """
        sorted_pc = torch.FloatTensor()
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        means = mean_x + mean_y
        order = torch.argsort(means)
        for ix in order:
            sorted_pc = torch.cat([sorted_pc, pc[:, :, ix].unsqueeze(-1)], dim=2)

        return sorted_pc

    @staticmethod
    def get_cluster_centroid(pc):
        """
        :param pc:
        :return:
        """
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        centroids = np.stack([mean_x, mean_y], axis=0)
        return centroids

    @staticmethod
    def get_labels_cls(pointcloud):
        """ Get labels for classification or segmentation

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        """
        labels = np.unique(pointcloud[:, 3].numpy()).astype(int)
        if 15 in labels or 14 in labels:
            label_cls = torch.LongTensor([1])
        else:
            label_cls = torch.LongTensor([0])
        return label_cls

    @staticmethod
    def get_labels_segmen(pointcloud):
        """

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> lines
        3 -> low-med vegetation
        4 -> high vegetation
        5 -> other towers

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 300  # med veg
        segment_labels[segment_labels == 5] = 400
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        return labels


class LidarDataset4Test(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

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
            # pc = torch.load(f)
            pc = pickle.load(f)
            pc = torch.from_numpy(pc).type(torch.FloatTensor)

        pc = np.concatenate((pc[:, :3], pc[:, 4:10], pc[:, 3].unsqueeze(-1)), axis=1)  # last col is label
        pc = self.pc_normalize_neg_one(pc)
        return pc, filename

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = pc[:, 0] * 2 - 1
        pc[:, 1] = pc[:, 1] * 2 - 1
        return pc


class LidarInferenceDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 files=None,
                 c_sample=False):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.files = files
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        If task is classification, it returns a raw point cloud (pc), labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               constrained_sample=self.constrained_sampling)
        # pc size [2048,11]
        return pc, filename

    @staticmethod
    def prepare_data(point_file,
                     constrained_sample=False):
        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [2048, 11]

        # np.random.shuffle(pc)

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 10] == 1]  # should be flag of position 10

        pc = torch.from_numpy(pc)
        return pc
