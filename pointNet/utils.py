import numpy as np
import matplotlib.pyplot as plt
import os
import torch
# from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from progressbar import progressbar


def transform_2d_img_to_point_cloud(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i] / 2) / img_array.shape[i]
    return indices.astype(np.float32)


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'bo', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    if save_to_file:
        fig.savefig('figures/Loss.png', dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'bo', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_3d(points, name, n_points=2000):
    points = points.view(n_points, -1).numpy()
    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], s=10, marker='o', cmap="viridis",
                    alpha=0.5)
    plt.colorbar(sc, shrink=0.5, pad=0.05)
    directory = 'figures/results_models'
    plt.title(name + ' classes: ' + str(set(points[:, 3].astype('int'))))
    plt.show()
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_subplots(points_tNet, fileName, points_i):
    fig = plt.figure(figsize=[12, 6])
    #  First subplot
    # ===============
    # set up the axes for the first plot
    # print('points_input', points_i.shape)
    # print('points_tNet', points_tNet.shape)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.title.set_text('Input data: ' + fileName)
    sc = ax.scatter(points_i[0, :], points_i[1, :], points_i[2, :], c=points_i[2, :], s=10,
                    marker='o',
                    cmap="winter", alpha=0.5)
    # fig.colorbar(sc, ax=ax, shrink=0.5)  #
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax.scatter(points_tNet[0, :], points_tNet[1, :], points_tNet[2, :], c=points_tNet[2, :], s=10,
                     marker='o',
                     cmap="winter", alpha=0.5)
    ax.title.set_text('Output of tNet')
    plt.show()
    directory = 'figures/plots_train/'
    name = 'tNetOut_' + str(fileName) + '.png'
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()


def plot_hist(points, rdm_num):
    n_bins = 50
    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0].hist(points[0, 0, :], bins=n_bins)
    # axs[1].hist(points[0, 1, :], bins=n_bins)
    # axs[0].title.set_text('x')
    # axs[1].title.set_text('y')
    # plt.show()

    # 2D histogram
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(points[0, :], points[1, :], bins=n_bins)
    fig.colorbar(hist[3], ax=ax)
    # plt.show()
    directory = 'figures'
    name = 'hist_tNet_out_' + str(rdm_num)
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):

    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 0.2))
    labels = pc[:, 3] == 15
    # convert array of booleans to array of integers
    labels = labels.numpy().astype(int)
    cmap = plt.cm.get_cmap('winter')
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=10, marker='o',  cmap=cmap.reversed(), vmin=0, vmax=1)
    # plt.colorbar(sc)
    tag = str(n_clusters) + 'k-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)


def plot_2d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w):
    """

    :param pc: [2048, 11]
    :param writer_tensorboard:
    :param filename:
    :param i_w:
    """
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 3], s=10, marker='o',  cmap='Spectral')
    plt.colorbar(sc)
    tag = 'k-means_2Dxy_' + filename.split('/')[-1]
    # plt.title('PC')
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)


def get_weights_effective_num_of_samples(n_of_classes, beta, samples_per_cls):
    """The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes, samples_per_cls, power=1.0):
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes, samples_per_cls):
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights4class(sample_weighing_method, n_classes, samples_per_cls, beta=None):
    """

       :param sample weighing_method: str, options available: "EFS" "INS" "ISNS"
       :param n_classes: int, representing the total number of classes in the entire train set
       :param samples_per_cls: A python list of size [n_classes]
       :param labels: torch tensor of size [batch] containing labels
       :param beta: float,

       :return weights4class: torch. tensor of size [batch, n_classes]
    """
    if sample_weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif sample_weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif sample_weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif sample_weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    return weights4class


def get_weights4sample(weights4class, labels=None):
    """
    :param weights4class: torch tensor of size [batch, n_classes]
    :param labels:
    :return:
    """
    # one-hot encoding
    labels = labels.to('cpu').numpy()  # [batch] labels defines columns of non-zero elements
    one_hot = np.zeros((labels.size, 2))  # [batch, 2]
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1

    weights4samples = weights4class.to('cpu').unsqueeze(0)
    weights4samples = weights4samples.repeat(labels.shape[0], 1)
    weights4samples = torch.tensor(np.array(weights4samples * one_hot))
    weights4samples = weights4samples.sum(1).cpu()

    return weights4samples


def split4classif_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                              targets=[], task='classification', device='cuda'):
    """ split point cloud in windows of fixed size (n_points)
        and padd with 0 needed points to fill the window

    :param lengths:
    :param filenames:
    :param task:
    :param targets:
    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot:
    :param writer_tensorboard:

    :return pc_w: point cloud in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    count_p = 0
    j = 0
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]

        else:
            # padd with zeros to fill the window -> només aplica a la última finestra del batch
            points_needed = end_batch - points.shape[1]
            in_points = points[:, j * n_points:, :]
            if points_needed != n_points:
                # padd with zeros
                padd_points = torch.zeros(points.shape[0], points_needed, points.shape[2]).to(device)
                in_points = torch.cat((in_points, padd_points), dim=1)
                if task == 'segmentation':
                    extra_targets = torch.full((targets.shape[0], points_needed), -1).to(device)
                    targets = torch.cat((targets, extra_targets), dim=1)

        if plot:
            # write figure to tensorboard
            ax = plt.axes(projection='3d')
            pc_plot = in_points.cpu()
            sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10, marker='o',
                            cmap='Spectral')
            plt.colorbar(sc)
            tag = filenames[0].split('/')[-1]
            plt.title(
                'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                    targets[0].cpu().numpy()))
            writer_tensorboard.add_figure(tag, plt.gcf(), j)

        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points into tensor w
        pc_w = torch.cat([pc_w, in_points], dim=3)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets


def split4segmen_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                             targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets: [batch, n_samples]
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        # get padded points (padding target value = -1)
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]

            # transform targets into Long Tensor
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

            # write figure to tensorboard
            if plot:
                ax = plt.axes(projection='3d')
                pc_plot = in_points.cpu()
                sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                marker='o',
                                cmap='Spectral')
                plt.colorbar(sc)
                tag = filenames[0].split('/')[-1]
                plt.title(
                    'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                        in_targets[0].cpu().numpy()))
                writer_tensorboard.add_figure(tag, plt.gcf(), j)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4segmen_test(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                      targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets:
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)

    count_p = 0
    j = 0
    # loop over windows
    while j < 4:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]
        else:
            # get duplicated points from previous windows
            rdm_list = np.random.randint(0, points.shape[1], n_points)
            in_points = points[:, rdm_list, :]  # [2048, 11]
            in_targets = targets[:, rdm_list]  # [2048]

        # transform targets into Long Tensor
        in_targets = torch.LongTensor(in_targets.cpu()).to(device)
        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points and targets into tensor w
        pc_w = torch.cat((pc_w, in_points), dim=3)
        targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4cls_kmeans(o_points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[],
                     targets=torch.Tensor(), duplicate=True):
    """ split point cloud in windows of fixed size (n_points) with k-means
        Fill empty windows with duplicate points of previous windows
        Number of points must be multiple of n_points, so points left over are removed

        :param o_points: input point cloud [batch, n_samples, dims]
        :param n_points: number of points
        :param plot: bool set to True for plotting windows
        :param writer_tensorboard:
        :param filenames: []
        :param targets: [batch, w_len]
        :param duplicate: bool

        :return pc_w: tensor containing point cloud in windows of fixed size [b, 2048, dims, w_len]
        :return targets_w: tensor of targets [b, w_len]

    """

    # o_points = o_points.to('cpu')

    # if point cloud is larger than n_points we cluster them with k-means
    if o_points.shape[1] > n_points:

        pc_batch = torch.FloatTensor()
        targets_batch = torch.LongTensor()

        if o_points.shape[1] % n_points != 0:
            # Number of points must be multiple of n_points, so points left over are removed
            o_points = o_points[:, :n_points * (o_points.shape[1] // n_points), :]

        K_clusters = int(np.floor(o_points.shape[1] / n_points))
        clf = KMeansConstrained(n_clusters=K_clusters, size_min=n_points, size_max=n_points, random_state=0)

        # loop over batches
        for b in progressbar(range(o_points.shape[0]), redirect_stdout=True):
            # tensor for points per window
            pc_w = torch.FloatTensor()

            # todo decide how many features get for clustering
            i_f = [4, 5, 6, 7, 8, 9]  # x,y,z,label,I,R,G,B,NIR,NDVI
            clusters = clf.fit_predict(o_points[b, :, i_f].numpy())  # array of ints -> indices to each of the windows

            # loop over clusters
            for c in range(K_clusters):
                ix_cluster = np.where(clusters == c)
                # sample and get all features again
                in_points = o_points[b, ix_cluster, :]  # [batch, 2048, 11]

                # get position of in_points where all features are 0
                i_bool = torch.all(in_points == 0, dim=2).view(-1)
                # if there are padding points in the cluster
                if True in i_bool:
                    added_p = True
                    points_needed = int(sum(i_bool))
                    if duplicate:
                        # get duplicated random points
                        first_points = in_points[:, ~i_bool, :]
                        rdm_list = np.random.randint(0, n_points, points_needed)

                        in_points = o_points[b, rdm_list, :].view(1, points_needed, 11)
                        # concat points if not all points are padding points
                        if first_points.shape[1] > 0:
                            in_points = torch.cat([first_points, in_points], dim=1)
                else:
                    added_p = False

                in_points = torch.unsqueeze(in_points, dim=3)  # [1, 2048, 11, 1]
                # concat points of cluster
                pc_w = torch.cat((pc_w, in_points), dim=3)

                if int(targets[b, 0]) == 1 or b == 0:  # if there is a tower
                    # write figure to tensorboard
                    if plot:
                        ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                        pc_plot = in_points
                        sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                        marker='o',
                                        cmap='Spectral')
                        plt.colorbar(sc)
                        tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                        plt.title('PC size: ' + str(o_points.shape[1]) + ' added P: ' + str(added_p))
                        writer_tensorboard.add_figure(tag, plt.gcf(), c)

                        if c == 4:
                            ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                            sc = ax.scatter(o_points[b, :, 0], o_points[b, :, 1], o_points[b, :, 2],
                                            c=o_points[b, :, 3],
                                            s=10,
                                            marker='o',
                                            cmap='Spectral')
                            plt.colorbar(sc)
                            tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                            plt.title('original PC size: ' + str(o_points.shape[1]))
                            writer_tensorboard.add_figure(tag, plt.gcf(), c)

            # concat batch
            pc_batch = torch.cat((pc_batch, pc_w), dim=0)
            targets_batch = torch.cat((targets_batch, targets[b, 0].unsqueeze(0)), dim=0)

        # broadcast targets_batch to shape [batch, w_len]
        targets_batch = targets_batch.unsqueeze(1)
        targets_batch = targets_batch.repeat(1, targets.shape[1])

    # if point cloud is equal n_points
    else:
        pc_batch = o_points
        targets_batch = targets

    return pc_batch, targets_batch


def split4cls_rdm(points, n_points=2048, targets=[], device='cuda', duplicate=True):
    """ Random split for classification
        split point cloud in windows of fixed size (n_points)
        check batches with padding (-1) and fill windows with duplicate points of previous windows

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points in window
    :param targets: [b, w_len]
    :param device: 'cpu' or 'cuda'
    :param duplicate: bool

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    points = points.cpu()

    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j].cpu()  # [batch, 1]

            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # get empty batches
                batches_null = (in_targets == -1).numpy()
                if duplicate:
                    # get duplicated points from previous window
                    rdm_list = np.random.randint(0, end_batch - n_points, n_points)
                    copy_points = points[:, rdm_list, :]
                    extra_points = copy_points[batches_null, :, :]
                    extra_points = extra_points.view(-1, n_points, 11)
                    in_points = torch.cat((in_points[~ batches_null, :, :], extra_points), dim=0)
                    extra_targets = targets[batches_null, 0]
                    in_targets = torch.cat((in_targets[~ batches_null].to(device), extra_targets), dim=0)
                else:
                    # padd with 0
                    in_points[batches_null, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)

            in_points = torch.unsqueeze(in_points, dim=3).to(device)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3).to(device)
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_targets = torch.unsqueeze(in_targets, dim=1)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def save_checkpoint_rnn(name, task, epoch, epochs_since_improvement, cls_model, seg_model, optimizer, accuracy,
                        batch_size, learning_rate, number_of_points, weighing_method):
    if task == 'segmentation':
        seg_model_state = seg_model.state_dict()
    else:
        seg_model_state = None

    state = {
        'cls_model': cls_model.state_dict(),
        'seg_model': seg_model_state,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': number_of_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method
    }
    filename = 'checkpoint_' + name + '.pth'

    torch.save(state, 'pointNet/checkpoints/' + filename)


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                    learning_rate, n_points, weighing_method):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method
    }
    filename = 'checkpoint_' + name + '.pth'

    torch.save(state, 'pointNet/checkpoints/' + filename)


def adjust_learning_rate(optimizer, shrink_factor=0.1):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
