import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np


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


def plot_kmens_sequence_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):
    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 0.3))
    labels = pc[:, 3] == 15
    # convert array of booleans to array of integers
    labels = labels.numpy().astype(int)
    cmap = plt.cm.get_cmap('winter')
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=10, marker='o', cmap=cmap.reversed(), vmin=0, vmax=1)
    tag = str(n_clusters) + 'k-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)


def plot_pc_tensorboard(pc, labels, writer_tensorboard, tag, step):
    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 0.3))
    labels = labels.numpy().astype(int)
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.75, 6))
    newcolors[:1, :] = np.array([255/256, 165/256, 0/256, 1])  # orange
    newcolors[3:4, :] = np.array([102/256, 256/256, 178/256, 1])  # light green
    cmap = ListedColormap(newcolors)
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=7, marker='o', cmap=cmap, vmin=0, vmax=5)
    plt.colorbar(sc, fraction=0.02, pad=0.1)
    plt.title('point cloud' + str(len(pc)))
    fig = plt.gcf()
    fig.set_dpi(100)
    writer_tensorboard.add_figure(tag, fig, global_step=step)


def plot_pointcloud_with_labels(pc, labels, iou_tower, name, path_plot=''):

    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, max(pc[:, 2])))
    labels = labels.numpy().astype(int)
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.75, 6))
    newcolors[:1, :] = np.array([255 / 256, 165 / 256, 0 / 256, 1])  # orange
    newcolors[3:4, :] = np.array([102 / 256, 256 / 256, 178 / 256, 1])  # light green
    cmap = ListedColormap(newcolors)
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=5, marker='o', cmap=cmap, vmin=0, vmax=5)
    plt.colorbar(sc, fraction=0.02, pad=0.1)
    plt.title('Point cloud - ' + str(len(pc)) + ' p -' + ' IoU tower: ' + str(iou_tower))
    fig = plt.gcf()
    fig.set_dpi(100)
    if path_plot:
        plt.savefig(os.path.join(path_plot, name))


def plot_2d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w):
    """
    Plot sequence of K-means clusters in Tensorboard

    :param pc: [2048, 11]
    :param writer_tensorboard:
    :param filename:
    :param i_w:
    """
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 3], s=10, marker='o', cmap='Spectral')
    plt.colorbar(sc)
    tag = 'k-means_2Dxy_' + filename.split('/')[-1]
    # plt.title('PC')
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)
