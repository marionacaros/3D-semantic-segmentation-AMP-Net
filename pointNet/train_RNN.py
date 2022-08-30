import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from progressbar import progressbar
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import LidarDataset
from model.pointnetRNN import RNNClassificationPointNet, RNNSegmentationPointNet
import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from utils import *
from collate_fns import *
from prettytable import PrettyTable
from collections import Counter

# warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    print(f"cuda available")
    device = 'cuda'
else:
    print(f"cuda not available")
    device = 'cpu'


def train_gru(task,
              dataset_folder,
              path_list_files,
              output_folder,
              n_points,
              batch_size,
              epochs,
              learning_rate,
              weighing_method,
              beta,
              number_of_workers,
              model_checkpoint,
              c_sample=False):
    start_time = time.time()
    print(f"Weighing method: {weighing_method}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'
    # if not os.path.isdir(output_folder):
    #     os.mkdir(output_folder)

    # Datasets train / val / test
    if task == 'classification':
        name = 'files'
    elif task == 'segmentation':
        name = 'seg_files'

    with open(os.path.join(path_list_files, 'train_' + name + '.txt'), 'r') as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path_list_files, 'val_' + name + '.txt'), 'r') as f:
        val_files = f.read().splitlines()
    print(f'Dataset folder: {dataset_folder}')

    NAME = 'GRU128w5rep'

    # Datasets train / val / test
    if task == 'classification':
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_train' + NAME)
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_val' + NAME)
        print(f"Tensorboard runs: {writer_train.get_logdir()}")

        # padding patch function
        collate_fn_padd = collate_classif_padd

    elif task == 'segmentation':
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_train' + NAME)
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_val' + NAME)
        print(f"Tensorboard runs: {writer_train.get_logdir()}")

        collate_fn_padd = collate_segmen_padd

    # Initialize datasets
    train_dataset = LidarDataset(dataset_folder=dataset_folder,
                                 task=task, number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=c_sample)
    val_dataset = LidarDataset(dataset_folder=dataset_folder,
                               task=task, number_of_points=n_points,
                               files=val_files,
                               fixed_num_points=c_sample)

    print(f'Towers PC in train: {train_dataset.len_towers}')
    print(f'Landscape PC in train: {train_dataset.len_landscape}')
    print(
        f'Proportion towers/landscape: {round((train_dataset.len_towers / (train_dataset.len_towers + train_dataset.len_landscape)) * 100, 3)}%')
    print(f'Towers PC in val: {val_dataset.len_towers}')
    print(f'Landscape PC in val: {val_dataset.len_landscape}')
    print(
        f'Proportion towers/landscape: {round((val_dataset.len_towers / (val_dataset.len_towers + val_dataset.len_landscape)) * 100, 3)}%')
    print(f'Samples for training: {len(train_dataset)}')
    print(f'Samples for validation: {len(val_dataset)}')
    print(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True,
                                                   collate_fn=collate_fn_padd)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True,
                                                 collate_fn=collate_fn_padd)
    # classification model with GRU
    cls_model = RNNClassificationPointNet(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                          hidden_size=128,
                                          point_dimension=train_dataset.POINT_DIMENSION)
    cls_model.to(device)
    seg_model = None

    if task == 'segmentation':
        # segmentation model with GRU
        seg_model = RNNSegmentationPointNet(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES)
        seg_model.to(device)

    best_vloss = 1_000_000.
    epochs_since_improvement = 0
    c_weights = None

    if task == 'classification':
        c_weights = get_weights4class(weighing_method,
                                      n_classes=2,
                                      samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape,
                                                       train_dataset.len_towers + val_dataset.len_towers],
                                      beta=beta).to(device)
    # masked loss
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, ignore_index=-1, reduce=None, reduction='none')
    # optimizer
    optimizer = optim.Adam(cls_model.parameters(), lr=learning_rate)

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        cls_model.load_state_dict(checkpoint['model'])  # cls_model
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # adjust_learning_rate(optimizer, learning_rate)

    # print cls_model and parameters
    # INPUT_SHAPE = (7, 2000)
    # summary(cls_model, INPUT_SHAPE)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in cls_model.named_parameters():
        # if not parameter.requires_grad: continue

        # Freeze all layers
        # parameter.requires_grad = False
        params = parameter.numel()
        # print(f'{name} {parameter}')
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        ce_train_loss = []
        epoch_train_acc = []
        targets_pos = []
        targets_neg = []
        epoch_val_loss = []
        epoch_val_acc = []
        detected_positive = []
        detected_negative = []
        ious_tower_train = []
        ious_tower_val = []

        if epochs_since_improvement == 10:
            adjust_learning_rate(optimizer, 0.5)
        # elif epoch == 5:
        #     adjust_learning_rate(optimizer, 0.5)
        # elif epoch == 15:
        #     adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            total_loss, nll_loss, accuracy, targets, pc_preds, c_weights, iou_tower = train_loop(data,
                                                                                      optimizer,
                                                                                      ce_loss,
                                                                                      cls_model,
                                                                                      seg_model,
                                                                                      n_points,
                                                                                      writer_train,
                                                                                      task,
                                                                                      True,
                                                                                      weighing_method,
                                                                                      beta,
                                                                                      c_weights)

            # tensorboard
            ce_train_loss.append(nll_loss.cpu().item())
            epoch_train_loss.append(total_loss.cpu().item())
            epoch_train_acc.append(accuracy)
            ious_tower_train.append(iou_tower)

        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():
            for data in val_dataloader:
                total_loss, nll_loss, accuracy, targets, pc_preds, c_weights, iou_tower = train_loop(data, optimizer,
                                                                                          ce_loss,
                                                                                          cls_model,
                                                                                          seg_model,
                                                                                          n_points,
                                                                                          writer_val,
                                                                                          task,
                                                                                          False,
                                                                                          weighing_method,
                                                                                          beta,
                                                                                          c_weights)
                ious_tower_val.append(iou_tower)

            # tensorboard
            epoch_val_loss.append(nll_loss.cpu().item())
            targets = targets.view(-1).cpu().numpy()
            mask = targets != [-1] * len(targets)
            targets = targets[mask]
            pc_preds = pc_preds[mask]

            targets_pos.append((targets == np.ones(len(targets))).sum() / len(targets))
            targets_neg.append((targets == np.zeros(len(targets))).sum() / len(targets))
            detected_negative.append((np.array(pc_preds) == np.zeros(len(pc_preds))).sum() / len(targets))
            detected_positive.append((np.array(pc_preds) == np.ones((len(pc_preds)))).sum() / len(targets))
            epoch_val_acc.append(accuracy)

        # ------------------------------------------------------------------------------------------------------
        # Tensorboard
        writer_train.add_scalar('loss', np.mean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.mean(epoch_val_loss), epoch)
        writer_train.add_scalar('loss_NLL', np.mean(ce_train_loss), epoch)
        writer_val.add_scalar('loss_NLL', np.mean(epoch_val_loss), epoch)
        writer_train.add_scalar('mean_detected_positive', np.mean(targets_pos), epoch)
        writer_val.add_scalar('mean_detected_positive', np.mean(detected_positive), epoch)
        writer_train.add_scalar('mean_detected_negative', np.mean(targets_neg), epoch)
        writer_val.add_scalar('mean_detected_negative', np.mean(detected_negative), epoch)
        writer_train.add_scalar('accuracy', np.mean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.mean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_train.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('_iou_tower', np.mean(ious_tower_train), epoch)
        writer_val.add_scalar('_iou_tower', np.mean(ious_tower_val), epoch)

        writer_train.add_scalar('c_weights', c_weights[1].cpu(), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            if task == 'classification':
                name = now.strftime("%m-%d-%H:%M") + '_clsGRU' + NAME
            elif task == 'segmentation':
                name = now.strftime("%m-%d-%H:%M") + '_segGRU' + NAME
            save_checkpoint(name, epoch, epochs_since_improvement, cls_model, seg_model, optimizer, accuracy,
                            batch_size,
                            learning_rate, n_points, weighing_method)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1
        if epochs_since_improvement > 40:
            exit()

    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


def plot_pc(in_points):
    # write figure to tensorboard
    ax = plt.axes(projection='3d')
    pc_plot = in_points.cpu()
    # sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
    #                 marker='o',
    #                 cmap='Spectral')
    # plt.colorbar(sc)
    # tag = filenames[0].split('/')[-1]
    # plt.title(
    #     'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
    #         in_targets[0].cpu().numpy()))
    # writer_tensorboard.add_figure(tag, plt.gcf(), j)


def train_loop(data, optimizer, ce_loss, cls_model,
               seg_model=None,
               n_points=2048,
               w_tensorboard=None,
               task='classification',
               train=True,
               weighing_method='EFS',
               beta=0.999,
               c_weights=[]):
    """

    :param data: tuple with (points, targets, filenames, lengths, len_w)
    :param optimizer: optimizer
    :param ce_loss: CrossEntropy loss
    :param cls_model: classification model
    :param seg_model: segmentation model
    :param n_points: number of points per window
    :param w_tensorboard: tensorboard writer
    :param task: 'classification' or 'segmentation'
    :param train: set to True to perform backward propagation
    :param beta:
    :param weighing_method:

    :return:
    """
    points, targets, filenames, lengths, len_w = data
    # classifications shapes: [b, n_samples, dims], [b, w_len], [b], [b], [b]
    # segmentation shapes : [b, n_samples, dims], [b, n_samples], [b], [b], [b]
    batch_size = points.shape[0]

    points, targets = points.to(device), targets.to(device)  # ([batch, 18802, 11]
    len_w = len_w.to(device)

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    # init hidden
    hidden = cls_model.initHidden(points, device)

    if train:
        cls_model = cls_model.train()
        if task == 'segmentation':
            seg_model = seg_model.train()
    else:
        cls_model = cls_model.eval()
        if task == 'segmentation':
            seg_model = seg_model.eval()

    # split into windows of fixed size
    if task == 'segmentation':
        pc_w, targets = split4segmen_point_cloud(points, n_points,
                                                 plot=False,
                                                 writer_tensorboard=w_tensorboard,
                                                 filenames=filenames,
                                                 lengths=lengths,
                                                 targets=targets,
                                                 device=device,
                                                 duplicate=True)
        # pc_w shape: [b, 2048, dims, w_len]

    if task == 'classification':
        pc_w, targets = split4cls_point_cloud(points, n_points,
                                              targets=targets,
                                              device=device,
                                              duplicate=True)

    if task == 'segmentation':
        # get weights for imbalanced loss
        targets_batch = targets.view(-1).cpu()
        points_tower = (np.array(targets_batch) == np.ones(len(targets_batch))).sum()
        points_landscape = (np.array(targets_batch) == np.zeros(len(targets_batch))).sum()
        if not points_tower:
            points_tower = 100
            points_landscape = 4000

        c_weights = get_weights4class(weighing_method,
                                      n_classes=2,
                                      samples_per_cls=[points_landscape, points_tower],
                                      beta=beta).to(device)
        # define loss with weights
        ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, ignore_index=-1, reduce=None, reduction='mean')

    pc_preds = []
    # forward pass
    for w in range(pc_w.shape[3]):

        # get window
        in_points = pc_w[:, :, :, w]
        logits, hidden, feat_transf, local_features = cls_model(in_points, hidden, get_preds=True)
        # [b, 2] [2,b,128] [b,64,64] [b,n_p,64]

        if task == 'segmentation':
            logits = seg_model(hidden, local_features)  # [b, 2048, 2]
            logits = logits.reshape(-1, logits.shape[2])
            targets_w = targets[:, w * n_points: (w + 1) * n_points]

            ce_loss_batch = ce_loss(logits, targets_w.reshape(-1))
            # ce_loss_batch = ce_loss_batch.view(batch_size, n_points)
            # ce_loss_batch = ce_loss_batch.sum(1) / (len_w * n_points)  # [b] get mean
            ce_loss_batch = ce_loss_batch.view(-1, 1)

        elif task == 'classification':
            ce_loss_batch = ce_loss(logits, targets[:, w]).view(-1, 1)  # [2, 1]

        # get accuracy
        probs = F.log_softmax(logits.detach().cpu(), dim=1)
        preds = probs.data.max(1)[1].view(-1, 1)
        if w == 0:
            preds_w = preds
            ce_loss_all = ce_loss_batch
            targets_w_cat = targets_w.reshape(-1, 1)
        else:
            # concatenate predictions of windows
            preds_w = torch.cat((preds_w, preds), 1)
            # concatenate losses of windows
            ce_loss_all = torch.cat((ce_loss_all, ce_loss_batch), 1)  # [b, w]  segmen: [b, w]
            targets_w_cat = torch.cat((targets_w_cat, targets_w.reshape(-1, 1)), 1)

    if task == 'classification':
        # loop over batches to get value most predicted != -1 among all windows
        for w in preds_w.numpy():
            c = Counter(w[w != -1])
            value, count = c.most_common()[0]
            pc_preds.append(value)
        corrects = torch.eq(torch.LongTensor(pc_preds), targets[:, 0].detach().cpu())
        accuracy = (corrects.sum() / batch_size)

        identity = torch.eye(feat_transf.shape[-1]).to(device)  # [64, 64]
        regularization_loss = torch.norm(identity - torch.bmm(feat_transf, feat_transf.transpose(2, 1)))

        loss_batch = ce_loss_all.sum(1) / len_w
        ce_loss_mean = loss_batch.sum() / batch_size

    elif task == 'segmentation':
        pc_preds = preds_w.view(-1)
        targets = targets_w_cat.view(-1)
        # get metrics
        targets_m = targets.detach().cpu()
        corrects = torch.eq(torch.LongTensor(pc_preds), targets_m)
        # all_neg = (np.array(targets.view(-1)) == np.zeros(len(targets))).sum()
        all_positive = (np.array(targets_m) == np.ones(len(targets_m))).sum()  # TP + FN
        detected_positive = (np.array(pc_preds) == np.ones(len(targets_m)))
        detected_negative = (np.array(pc_preds) == np.zeros(len(targets_m)))
        tp = np.logical_and(corrects, detected_positive).sum()
        tn = np.logical_and(corrects, detected_negative).sum()
        fp = np.array(detected_positive).sum() - tp
        fn = np.array(detected_negative).sum() - tn
        iou_tower = tp / (all_positive + fp)
        accuracy = (corrects.sum() / (batch_size * pc_w.shape[3] * n_points))

        identity = torch.eye(feat_transf.shape[-1]).to(device)  # [64, 64]
        regularization_loss = torch.norm(identity - torch.bmm(feat_transf, feat_transf.transpose(2, 1)))

        ce_loss_mean = ce_loss_all.sum() / batch_size

    if train:
        loss = ce_loss_mean + 0.001 * regularization_loss
        loss.backward()
        optimizer.step()
    #  validation
    else:
        loss = ce_loss_mean

    return loss, ce_loss_mean, accuracy, targets, pc_preds, c_weights, iou_tower


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--path_list_files', type=str,
                        default='pointNet/data/train_test_files/RGBN',
                        help='output folder')
    parser.add_argument('--output_folder', type=str, default='pointNet/results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--beta', type=float, default=0.999, help='model checkpoint path')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--c_sample', type=bool, default=False, help='use constrained sampling')

    args = parser.parse_args()

    # logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    #                     level=logging.INFO,
    #                     datefmt='%Y-%m-%d %H:%M:%S')

    train_gru(args.task,
              args.dataset_folder,
              args.path_list_files,
              args.output_folder,
              args.number_of_points,
              args.batch_size,
              args.epochs,
              args.learning_rate,
              args.weighing_method,
              args.beta,
              args.number_of_workers,
              args.model_checkpoint,
              args.c_sample)
