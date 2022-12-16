import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
import os
from torch.utils.tensorboard import SummaryWriter
from pointNet.datasets import LidarDataset
from pointNet.model.light_pointnet_IGBVI import SegmentationPointNet_IGBVI
from utils.utils import *
from utils.get_metrics import *
import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

GLOBAL_FEAT_SIZE = 256


def train(
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
        c_sample):
    start_time = time.time()
    c_weights = torch.FloatTensor([0.2, 0.2, 0.2, 0.2, 0.2]).to(device)

    # logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Constrained sampling: {c_sample}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/segmentation/'

    # Datasets train / val / test
    with open(os.path.join(path_list_files, 'train_seg_files.txt'), 'r') as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path_list_files, 'val_seg_files.txt'), 'r') as f:
        val_files = f.read().splitlines()

    NAME = 'base80x80' + str(GLOBAL_FEAT_SIZE)

    writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_train' + NAME)
    writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_val' + NAME)
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # Initialize datasets
    train_dataset = LidarDataset(dataset_folder=dataset_folder,
                                 task='segmentation', number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=True,
                                 c_sample=c_sample)
    val_dataset = LidarDataset(dataset_folder=dataset_folder,
                               task='segmentation', number_of_points=n_points,
                               files=val_files,
                               fixed_num_points=True,
                               c_sample=c_sample)

    logging.info(f'Towers PC in train: {train_dataset.len_towers}')
    logging.info(f'Landscape PC in train: {train_dataset.len_landscape}')
    logging.info(
        f'Proportion towers/landscape: {round((train_dataset.len_towers / (train_dataset.len_towers + train_dataset.len_landscape)) * 100, 3)}%')
    logging.info(f'Towers PC in val: {val_dataset.len_towers}')
    logging.info(f'Landscape PC in val: {val_dataset.len_landscape}')
    logging.info(
        f'Proportion towers/landscape: {round((val_dataset.len_towers / (val_dataset.len_towers + val_dataset.len_landscape)) * 100, 3)}%')
    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)

    pointnet = SegmentationPointNet_IGBVI(num_classes=6, point_dimension=train_dataset.POINT_DIMENSION)
    pointnet.to(device)

    # print model and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in pointnet.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    logging.info(f"Total Trainable Params: {total_params}")

    # loss
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate)

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        pointnet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        ce_train_loss = []
        epoch_train_acc = []
        epoch_train_acc_w = []
        targets_pos = []
        targets_neg = []
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
        detected_positive = []
        detected_negative = []
        iou = {
            'tower_train': [],
            'tower_val': [],
            'low_veg_train': [],
            'low_veg_val': [],
            'med_veg_train': [],
            'med_veg_val': [],
            'high_veg_train': [],
            'high_veg_val': [],
            'bckg_train': [],
            'bckg_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
            'cables_train': [],
            'cables_val': []
        }
        last_epoch = -1

        if epochs_since_improvement == 10:
            adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_train, True,
                                                             c_weights, epoch, last_epoch)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics, 'segmentation', c_weights)

            # Segmentation labels:
            # 0 -> background (other classes we're not interested)
            # 1 -> tower
            # 2 ->
            # 3 -> low vegetation
            # 4 -> medium vegetation
            # 5 -> high vegetation
            iou['bckg_train'].append(get_iou_obj(targets, preds, 0))
            iou['tower_train'].append(get_iou_obj(targets, preds, 1))
            iou['cables_train'].append(get_iou_obj(targets, preds, 2))
            iou['low_veg_train'].append(get_iou_obj(targets, preds, 3))
            iou['med_veg_train'].append(get_iou_obj(targets, preds, 4))
            iou['high_veg_train'].append(get_iou_obj(targets, preds, 5))

            # tensorboard
            ce_train_loss.append(metrics['ce_loss'].cpu().item())
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])
            epoch_train_acc_w.append(metrics['accuracy_w'])

        with torch.no_grad():
            first_batch = True
            for data in val_dataloader:
                metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_val, False,
                                                                 c_weights, epoch, last_epoch, first_batch)
                first_batch = False

                metrics = get_accuracy(preds, targets, metrics, 'segmentation', c_weights)

                iou['bckg_val'].append(get_iou_obj(targets, preds, 0))
                iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                iou['cables_val'].append(get_iou_obj(targets, preds, 2))
                iou['low_veg_val'].append(get_iou_obj(targets, preds, 3))
                iou['med_veg_val'].append(get_iou_obj(targets, preds, 4))
                iou['high_veg_val'].append(get_iou_obj(targets, preds, 5))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])
                epoch_val_acc_w.append(metrics['accuracy_w'])

                targets = targets.cpu().numpy()
                n_samples = len(targets)
                # targets_pos.append((targets == np.ones(n_samples)).sum() / n_samples)
                # targets_neg.append((targets == np.zeros(n_samples)).sum() / n_samples)
                # detected_negative.append((np.array(preds) == np.zeros(n_samples)).sum() / n_samples)
                # detected_positive.append((np.array(preds) == np.ones(n_samples)).sum() / n_samples)

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
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # writer_train.add_scalar('c_weights', c_weights[1].cpu(), epoch)
        # writer_val.add_scalar('c_weights', c_weights[0].cpu(), epoch)
        # if task == 'segmentation':
        writer_train.add_scalar('_iou_tower', np.mean(iou['tower_train']), epoch)
        writer_val.add_scalar('_iou_tower', np.mean(iou['tower_val']), epoch)
        writer_train.add_scalar('_iou_background', np.mean(iou['bckg_train']), epoch)
        writer_val.add_scalar('_iou_background', np.mean(iou['bckg_val']), epoch)
        writer_train.add_scalar('_iou_low_veg', np.mean(iou['low_veg_train']), epoch)
        writer_val.add_scalar('_iou_low_veg', np.mean(iou['low_veg_val']), epoch)
        writer_train.add_scalar('_iou_med_veg', np.mean(iou['med_veg_train']), epoch)
        writer_val.add_scalar('_iou_med_veg', np.mean(iou['med_veg_val']), epoch)
        writer_train.add_scalar('_iou_high_veg', np.mean(iou['high_veg_train']), epoch)
        writer_val.add_scalar('_iou_high_veg', np.mean(iou['high_veg_val']), epoch)
        writer_train.add_scalar('_iou_cables', np.mean(iou['cables_train']), epoch)
        writer_val.add_scalar('_iou_cables', np.mean(iou['cables_val']), epoch)
        # elif task == 'classification':
        #     writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        #     writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            # if task == 'classification':
            #     name = now.strftime("%m-%d-%H:%M") + '_cls' + NAME
            name = now.strftime("%m-%d-%H:%M") + '_seg' + NAME
            save_checkpoint(name, epoch, epochs_since_improvement, pointnet,
                            optimizer, metrics['accuracy'],
                            batch_size, learning_rate, n_points, weighing_method)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1
        if epochs_since_improvement > 40:
            exit()

    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               c_weights=torch.Tensor(), epoch=0, last_epoch=0, first_batch_val=False):
    """

    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy':[]}
    pc, targets, filenames = data

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()

    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # points = points.view(batch_size, n_points, -1).to(device)  # [batch, n_samples, dims]
    # targets = targets.view(batch_size, -1).to(device)  # [batch, n_samples]
    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]

    logits, feat_transform = pointnet(pc)

    # CrossEntropy loss
    metrics['ce_loss'] = ce_loss(logits, targets).view(-1, 1)  # [1, 1]
    targets_pc = targets.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])

    # plot predictions in Tensorboard
    # if epoch >= 0:
    #     if epoch != last_epoch or first_batch_val == True:
    #         preds_plot, targets_plot, mask = rm_padding(preds[0, :].cpu(), targets_pc[0, :])
    #         # Tensorboard
    #         plot_pc_tensorboard(pc[0, mask, :], targets_plot, w_tensorboard, 'b0_plot_targets', step=epoch)
    #         plot_pc_tensorboard(pc[0, mask, :], preds_plot, w_tensorboard, 'b0_plot_predictions', step=epoch)
    #
    #         last_epoch = epoch

    # compute regularization loss
    identity = torch.eye(feat_transform.shape[-1]).to(device)  # [64, 64]
    metrics['reg_loss'] = torch.norm(identity - torch.bmm(feat_transform, feat_transform.transpose(2, 1)))

    if train:
        metrics['loss'] = metrics['ce_loss'] + 0.001 * metrics['reg_loss']
        metrics['loss'].backward()
        optimizer.step()
    #  validation
    else:
        metrics['loss'] = metrics['ce_loss']

    return metrics, targets_pc, preds, last_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/RGBN_x10_80x80')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--beta', type=float, default=0.999, help='model checkpoint path')
    parser.add_argument('--number_of_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--c_sample', type=bool, default=False, help='use constrained sampling')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    train(args.dataset_folder,
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
