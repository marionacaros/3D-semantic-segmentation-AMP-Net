import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
import os
from torch.utils.tensorboard import SummaryWriter
from pointNet.datasets import LidarDatasetExpanded
from pointNet.model.light_pointnet_256 import SegmentationPointNet
# from pointNet.model.pointnet import SegmentationPointNet
from utils.utils import *
from utils.get_metrics import *
import logging
import datetime
# from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable

# warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

GLOBAL_FEAT_SIZE = 256
NUM_CLASES = 5


def train(
        dataset_folder,
        path_list_files,
        output_folder,
        n_points,
        batch_size,
        epochs,
        learning_rate,
        number_of_workers,
        model_checkpoint,
        c_sample):
    start_time = time.time()
    logging.info(f"Constrained sampling: {c_sample}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/segmentation/'

    # Datasets train / val / test
    with open(os.path.join(path_list_files, 'train_seg_files.txt'), 'r') as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path_list_files, 'val_seg_files.txt'), 'r') as f:
        val_files = f.read().splitlines()

    NAME = 'Base100_' + str(GLOBAL_FEAT_SIZE)

    writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_train' + NAME)
    writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_val' + NAME)
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # Initialize datasets
    train_dataset = LidarDatasetExpanded(dataset_folder=dataset_folder,
                                 task='segmentation', number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=True)
    val_dataset = LidarDatasetExpanded(dataset_folder=dataset_folder,
                               task='segmentation', number_of_points=n_points,
                               files=val_files,
                               fixed_num_points=True)

    # logging.info(f'Towers PC in train: {train_dataset.len_towers}')
    # logging.info(f'Landscape PC in train: {train_dataset.len_landscape}')
    # logging.info(
    #     f'Proportion towers/landscape: {round((train_dataset.len_towers / (train_dataset.len_towers + train_dataset.len_landscape)) * 100, 3)}%')
    # logging.info(f'Towers PC in val: {val_dataset.len_towers}')
    # logging.info(f'Landscape PC in val: {val_dataset.len_landscape}')
    # logging.info(
    #     f'Proportion towers/landscape: {round((val_dataset.len_towers / (val_dataset.len_towers + val_dataset.len_landscape)) * 100, 3)}%')
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

    pointnet = SegmentationPointNet(num_classes=NUM_CLASES, point_dimension=3)
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
    optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate)
    c_weights = torch.FloatTensor([1, 2, 2, 1, 1]).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1)

    scheduler_pointnet = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[50, 100, 300],  # List of epoch indices
                                     gamma=0.5)  # Multiplicative factor of learning rate decay

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
            'other_towers_train': [],
            'other_towers_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
            'cables_train': [],
            'cables_val': []
        }
        last_epoch = -1

        # if epochs_since_improvement == 10 or epoch == 15:
        #     adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_train, True,
                                                             epoch, last_epoch)
            preds=preds.view(-1)
            targets=targets.view(-1)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics, 'segmentation')

            # Segmentation labels:
            # 0 -> background (other classes we're not interested)
            # 1 -> tower
            # 2 -> cables
            # 3 -> low vegetation
            # 4 -> high vegetation
            iou['bckg_train'].append(get_iou_obj(targets, preds, 0))
            iou['tower_train'].append(get_iou_obj(targets, preds, 1))
            iou['cables_train'].append(get_iou_obj(targets, preds, 2))
            iou['low_veg_train'].append(get_iou_obj(targets, preds, 3))
            iou['high_veg_train'].append(get_iou_obj(targets, preds, 4))

            # tensorboard
            ce_train_loss.append(metrics['ce_loss'].cpu().item())
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        # --------------------------------------------- val loop ---------------------------------------------
        scheduler_pointnet.step()

        with torch.no_grad():
            first_batch = True
            for data in val_dataloader:
                metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_val, False,
                                                                 epoch, last_epoch, first_batch)
                first_batch = False

                metrics = get_accuracy(preds, targets, metrics, 'segmentation')

                iou['bckg_val'].append(get_iou_obj(targets, preds, 0))
                iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                iou['cables_val'].append(get_iou_obj(targets, preds, 2))
                iou['low_veg_val'].append(get_iou_obj(targets, preds, 3))
                iou['high_veg_val'].append(get_iou_obj(targets, preds, 4))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])
                epoch_val_acc_w.append(metrics['accuracy_w'])

                # targets = targets.cpu().numpy()
                # n_samples = len(targets)
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
        # if task == 'segmentation':
        writer_train.add_scalar('_iou_tower', np.mean(iou['tower_train']), epoch)
        writer_val.add_scalar('_iou_tower', np.mean(iou['tower_val']), epoch)
        writer_train.add_scalar('_iou_background', np.mean(iou['bckg_train']), epoch)
        writer_val.add_scalar('_iou_background', np.mean(iou['bckg_val']), epoch)
        writer_train.add_scalar('_iou_low_veg', np.mean(iou['low_veg_train']), epoch)
        writer_val.add_scalar('_iou_low_veg', np.mean(iou['low_veg_val']), epoch)
        # writer_train.add_scalar('_iou_med_veg', np.mean(iou['med_veg_train']), epoch)
        # writer_val.add_scalar('_iou_med_veg', np.mean(iou['med_veg_val']), epoch)
        writer_train.add_scalar('_iou_high_veg', np.mean(iou['high_veg_train']), epoch)
        writer_val.add_scalar('_iou_high_veg', np.mean(iou['high_veg_val']), epoch)
        writer_train.add_scalar('_iou_cables', np.mean(iou['cables_train']), epoch)
        writer_val.add_scalar('_iou_cables', np.mean(iou['cables_val']), epoch)
        # writer_train.add_scalar('_iou_other_towers', np.mean(iou['other_towers_train']), epoch)
        # writer_val.add_scalar('_iou_other_towers', np.mean(iou['other_towers_val']), epoch)
        # elif task == 'classification':
        #     writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        #     writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            name = now.strftime("%m-%d-%H:%M") + 'seg' + NAME
            save_checkpoint(name, epoch, epochs_since_improvement, pointnet,
                            optimizer, metrics['accuracy'],
                            batch_size, learning_rate, n_points)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1
        if epochs_since_improvement > 100:
            exit()

    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, last_epoch=0, first_batch_val=False):
    """

    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    pc = pc.data.numpy()
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.Tensor(pc)
    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # PointNet model
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
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/dades/LIDAR/towers_detection/datasets/towers_100x100_cls')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/RGBN_100x100')
    parser.add_argument('--output_folder', type=str, help='output folder', default='results')
    parser.add_argument('--number_of_points', type=int, default=4096, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=10, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--c_sample', type=bool, default=False, help='use constrained sampling')

    args = parser.parse_args()

    train(args.dataset_folder,
          args.path_list_files,
          args.output_folder,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.number_of_workers,
          args.model_checkpoint,
          args.c_sample)
