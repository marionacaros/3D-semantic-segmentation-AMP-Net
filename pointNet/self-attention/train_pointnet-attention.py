import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from pointNet.datasets import LidarKmeansDataset
from pointNet.model.pointnetAtt import BasePointNet, ClassificationWithAttention, SegmentationWithAttention
import datetime
from pointNet.collate_fns import *
from prettytable import PrettyTable
from utils.utils_plot import *
from utils.get_metrics import *
from utils.utils import *
from torch.optim.lr_scheduler import MultiStepLR


if torch.cuda.is_available():
    print(f"cuda available")
    device = 'cuda'
else:
    print(f"cuda not available")
    device = 'cpu'

ATT_HEADS = 8
GLOBAL_FEAT_SIZE = 256


def train_att(task: str,
              dataset_folder: str,
              path_list_files: str,
              output_folder: str,
              n_points: int,
              batch_size: int,
              epochs: int,
              learning_rate: float,
              weighing_method: str = 'EFS',
              beta: float = 0.999,
              number_of_workers: int = 4,
              model_checkpoint: str = None):

    start_time = time.time()
    print(f"Weighing method: {weighing_method}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/segmentation/'
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

    NAME = 'ATT'+ 'g' + str(GLOBAL_FEAT_SIZE) + 'w100'+'xyz'

    # Datasets train / val / test
    if task == 'classification':
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_train' + NAME)
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_val' + NAME)
        print(f"Tensorboard runs: {writer_train.get_logdir()}")
        # padding batch function
        collate_fn = collate_seq_padd

    elif task == 'segmentation':
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_train' + NAME)
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_val' + NAME)
        print(f"Tensorboard runs: {writer_train.get_logdir()}")

        collate_fn = collate_seq_padd

    # Initialize datasets
    train_dataset = LidarKmeansDataset(dataset_folder=dataset_folder,
                                       task=task,
                                       number_of_points=n_points,
                                       files=train_files)
    val_dataset = LidarKmeansDataset(dataset_folder=dataset_folder,
                                     task=task,
                                     number_of_points=n_points,
                                     files=val_files)

    print(f'Samples for training: {len(train_dataset)}')
    print(f'Samples for validation: {len(val_dataset)}')
    print(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True,
                                                   collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True,
                                                 collate_fn=collate_fn)

    # ------------------------------------------- Models initialization -------------------------------------------

    pointnet = BasePointNet(point_dimension=3,
                            return_local_features=True,
                            global_feat_dim=GLOBAL_FEAT_SIZE,
                            device=device)

    if task == 'classification':
        att_net = ClassificationWithAttention(GLOBAL_FEAT_SIZE, ATT_HEADS, num_classes=5)
    elif task == 'segmentation':
        att_net = SegmentationWithAttention(GLOBAL_FEAT_SIZE, ATT_HEADS, num_classes=5,local_dim=64 )

    # Models to device
    pointnet.to(device)
    att_net.to(device)

    best_vloss = 1_000_000.
    epoch_ini = 0
    epochs_since_improvement = 0
    c_weights = torch.FloatTensor([1, 2, 2, 1, 1]).to(device)

    if task == 'classification':
        # todo add classes to samples_per_cls
        c_weights = get_weights4class(weighing_method,
                                      n_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                      samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape,
                                                       train_dataset.len_towers + val_dataset.len_towers],
                                      beta=beta).to(device)

    # weighted loss
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1)
    # optimizers
    optimizer_pointnet = optim.Adam(pointnet.parameters(), lr=learning_rate)
    optimizer_att = optim.Adam(att_net.parameters(), lr=learning_rate)

    # schedulers
    scheduler_pointnet = MultiStepLR(optimizer_pointnet,
                                     milestones=[150, 250, 350],  # List of epoch indices
                                     gamma=0.5)  # Multiplicative factor of learning rate decay
    scheduler_att = MultiStepLR(optimizer_att,
                                milestones=[150, 250, 350],  # List of epoch indices
                                gamma=0.5)  # Multiplicative factor of learning rate decay

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)

        pointnet.load_state_dict(checkpoint['base_pointnet'])
        att_net.load_state_dict(checkpoint['segmen_net'])
        batch_size = checkpoint['batch_size']
        learning_rate = checkpoint['lr']
        epoch_ini = checkpoint['epoch']

        optimizer_pointnet.load_state_dict(checkpoint['opt_pointnet'])
        optimizer_att.load_state_dict(checkpoint['opt_segmen'])

    # print model and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in pointnet.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    for name, parameter in att_net.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    for epoch in progressbar(range(epoch_ini, epochs), redirect_stdout=True):
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

        # if epoch < 300:
        #     if epochs_since_improvement == 50:
        #         adjust_learning_rate(optimizer_pointnet, 0.5)
        #         adjust_learning_rate(optimizer_att, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds, last_epoch = train_loop(data, optimizer_pointnet, optimizer_att, ce_loss,
                                                             pointnet, att_net, writer_train, task, True,
                                                             epoch, last_epoch)
            # discard padded windows to compute accuracy metrics
            preds, targets, _ = rm_padding(preds, targets)
            # compute accuracy
            metrics = get_accuracy(preds, targets, metrics, task, c_weights)

            if task == 'segmentation':
                # Segmentation labels:
                # 0 -> others (other classes we're not interested)
                # 1 -> tower
                # 2 -> power lines
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
            epoch_train_acc_w.append(metrics['accuracy_w'])
        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():
            first_batch = True
            for data in val_dataloader:
                metrics, targets, preds, last_epoch = train_loop(data, optimizer_pointnet, optimizer_att, ce_loss,
                                                                 pointnet, att_net, writer_val, task, False,
                                                                 epoch, last_epoch, first_batch)
                first_batch = False
                # discard padded windows to compute accuracy metrics
                preds, targets, _ = rm_padding(preds, targets)

                metrics = get_accuracy(preds, targets, metrics, task, c_weights)

                if task == 'segmentation':
                    iou['bckg_val'].append(get_iou_obj(targets, preds, 0))
                    iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                    iou['cables_val'].append(get_iou_obj(targets, preds, 2))
                    iou['low_veg_val'].append(get_iou_obj(targets, preds, 3))
                    iou['high_veg_val'].append(get_iou_obj(targets, preds, 4))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])
                epoch_val_acc_w.append(metrics['accuracy_w'])

                targets = targets.cpu().numpy()
                n_samples = len(targets)
                targets_pos.append((targets == np.ones(n_samples)).sum() / n_samples)
                targets_neg.append((targets == np.zeros(n_samples)).sum() / n_samples)
                detected_negative.append((np.array(preds) == np.zeros(n_samples)).sum() / n_samples)
                detected_positive.append((np.array(preds) == np.ones(n_samples)).sum() / n_samples)

        scheduler_pointnet.step()
        scheduler_att.step()
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
        writer_val.add_scalar('learning_rate', optimizer_pointnet.param_groups[0]['lr'], epoch)
        # writer_train.add_scalar('c_weights', c_weights[1].cpu(), epoch)
        # writer_val.add_scalar('c_weights', c_weights[0].cpu(), epoch)
        if task == 'segmentation':
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
        elif task == 'classification':
            writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
            writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            if task == 'classification':
                name = now.strftime("%m-%d-%H:%M") + '_cls' + NAME
            elif task == 'segmentation':
                name = now.strftime("%m-%d-%H:%M") + '_seg' + NAME

            save_checkpoint_segmen_model(name, task, epoch, epochs_since_improvement, pointnet, att_net,
                                         optimizer_pointnet,
                                         optimizer_att,
                                         metrics['accuracy'],
                                         batch_size, learning_rate, n_points, weighing_method)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1

    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer_pointnet, optimizer_att, ce_loss, pointnet, att_net,
               w_tensorboard=None, task='classification', train=True, epoch=0, last_epoch=0, first_batch_val=False):
    """
    :param data: tuple with (points, targets, filenames, lengths, len_w)
    :param optimizer_rnn:
    :param optimizer_pred:
    :param ce_loss: CrossEntropy loss
    :param gru_model:
    :param pointnet:
    :param w_tensorboard: tensorboard writer
    :param task: 'classification' or 'segmentation'
    :param train: set to True to perform backward propagation
    :param c_weights:
    :param first_batch_val:
    :param last_epoch:
    :param epoch:

    :return:
    metrics, targets, pc_preds, last_epoch
    """

    metrics = {}
    pc_clusters, targets, filenames, centroids = data
    centroids = centroids.to(device)

    # classifications shapes: [b, n_samples, dims, w_len], [b, w_len], [b]
    # segmentation shapes : [b, n_samples, dims, w_len], [b, n_samples], [b]
    # targets classif: [b, w_len]
    # targets segmen: [b, n_points, w_len]
    # centroids: [b, n_clusters, 2]

    batch_size = pc_clusters.shape[0]
    n_clusters = pc_clusters.shape[3]

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer_pointnet.zero_grad()
    optimizer_att.zero_grad()

    if train:
        pointnet = pointnet.train()
        att_net = att_net.train()
    else:
        pointnet = pointnet.eval()
        att_net = att_net.eval()

    # Variables to store sequence
    np_cluster = []
    lo_feats = torch.FloatTensor().to(device)
    gl_feats = torch.FloatTensor().to(device)
    targets_pc = torch.LongTensor().to(device)
    pc = torch.LongTensor()

    # shuffle clusters
    pc_clusters, targets = shuffle_clusters(pc_clusters, targets)

    # set rotation angle
    r_angle = np.random.uniform() * 2 * np.pi

    # BasePointNet through all clusters
    for w in range(n_clusters):
        # get window
        in_points = pc_clusters[:, :, :, w].numpy()
        targets_w = targets[:, :, w]
        # data augmentation
        if train:
            # rotate point cloud
            in_points[:, :, :3] = rotate_point_cloud_z(in_points[:, :, :3], rotation_angle=r_angle)
            # shuffle data and labels todo check
            in_points, targets_w, ix = shuffle_data(in_points, targets_w)

        targets_w = torch.LongTensor(targets_w).to(device)
        in_points = torch.Tensor(in_points).to(device)
        # get embeds from PointNet
        local_global_features, feat_transform = pointnet(in_points)
        local_feat = local_global_features[:, :, -64:]  # [batch, n_point, 64]
        global_feat = local_global_features[:, 0, :-64]  # [batch, 256]
        global_feat = global_feat.view(-1, 1, GLOBAL_FEAT_SIZE)  # (b, seq, feat)
        # store
        np_cluster.append(local_feat.shape[1])
        lo_feats = torch.cat((lo_feats, local_feat), dim=1)
        gl_feats = torch.cat((gl_feats, global_feat), dim=1)

        if task == 'segmentation':
            # we get labels of points instead of "targets" because there is an error if we concatenate w of targets
            # labels = (in_points[:, :, 3] == 15).type(torch.LongTensor).to(device)
            targets_pc = torch.cat((targets_pc, targets_w), dim=1)

            pc = torch.cat((pc, in_points.cpu()), dim=1)

    # key_padding_mask (Optional[Tensor]) – If specified, a mask of shape (N, S)(N,S) indicating which elements
    # within key to ignore for the purpose of attention (i.e. treat as “padding”)
    targets_mask = targets_pc.view(batch_size, -1, n_clusters)
    mask = torch.where(targets_mask != -1, torch.zeros_like(targets_mask, dtype=torch.bool),
                       torch.ones_like(targets_mask, dtype=torch.bool))
    mask = torch.all(mask, 1)  # [b,n_clusters]
    gl_feats = torch.transpose(gl_feats, 0, 1)

    if task == 'segmentation':
        logits, att_weights = att_net(gl_feats, lo_feats, centroids, np_cluster, mask)  # [b, 2, 10240]
        # gl_feats [b, 5, 256]
        # lo_feats [b, 10240, 64]
        # np_cluster [2048, 2048, 2048, 2048, 2048]

    else:  # classification
        # logits, att_weights = gru_model(pc_rnn_embeds)
        targets_pc = targets[:, 0]

    # CrossEntropy loss
    metrics['ce_loss'] = ce_loss(logits, targets_pc).view(-1, 1)  # [1, 1]
    targets_pc = targets_pc.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().cpu(), dim=1)
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
        optimizer_pointnet.step()
        optimizer_att.step()
    #  validation
    else:
        metrics['loss'] = metrics['ce_loss']

    return metrics, targets_pc, preds, last_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_path', type=str, help='path to the dataset folder')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], help='type of task',
                        default='segmentation')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/RGBN_100x100',
                        help='output folder')
    parser.add_argument('--out_path', type=str, default='pointNet/results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_windows', type=int, default=9, help='number of maximum windows per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--beta', type=float, default=0.999, help='model checkpoint path')
    parser.add_argument('--number_of_workers', type=int, default=8, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()

    train_att(args.task,
              args.dataset_path,
              args.path_list_files,
              args.out_path,
              args.number_of_points,
              args.batch_size,
              args.epochs,
              args.learning_rate,
              args.weighing_method,
              args.beta,
              args.number_of_workers,
              args.model_checkpoint)
