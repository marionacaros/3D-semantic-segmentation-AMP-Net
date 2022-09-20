import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from pointNet.datasets import LidarDataset
from pointNet.model.pointnetRNN import GRUPointNet, RNNSegmentationPointNet, AttentionClassifier
import logging
import datetime
import glob
from sklearn.metrics import balanced_accuracy_score
import warnings
from pointNet.collate_fns import *
from prettytable import PrettyTable
# warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    print(f"cuda available")
    device = 'cuda'
else:
    print(f"cuda not available")
    device = 'cpu'

HIDDEN_SIZE = 256
ATT_HEADS = 1
ATT_EMBEDDING = HIDDEN_SIZE * 2 * ATT_HEADS


def train_gru(task: str,
              dataset_folder: str,
              path_list_files: str,
              output_folder: str,
              n_points: int,
              n_windows: int,
              batch_size: int,
              epochs: int,
              learning_rate: float,
              weighing_method: str = 'EFS',
              beta: float = 0.999,
              number_of_workers: int = 4,
              model_checkpoint: str = None,
              c_sample: bool = False,
              use_kmeans: bool = True):

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

    NAME = 'GRU256w5_mask'

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
                                 task=task,
                                 number_of_points=n_points,
                                 number_of_windows=n_windows,
                                 files=train_files,
                                 fixed_num_points=c_sample,
                                 c_sample=False,
                                 split='kmeans')
    val_dataset = LidarDataset(dataset_folder=dataset_folder,
                               task=task,
                               number_of_points=n_points,
                               number_of_windows=n_windows,
                               files=val_files,
                               fixed_num_points=c_sample,
                               c_sample=False,
                               split='kmeans')

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
    # Models initialization
    rnn_pointnet = GRUPointNet(hidden_size=HIDDEN_SIZE,
                               point_dimension=train_dataset.POINT_DIMENSION,
                               global_feat_size=256,
                               num_att_heads=ATT_HEADS)
    attn_classifier = AttentionClassifier(ATT_EMBEDDING, ATT_HEADS)
    # Models to device
    rnn_pointnet.to(device)
    attn_classifier.to(device)
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

    # weighted loss
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean')
    # optimizer
    optimizer = optim.Adam(rnn_pointnet.parameters(), lr=learning_rate)

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        rnn_pointnet.load_state_dict(checkpoint['model'])  # cls_model
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # adjust_learning_rate(optimizer, learning_rate)

    # print cls_model and parameters
    # INPUT_SHAPE = (7, 2000)
    # summary(cls_model, INPUT_SHAPE)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in rnn_pointnet.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
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
        epoch_train_acc_w = []
        targets_pos = []
        targets_neg = []
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
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
            metrics, targets, pc_preds, c_weights = train_loop(data, optimizer, ce_loss,
                                                               rnn_pointnet, seg_model, attn_classifier,
                                                               n_points, n_windows, writer_train, task, True,
                                                               weighing_method, beta, c_weights,
                                                               kmeans=use_kmeans)

            # tensorboard
            ce_train_loss.append(metrics['ce_loss'].cpu().item())
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])
            epoch_train_acc_w.append(metrics['accuracy_w'])
            ious_tower_train.append(metrics['iou_tower'])

        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():
            for data in val_dataloader:
                metrics, targets, pc_preds, c_weights = train_loop(data, optimizer, ce_loss,
                                                                   rnn_pointnet, seg_model, attn_classifier,
                                                                   n_points, n_windows, writer_val, task, False,
                                                                   weighing_method, beta, c_weights,
                                                                   kmeans=use_kmeans)
                ious_tower_val.append(metrics['iou_tower'])

            # tensorboard
            epoch_val_loss.append(metrics['ce_loss'].cpu().item())  # in val ce_loss and total_loss are the same
            targets = targets.view(-1).cpu().numpy()

            if task == 'segmentation':  # todo check this
                mask = targets != [-1] * len(targets)
                targets = targets[mask]
                pc_preds = pc_preds[mask]

            targets_pos.append((targets == np.ones(len(targets))).sum() / len(targets))
            targets_neg.append((targets == np.zeros(len(targets))).sum() / len(targets))
            detected_negative.append((np.array(pc_preds) == np.zeros(len(pc_preds))).sum() / len(targets))
            detected_positive.append((np.array(pc_preds) == np.ones((len(pc_preds)))).sum() / len(targets))
            epoch_val_acc.append(metrics['accuracy'])
            epoch_val_acc_w.append(metrics['accuracy_w'])

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
        writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_train.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('c_weights', c_weights[1].cpu(), epoch)
        writer_val.add_scalar('c_weights', c_weights[0].cpu(), epoch)
        if task == 'segmentation':
            writer_train.add_scalar('_iou_tower', np.mean(ious_tower_train), epoch)
            writer_val.add_scalar('_iou_tower', np.mean(ious_tower_val), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            if task == 'classification':
                name = now.strftime("%m-%d-%H:%M") + '_clsGRU' + NAME
            elif task == 'segmentation':
                name = now.strftime("%m-%d-%H:%M") + '_segGRU' + NAME
            save_checkpoint_rnn(name, task, epoch, epochs_since_improvement, rnn_pointnet, seg_model, optimizer,
                                metrics['accuracy'],
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


def train_loop(data, optimizer, ce_loss, rnn_pointnet, seg_model, attn_classifier, n_points=2048, n_windows=5,
               w_tensorboard=None, task='classification', train=True, weighing_method='EFS', beta=0.999, c_weights=[],
               kmeans=True):
    """
    :param attn_classifier:
    :param data: tuple with (points, targets, filenames, lengths, len_w)
    :param optimizer: optimizer
    :param ce_loss: CrossEntropy loss
    :param rnn_pointnet: classification model
    :param seg_model: segmentation model
    :param n_points: number of points per window
    :param n_windows: number of windows
    :param w_tensorboard: tensorboard writer
    :param task: 'classification' or 'segmentation'
    :param train: set to True to perform backward propagation
    :param beta:
    :param weighing_method:
    :param kmeans:
    :param c_weights:

    :return:
    metrics, targets, pc_preds, c_weights
    """

    metrics = {'loss': [], 'reg_loss': [], 'ce_loss': [], 'accuracy': [], 'iou_tower': None}
    # iou_tower used only in segmentation
    pc_w, targets, filenames = data
    # classifications shapes: [b, n_samples, dims, w_len], [b, w_len], [b]
    # segmentation shapes : [b, n_samples, dims, w_len], [b, n_samples], [b]
    batch_size = pc_w.shape[0]

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    # init hidden
    hidden = rnn_pointnet.initHidden(pc_w.to(device), device)

    if train:
        rnn_pointnet = rnn_pointnet.train()
        if task == 'segmentation':
            seg_model = seg_model.train()
    else:
        rnn_pointnet = rnn_pointnet.eval()
        if task == 'segmentation':
            seg_model = seg_model.eval()

    pc_w, targets = pc_w.to(device), targets.to(device)  # [b, 2048, dims, w_len], [b, w_len]

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

    # RNN forward pass
    all_hiddens = []
    for w in range(pc_w.shape[3]):
        in_points = pc_w[:, :, :, w]  # get window
        out_h, feat_transf, local_features = rnn_pointnet(in_points, hidden)
        # [batch, 1, 512] [b,64,64] [b,n_p,64]
        all_hiddens.append(out_h)

        if task == 'segmentation':
            logits = seg_model(hidden, local_features)  # [b, 2048, 2]
            logits = logits.reshape(-1, logits.shape[2])
            targets_w = targets[:, w * n_points: (w + 1) * n_points]

            # ce_loss_batch = ce_loss(logits, targets_w.reshape(-1))
            # ce_loss_batch = ce_loss_batch.view(batch_size, n_points)
            # ce_loss_batch = ce_loss_batch.sum(1) / (len_w * n_points)  # [b] get mean
            # ce_loss_batch = ce_loss_batch.view(-1, 1)

    all_hiddens = torch.stack(all_hiddens, dim=1).to(device)  # [b, w_len, 512]
    all_hiddens = all_hiddens.view(targets.shape[1], batch_size, -1)  # [w_len, b, 512]

    # mask for attention: True value indicates that the corresponding position is not allowed
    mask = torch.where(targets != -1, torch.zeros_like(targets, dtype=torch.bool),
                       torch.ones_like(targets, dtype=torch.bool))
    # mask size [b, w_len]
    mask = mask.repeat(ATT_HEADS, 1).unsqueeze(dim=2)  # [b*n_heads, 5, 1]
    mask = mask.repeat(1, 1, n_windows).to(device)  # [b*n_heads, 5, 5] (N⋅num_heads,L,S)
    # mask = mask.expand(-1, targets.shape[1], targets.shape[1]).to(device)  # [8, w_len, w_len]  (N⋅num_heads,L,S)
    # todo check mask
    logits, att_weights = attn_classifier(all_hiddens, attn_mask=mask)
    # CrossEntropy loss
    metrics['ce_loss'] = ce_loss(logits, targets[:, 0]).view(-1, 1)  # [1, 1]
    targets_pc = targets[:, 0].detach().cpu()
    # get accuracy
    probs = F.log_softmax(logits.detach().cpu(), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])
    corrects = torch.eq(preds, targets_pc)
    metrics['accuracy'] = (corrects.sum() / batch_size)

    sample_weights = get_weights4sample(c_weights, targets_pc)
    accuracy_w = balanced_accuracy_score(targets_pc, preds, sample_weight=sample_weights)
    metrics['accuracy_w'] = accuracy_w

    identity = torch.eye(feat_transf.shape[-1]).to(device)  # [64, 64]
    metrics['reg_loss'] = torch.norm(identity - torch.bmm(feat_transf, feat_transf.transpose(2, 1)))

    if task == 'segmentation':
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
        metrics['loss'] = metrics['ce_loss'] + 0.001 * metrics['reg_loss']
        metrics['loss'].backward()
        optimizer.step()
    #  validation
    else:
        metrics['loss'] = metrics['ce_loss']

    return metrics, targets_pc, preds, c_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--path_list_files', type=str,
                        default='pointNet/data/train_test_files/RGBN',
                        help='output folder')
    parser.add_argument('--output_folder', type=str, default='pointNet/results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_windows', type=int, default=5, help='number of maximum windows per cloud')
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
              args.number_of_windows,
              args.batch_size,
              args.epochs,
              args.learning_rate,
              args.weighing_method,
              args.beta,
              args.number_of_workers,
              args.model_checkpoint,
              args.c_sample)
