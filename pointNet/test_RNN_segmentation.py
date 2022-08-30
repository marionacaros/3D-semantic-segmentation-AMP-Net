import argparse
import json
import time

import torch
from progressbar import progressbar
from torch.utils.data import random_split
import torch.nn.functional as F
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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from collections import Counter

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def test(dataset_folder,
         n_points,
         output_folder,
         number_of_workers,
         model_checkpoint,
         path_list_files='pointNet/data/train_test_files/RGBN'):
    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)

    with open(os.path.join(path_list_files, 'test_seg_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    logging.info(f'Dataset folder: {dataset_folder}')
    # Tensorboard location and plot names
    # now = datetime.datetime.now()
    # location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'
    writer_test = None

    # Initialize dataset
    test_dataset = LidarDataset(dataset_folder=dataset_folder,
                                task='segmentation', number_of_points=n_points,
                                files=test_files,
                                fixed_num_points=False)

    logging.info(f'Tower PC in test: {test_dataset.len_towers}')
    logging.info(f'Landscape PC in test: {test_dataset.len_landscape}')
    logging.info(
        f'Proportion towers/landscape: {round((test_dataset.len_towers / (test_dataset.len_towers + test_dataset.len_landscape)) * 100, 3)}%')
    logging.info(f'Total samples: {len(test_dataset)}')
    logging.info(f'Task: {test_dataset.task}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False,
                                                  collate_fn=collate_segmen_padd)

    cls_model = RNNClassificationPointNet(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
                                          hidden_size=128,
                                          point_dimension=test_dataset.POINT_DIMENSION)
    seg_model = RNNSegmentationPointNet(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES)

    cls_model.to(device)
    seg_model.to(device)
    cls_model.load_state_dict(checkpoint['cls_model'])
    seg_model.load_state_dict(checkpoint['seg_model'])
    weighing_method = checkpoint['weighing_method']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']
    logging.info('--- Checkpoint loaded ---')

    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    name = model_checkpoint.split('/')[-1]
    logging.info(f'Model name: {name} ')

    # print model and parameters
    # INPUT_SHAPE = (7, 2000)
    # summary(model, INPUT_SHAPE)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in cls_model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # print(f'{name} {parameter}')
        table.add_row([name, params])
        total_params += params
    # print(table)
    logging.info(f"Total model params: {total_params}")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    mean_iou_tower = []
    mean_iou_veg = []

    for data in progressbar(test_dataloader):
        points, targets, filenames, lengths, len_w = data
        # classifications shapes: [b, n_samples, dims], [b, w_len], [b], [b], [b]
        # segmentation shapes : [b, n_samples, dims], [b, n_samples], [b], [b], [b]

        points, targets = points.to(device), targets.to(device)  # ([batch, 18802, 11]
        # len_w = len_w.to(device)

        cls_model = cls_model.eval()
        hidden = cls_model.initHidden(points, device)

        # split into windows of fixed size and duplicate points up to 4 windows
        pc_w, targets = split4segmen_test(points, n_points, False, writer_test, filenames, lengths='',
                                                 targets=targets,
                                                 device=device,
                                                 duplicate=True)
        targets = targets.reshape(-1).cpu().numpy()
        # mask = targets != [-1] * len(targets)
        # targets = targets[mask]

        preds_pc = []
        # loop over windows
        for w in range(pc_w.shape[3]):
            in_points = pc_w[:, :, :, w]
            # get hidden from classification model
            _, hidden, feat_transf, local_features = cls_model(in_points, hidden, get_preds=True)
            # [b, n_points, 2] [2, b, 128] [b,64,64]

            # get predictions of segmentation model
            logits = seg_model(hidden, local_features)  # [b, 2048, 2]
            # logits = logits.reshape(-1, logits.shape[2])
            # targets_w = targets[:, w * n_points: (w + 1) * n_points]

            # get probabilities and predictions
            log_probs = F.log_softmax(logits.detach().cpu(), dim=1)
            probs = torch.exp(log_probs)
            # preds = log_probs.data.max(1)[1]
            preds = probs.data.max(2)[1]
            preds = preds.unsqueeze(-1)

            if w == 0:
                preds_pc = preds
            else:
                preds_pc = torch.cat((preds_pc, preds), 2)

        # preds_pc = preds_pc.reshape(-1)
        # preds_sum = torch.sum(preds_pc, 2)
        # preds_pc = preds_sum > 2
        all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN
        all_neg = (np.array(targets) == np.zeros(len(targets))).sum()  # TN + FP
        # apply mask
        # preds_pc = preds_pc[mask]
        detected_positive = (np.array(preds_pc) == np.ones(len(targets)))  # boolean with positions of 1s
        detected_negative = (np.array(preds_pc) == np.zeros(len(targets)))  # boolean with positions of 0s

        corrects = np.array(np.array(preds_pc) == np.array(targets))
        tp = np.logical_and(corrects, detected_positive).sum()
        tn = np.logical_and(corrects, detected_negative).sum()
        fp = np.array(detected_positive).sum() - tp
        fn = np.array(detected_negative).sum() - tn

        # summarize scores
        # file_name = file_name[0].split('/')[-1]
        # print(file_name)
        # print('detected_positive: ', np.array(detected_positive).sum())

        iou_veg = tn / (all_neg + fn)

        if all_positive.sum() > 0:
            iou_tower = tp / (all_positive + fp)
            print('IOU tower: ', iou_tower)
            print('IOU veg: ', iou_tower)
            mean_iou_tower.append(iou_tower)
            print('-------------')

        mean_iou_veg.append(iou_veg)

    mean_iou_tower = np.array(mean_iou_tower).sum() / len(mean_iou_tower)
    mean_iou_veg = np.array(mean_iou_veg).sum() / len(mean_iou_veg)
    print('-------------')
    print('mean_iou_tower: ', mean_iou_tower)
    print('mean_iou_veg: ', mean_iou_veg)
    print('mean_iou: ', (mean_iou_tower + mean_iou_veg) / 2)
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--output_folder', type=str, default='pointNet/results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str,
                        default='pointNet/data/train_test_files/RGBN',
                        help='output folder')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    test(args.dataset_folder,
         args.number_of_points,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
