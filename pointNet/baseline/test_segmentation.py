import argparse
import os.path
import time
import torch.nn.functional as F
from torch.utils.data import random_split
from pointNet.datasets import LidarDatasetExpanded
import logging
# from pointNet.model.light_pointnet_256 import *
from pointNet.model.pointnet import SegmentationPointNet
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
from codecarbon import track_emissions

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

n_points = 4096

@track_emissions()
def test(dataset_folder,
         output_folder,
         number_of_workers,
         model_checkpoint,
         path_list_files):
    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)
    iou = {
        'bckg': [],
        'tower': [],
        'cables': [],
        'low_veg': [],
        'high_veg': []
    }
    metrics = {}
    accuracy = []

    with open(os.path.join(path_list_files, 'test_seg_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Initialize dataset
    test_dataset = LidarDatasetExpanded(dataset_folder=dataset_folder,
                                task='segmentation', number_of_points=None,
                                files=test_files,
                                fixed_num_points=False)

    logging.info(f'Total samples: {len(test_dataset)}')
    logging.info(f'Task: {test_dataset.task}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    model = SegmentationPointNet(num_classes=5,
                                 point_dimension=3)

    model.to(device)
    logging.info('--- Checkpoint loaded ---')
    model.load_state_dict(checkpoint['model'])
    weighing_method = checkpoint['weighing_method']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']

    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    model_name = model_checkpoint.split('/')[-1].split('.')[0]
    logging.info(f'Model name: {model_name} ')
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    # with open(os.path.join(output_folder, 'IoU-results-%s.csv' % name), 'w+') as fid:
    #     fid.write('file_name,n_points,w,IoU_tower,IoU_low_veg,IoU_med_veg,IoU_high_veg,IoU_bckg,IoU_cables\n')

    if not os.path.exists(os.path.join(output_folder, 'figures')):
        os.makedirs(os.path.join(output_folder, 'figures'))

    for data in progressbar(test_dataloader):
        pc, targets, file_name = data  # [1, 2000, 12], [1, 2000]
        file_name = file_name[0].split('/')[-1].split('.')[0]

        model = model.eval()
        logits, feature_transform = model(pc.to(device))  # [batch, n_points, 2] [2, batch, 128]

        # get predictions
        probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
        preds = probs.data.max(1)[1].view(-1).numpy()
        targets = targets.view(-1).numpy()

        # compute metrics
        metrics = get_accuracy(preds, targets, metrics, 'segmentation', None)
        accuracy.append(metrics['accuracy'])

        labels = set(targets)

        if 0 in labels:
            iou_bckg = get_iou_obj(preds, targets, 0)
            iou['bckg'].append(iou_bckg)
        else:
            iou_bckg = None
        if 1 in labels:
            iou_tower = get_iou_obj(targets, preds, 1)
            iou['tower'].append(iou_tower)
        else:
            iou_tower = None
        if 2 in labels:
            iou_cables = get_iou_obj(preds, targets, 2)
            iou['cables'].append(iou_cables)
        else:
            iou_cables = None
        if 3 in labels:
            iou_low_veg = get_iou_obj(preds, targets, 3)
            iou['low_veg'].append(iou_low_veg)
        else:
            iou_low_veg = None

        if 4 in labels:
            iou_high_veg = get_iou_obj(preds, targets, 4)
            iou['high_veg'].append(iou_high_veg)
        else:
            iou_high_veg = None

        # plot predictions
        # if iou['tower'][-1] < 1:
        if iou_tower:
            iou_tower = round(iou_tower, 3)
        else:
            iou_tower = 'No'

        #
        #
        # plot_pointcloud_with_labels(pc.squeeze(0).numpy(),
        #                             preds,
        #                             iou_tower,
        #                             file_name + '40x40' + '_preds.png',
        #                             path_plot=os.path.join(output_folder, 'figures'))
        # plot_pointcloud_with_labels(pc.squeeze(0).numpy(),
        #                             targets.reshape(-1),
        #                             iou_tower,
        #                             file_name + '40x40' + '_targets.png',
        #                             path_plot=os.path.join(output_folder, 'figures'))

        # mean_ptg_corrects.append(ptg_corrects)
        # with open(os.path.join(output_folder, 'IoU-results-%s.csv' % name), 'a') as fid:
        #     fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (file_name,
        #                                                 targets.view(-1).shape[0],
        #                                                 1,
        #                                                 iou_tower,
        #                                                 iou_low_veg,
        #                                                 iou_med_veg,
        #                                                 iou_high_veg,
        #                                                 iou_bckg,
        #                                                 iou_cables,
        #                                                 ))

        # # # store segmentation results in pickle file for plotting
        # points = points.reshape(-1, 11)
        # print(points.shape)
        # preds = preds[..., np.newaxis]
        # print(preds.shape)
        #
        # points = np.concatenate((points.cpu().numpy(), preds), axis=1)
        # dir_results = 'segmentation_regular'
        # with open(os.path.join(output_folder, dir_results, file_name), 'wb') as f:
        #     pickle.dump(points, f)
    iou_arr = [np.mean(iou['tower']), np.mean(iou['low_veg']),
               np.mean(iou['high_veg']), np.mean(iou['bckg']), np.mean(iou['cables'])]
    mean_iou = np.mean(iou_arr)
    print('-------------')
    print('mean_iou_tower: ', round(float(np.mean(iou['tower'])), 3))
    print('mean_iou_low_veg: ', round(float(np.mean(iou['low_veg'])), 3))
    # print('mean_iou_med_veg: ', round(float(np.mean(iou['med_veg'])), 3))
    print('mean_iou_high_veg: ', round(float(np.mean(iou['high_veg'])), 3))
    print('mean_iou_cables: ', round(float(np.mean(iou['cables'])), 3))
    print('mean_iou_background: ', round(float(np.mean(iou['bckg'])), 3))
    print('mean_iou: ', round(float(mean_iou), 3))
    print('accuracy: ', round(float(np.mean(accuracy)), 3))
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

    # 'model_name,n_points,IoU_tower,IoU_low_veg,IoU_high_veg,IoU_cables,IoU_bckg,mIoU,OA,params,inf_time\n')
    with open(os.path.join(output_folder, 'IoU-results-v2.csv'), 'a') as fid:
        fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (model_name,
                                                          n_points,
                                                          round(float(np.mean(iou['tower'])), 3),
                                                          round(float(np.mean(iou['low_veg'])), 3),
                                                          round(float(np.mean(iou['high_veg'])), 3),
                                                          round(float(np.mean(iou['cables'])), 3),
                                                          round(float(np.mean(iou['bckg'])), 3),
                                                          round(float(mean_iou), 4),
                                                          round(float(np.mean(accuracy)), 3),
                                                          total_params,
                                                          round((time.time() - start_time) / 60, 3)
                                                          ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/dades/LIDAR/towers_detection/datasets/towers_100x100_cls')
    parser.add_argument('--output_folder', type=str,
                        default='/home/m.caros/work/objectDetection/pointNet/results',
                        help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_100x100')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    test(args.dataset_folder,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
