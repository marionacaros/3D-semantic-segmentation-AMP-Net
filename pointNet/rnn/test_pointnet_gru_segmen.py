import argparse
import time
from torch.utils.data import random_split
from pointNet.datasets import LidarKmeansDataset4Test
import logging
# from model.pointnet import *
from pointNet.model.pointnetRNN import BasePointNet, SegmentationWithGRU, ClassificationFromGRU
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
from pointNet.collate_fns import *
import torch.nn.functional as F

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def test(dataset_folder,
         output_folder,
         n_points,
         number_of_workers,
         model_checkpoint,
         path_list_files):
    start_time = time.time()
    NAME = 'pointNetGRU_segmen'
    n_windows = 5

    checkpoint = torch.load(model_checkpoint)
    iou = {
        'bckg': [],
        'tower': [],
        'cables': [],
        'low_veg': [],
        'med_veg': [],
        'high_veg': []
    }
    metrics = {}
    accuracy = []

    with open(os.path.join(path_list_files, 'test_seg_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Initialize dataset
    test_dataset = LidarKmeansDataset4Test(dataset_folder=dataset_folder,
                                           task='segmentation',
                                           number_of_points=n_points,
                                           number_of_windows=n_windows,
                                           files=test_files,
                                           fixed_num_points=False,
                                           c_sample=False)

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
                                                  drop_last=False)

    GLOBAL_FEAT_SIZE = 256
    HIDDEN_SIZE = 64
    base_pointnet = BasePointNet(point_dimension=test_dataset.POINT_DIMENSION,
                                 return_local_features=True,
                                 global_feat_dim=GLOBAL_FEAT_SIZE,
                                 device=device)
    segmen_net = SegmentationWithGRU(num_classes=6, global_feat_size=GLOBAL_FEAT_SIZE, hidden_size=HIDDEN_SIZE,
                                     device=device)

    base_pointnet.to(device)
    segmen_net.to(device)
    logging.info('--- Checkpoint loaded ---')
    base_pointnet.load_state_dict(checkpoint['base_pointnet'])
    segmen_net.load_state_dict(checkpoint['segmen_net'])

    # weighing_method = checkpoint['weighing_method']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']

    # logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    name = model_checkpoint.split('/')[-1].split('.')[0]
    logging.info(f'Model name: {name} ')
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in base_pointnet.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    for name_param, parameter in segmen_net.named_parameters():
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    with open(os.path.join(output_folder, 'IoU-results-%s.csv' % name), 'w+') as fid:
        fid.write('file_name,n_points,w,IoU_tower,IoU_low_veg,IoU_med_veg,IoU_high_veg,IoU_bckg,IoU_cables\n')

    for data in progressbar(test_dataloader):
        clusters_list, targets_list, file_name = data  # [1, 2000, 12], [1, 2000]
        file_name = file_name[0].split('/')[-1].split('.')[0]

        base_pointnet = base_pointnet.eval()
        segmen_net = segmen_net.eval()

        # Variables to store output of GRU
        np_cluster = []
        lo_feats = torch.FloatTensor().to(device)
        gl_feats = torch.FloatTensor().to(device)
        targets_pc = torch.LongTensor()
        pc = torch.LongTensor()

        # BasePointNet through all clusters
        for ix, in_points in enumerate(clusters_list):
            # in_points = cluster.unsqueeze(0)
            # in_points = pc_w[:, :, :, w]  # get window
            local_global_features, feat_transform = base_pointnet(in_points.to(device))
            local_feat = local_global_features[:, :, -64:]  # [batch, n_point, 64]
            global_feat = local_global_features[:, 0, :-64]  # [batch, 256]
            global_feat = global_feat.view(-1, 1, GLOBAL_FEAT_SIZE)  # (b, seq, feat)
            # store
            np_cluster.append(local_feat.shape[1])
            lo_feats = torch.cat((lo_feats, local_feat), dim=1)
            gl_feats = torch.cat((gl_feats, global_feat), dim=1)

            targets_pc = torch.cat((targets_pc, targets_list[ix]), dim=1)
            pc = torch.cat((pc, in_points.cpu()), dim=1)

        logits = segmen_net(gl_feats, lo_feats, np_cluster)  # [b, 2, 10240]

        # get predictions
        probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
        preds = torch.LongTensor(probs.data.max(1)[1])

        # remove_padding
        preds, targets_pc, mask = rm_padding(preds.cpu(), targets_pc)
        pc = pc[mask, :]
        # pc = pc.squeeze(0).numpy()

        # compute metrics
        metrics = get_accuracy(preds, targets_pc, metrics, 'segmentation', None)
        accuracy.append(metrics['accuracy'])

        labels = set(targets_pc.view(-1).numpy())
        iou['tower'].append(get_iou_obj(preds, targets_pc, 1))
        print(iou['tower'][-1])

        if 0 in labels:
            iou_bckg = get_iou_obj(preds, targets_pc, 0)
            iou['bckg'].append(iou_bckg)
        else:
            iou_bckg = None
        if 2 in labels:
            iou_cables = get_iou_obj(preds, targets_pc, 2)
            iou['cables'].append(iou_cables)
        else:
            iou_cables = None
        if 3 in labels:
            iou_low_veg = get_iou_obj(preds, targets_pc, 3)
            iou['low_veg'].append(iou_low_veg)
        else:
            iou_low_veg = None
        if 4 in labels:
            iou_med_veg = get_iou_obj(preds, targets_pc, 4)
            iou['med_veg'].append(iou_med_veg)
        else:
            iou_med_veg = None
        if 5 in labels:
            iou_high_veg = get_iou_obj(preds, targets_pc, 5)
            iou['high_veg'].append(iou_high_veg)
        else:
            iou_high_veg = None

        # plot predictions in tensorboard
        # if iou['tower'][-1] < 0.3:
        #     plot_pointcloud_with_labels(pc,
        #                                 preds,
        #                                 round(iou['tower'][-1], 3),
        #                                 file_name + NAME + '_preds.png',
        #                                 path_plot=os.path.join(output_folder, 'figures'))
        #     plot_pointcloud_with_labels(pc,
        #                                 targets_pc.reshape(-1),
        #                                 round(iou['tower'][-1], 3),
        #                                 file_name + NAME + '_targets.png',
        #                                 path_plot=os.path.join(output_folder, 'figures'))

        # mean_ptg_corrects.append(ptg_corrects)
        with open(os.path.join(output_folder, 'IoU-results-%s.csv' % name), 'a') as fid:
            fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (file_name,
                                                        targets_pc.view(-1).shape[0],
                                                        len(clusters_list),
                                                        round(iou['tower'][-1], 3),
                                                        iou_low_veg,
                                                        iou_med_veg,
                                                        iou_high_veg,
                                                        iou_bckg,
                                                        iou_cables,
                                                        ))
        # # store segmentation results in pickle file for plotting
        # points = points.reshape(-1, 11)
        # print(points.shape)
        # preds = preds[..., np.newaxis]
        # print(preds.shape)
        #
        # points = np.concatenate((points.cpu().numpy(), preds), axis=1)
        # dir_results = 'segmentation_regular'
        # with open(os.path.join(output_folder, dir_results, file_name), 'wb') as f:
        #     pickle.dump(points, f)
    iou_arr = [np.mean(iou['tower']), np.mean(iou['low_veg']), np.mean(iou['med_veg']),
               np.mean(iou['high_veg']), np.mean(iou['bckg']), np.mean(iou['cables'])]
    mean_iou = np.mean(iou_arr)
    print('-------------')
    print('mean_iou_tower: ', np.mean(iou['tower']))
    print('mean_iou_low_veg: ', np.mean(iou['low_veg']))
    print('mean_iou_med_veg: ', np.mean(iou['med_veg']))
    print('mean_iou_high_veg: ', np.mean(iou['high_veg']))
    print('mean_iou_cables: ', np.mean(iou['cables']))
    print('mean_iou_background: ', np.mean(iou['bckg']))
    print('mean_iou: ', mean_iou)
    print('accuracy: ', np.mean(accuracy))
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--output_folder', type=str,
                        default='/home/m.caros/work/objectDetection/pointNet/results',
                        help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_x10_pc')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    test(args.dataset_folder,
         args.output_folder,
         args.number_of_points,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
