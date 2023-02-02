import argparse
import pickle
import time
from torch.utils.data import random_split
from pointNet.datasets import LidarDataset4Test
import logging
# from model.pointnet import *
from pointNet.model.pointnetAtt import BasePointNet, SegmentationWithAttention
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
from pointNet.collate_fns import *
import torch.nn.functional as F
from codecarbon import track_emissions
from tqdm import tqdm

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


# @track_emissions()
def test(dataset_path,
         out_path,
         n_points,
         number_of_workers,
         model_checkpoint,
         path_list_files):
    start_time = time.time()

    MAX_CLUSTERS = 18
    GLOBAL_FEAT_SIZE = 256
    ATT_HEADS = 8
    NUM_CLASSES = 5
    NAME_DIR = 'preds_Att'

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
    arr_points = []

    with open(os.path.join(path_list_files, 'test_seg_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Initialize dataset
    test_dataset = LidarDataset4Test(dataset_folder=dataset_path,
                                     task='segmentation',
                                     number_of_points=n_points,
                                     files=test_files,
                                     fixed_num_points=False,
                                     c_sample=False)

    logging.info(f'Total samples: {len(test_dataset)}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    base_pointnet = BasePointNet(point_dimension=3,
                                 return_local_features=True,
                                 global_feat_dim=GLOBAL_FEAT_SIZE,
                                 device=device)
    segmen_net = SegmentationWithAttention(GLOBAL_FEAT_SIZE, ATT_HEADS, local_dim=64, num_classes=NUM_CLASSES,
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
    model_name = model_checkpoint.split('/')[-1].split('.')[0]
    logging.info(f'Model name: {model_name} ')
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

    out_path = os.path.join(out_path, NAME_DIR)
    if not os.path.exists(out_path + '/figures'):
        os.makedirs(out_path + '/figures')

    # with open(os.path.join(output_folder, 'IoU-results.csv'), 'w+') as fid:
    #     fid.write('model_name,n_points,IoU_tower,IoU_low_veg,IoU_high_veg,IoU_cables,IoU_bckg,mIoU,OA,params,inf_time\n')

    xstr = lambda x: "None" if x is None else str(round(x, 2))

    for data in tqdm(test_dataloader):

        pc, file_name = data  # [1, points, 12], [1, 2000]
        D = pc.shape[2]
        file_name = file_name[0].split('/')[-1].split('.')[0]
        arr_points.append(pc.shape[1])

        # clusters_list, centroids = kmeans_clustering(pc, n_points=n_points, get_centroids=True,
        #                                              max_clusters=MAX_CLUSTERS,
        #                                              out_path='k_means2k_'+str(MAX_CLUSTERS),
        #                                              file_name=file_name)

        # # load clustered points
        with open('k_means_25/'+file_name + '_clusters_list.pkl', 'rb') as f:
            clusters_list = pickle.load(f)
        with open('k_means_25/'+file_name + '_centroids.pkl', 'rb') as f:
            centroids = pickle.load(f)

        centroids = centroids.unsqueeze(0)
        # pc_w size [2048, dims, w_len]
        targets_list = get_labels(clusters_list)

        base_pointnet = base_pointnet.eval()
        segmen_net = segmen_net.eval()

        # Variables to store output of model
        np_cluster = []
        lo_feats = torch.FloatTensor().to(device)
        gl_feats = torch.FloatTensor().to(device)
        pc = torch.LongTensor()
        targets_pc = torch.LongTensor()

        # BasePointNet through all clusters
        for ix, in_points in enumerate(clusters_list):
            in_points = in_points[:, :9]
            in_points = in_points.unsqueeze(0)

            local_global_features, feat_transform = base_pointnet(in_points.to(device))
            local_feat = local_global_features[:, :, -64:]  # [batch, n_point, 64]
            global_feat = local_global_features[:, 0, :-64]  # [batch, 256]
            global_feat = global_feat.view(-1, 1, GLOBAL_FEAT_SIZE)  # (b, seq, feat)
            # store
            np_cluster.append(local_feat.shape[1])
            lo_feats = torch.cat((lo_feats, local_feat), dim=1)
            gl_feats = torch.cat((gl_feats, global_feat), dim=1)

            targets_pc = torch.cat((targets_pc, targets_list[ix]), dim=0)
            pc = torch.cat((pc, in_points.cpu()), dim=1)

        gl_feats = torch.transpose(gl_feats, 0, 1)
        logits, att_weights = segmen_net(gl_feats, lo_feats, centroids.to(device), np_cluster)  # [b, 2, 10240]

        # get predictions
        probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
        preds = torch.LongTensor(probs.data.max(1)[1]).view(-1).cpu()

        targets_pc = torch.LongTensor(targets_pc.view(-1))

        # compute metrics
        metrics = get_accuracy(preds.numpy(), targets_pc.numpy(), metrics, 'segmentation', None)
        accuracy.append(metrics['accuracy'])

        # compute metrics
        labels = set(targets_pc.numpy().reshape(-1))

        if 0 in labels:
            iou_others = get_iou_obj(preds, targets_pc, 0)
            iou['bckg'].append(iou_others)
        else:
            iou_others = None
        if 1 in labels:
            iou_tower = get_iou_obj(preds, targets_pc, 1)
            iou['tower'].append(iou_tower)
        else:
            iou_tower = None
        if 2 in labels:
            iou_lines = get_iou_obj(preds, targets_pc, 2)
            iou['cables'].append(iou_lines)
        else:
            iou_lines = None
        if 3 in labels:
            iou_low_veg = get_iou_obj(preds, targets_pc, 3)
            iou['low_veg'].append(iou_low_veg)
        else:
            iou_low_veg = None
        if 4 in labels:
            iou_high_veg = get_iou_obj(preds, targets_pc, 4)
            iou['high_veg'].append(iou_high_veg)
        else:
            iou_high_veg = None

        mIoU = np.nanmean([np.array([iou_tower, iou_lines, iou_low_veg, iou_high_veg, iou_others],
                                    dtype=np.float64)])
        # plot results
        ious = [iou_tower, iou_lines, mIoU]
        print(ious)
        # pc = pc.squeeze(0)
        # plot_pointcloud_with_labels(pc,
        #                             preds.numpy(),
        #                             targets_pc.reshape(-1),
        #                             ious,
        #                             file_name + path_list_files.split('/')[-1] + '_preds',
        #                             path_plot=out_path + '/figures',
        #                             point_size=4)
        #
        # with open(os.path.join(os.path.dirname(out_path), 'IoU-results-test-samples-cls.csv'), 'a') as fid:
        #     fid.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (file_name,
        #                                              xstr(iou_tower),
        #                                              xstr(iou_lines),
        #                                              xstr(iou_low_veg),
        #                                              xstr(iou_high_veg),
        #                                              xstr(iou_others),
        #                                              xstr(mIoU),
        #                                              metrics['accuracy']))

        # store segmentation results in pickle file for plotting
        # points = pc
        # points = points.reshape(-1, D-1)
        # preds = preds[..., np.newaxis]
        # points = np.concatenate((points.cpu().numpy(), preds), axis=1)
        # preds_dir = os.path.join(out_path, 'data_pred')
        # # store predictions in pickle files
        # if not os.path.exists(preds_dir):
        #     os.makedirs(preds_dir)
        # with open(os.path.join(preds_dir, file_name + '.pkl'), 'wb') as f:
        #     pickle.dump(points, f)

    # plot histogram of number of points
    # plot_hist(arr_points, 'hist_points_test'+NAME_DIR)
    iou_arr = [np.mean(iou['tower']), np.mean(iou['low_veg']),  # , np.mean(iou['med_veg'])
               np.mean(iou['high_veg']), np.mean(iou['bckg']), np.mean(iou['cables'])]
    mean_iou = np.mean(iou_arr)
    print('-------------')
    print('mean_iou_tower: ', np.mean(iou['tower']))
    print('mean_iou_low_veg: ', np.mean(iou['low_veg']))
    # print('mean_iou_med_veg: ', np.mean(iou['med_veg']))
    print('mean_iou_high_veg: ', np.mean(iou['high_veg']))
    print('mean_iou_cables: ', np.mean(iou['cables']))
    print('mean_iou_background: ', np.mean(iou['bckg']))
    print('mean_iou: ', mean_iou)
    print('accuracy: ', np.mean(accuracy))
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

    # 'model_name,n_points,IoU_tower,IoU_low_veg,IoU_high_veg,IoU_cables,IoU_bckg,mIoU,OA,params,inf_time\n')
    with open(os.path.join(os.path.dirname(out_path), 'IoU-results-v2.csv'), 'a') as fid:
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
    parser.add_argument('--dataset_path', type=str, help='path to the dataset folder',
                        default='/dades/LIDAR/towers_detection/datasets/towers_100x100')
    parser.add_argument('--out_path', type=str, default='results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str, help='output folder',
                        default='train_test_files/RGBN_100x100_old')

    args = parser.parse_args()

    test(args.dataset_path,
         args.out_path,
         args.number_of_points,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
