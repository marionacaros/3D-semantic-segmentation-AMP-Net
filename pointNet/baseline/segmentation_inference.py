import argparse
import time
from torch.utils.data import random_split
from pointNet.datasets import LidarInferenceDataset
import logging
from progressbar import progressbar
from pointNet.model.light_pointnet_256 import *
from utils.utils_plot import *

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cpu'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def test(dataset_folder,
         output_folder,
         number_of_workers,
         model_checkpoint,
         test_files):
    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)

    # Initialize dataset
    test_dataset = LidarInferenceDataset(dataset_folder=dataset_folder,
                                         task='segmentation',
                                         files=test_files,
                                         c_sample=False)

    logging.info(f'Total samples: {len(test_dataset)}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    model = SegmentationPointNet(num_classes=6,
                                 point_dimension=test_dataset.POINT_DIMENSION,
                                 device=device)

    model.to(device)
    logging.info('--- Checkpoint loaded ---')
    model.load_state_dict(checkpoint['model'])
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    # name = model_checkpoint.split('/')[-1].split('.')[0]

    figures_path = os.path.join(output_folder, 'figures')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    for data in progressbar(test_dataloader):
        pc, file_name = data  # [1, 2000, 14], [1, 2000]
        file_name = file_name[0].split('/')[-1].split('.')[0]

        model = model.eval()
        logits, feature_transform = model(pc.to(device))  # [batch, n_points, 2] [2, batch, 128]

        # get predictions
        probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
        preds = torch.LongTensor(probs.data.max(1)[1])

        # Length tower points
        tower_pts = len(preds[preds == 1])

        block = file_name.split('_')[-2]
        preds = preds.numpy().reshape(-1)
        ix_tower = np.where(preds == 1)[0]
        ix_cable = np.where(preds == 2)[0]

        if tower_pts > 20:
            # print(np.unique(pc[:, :, 3].numpy().astype(int)))
            if 15 in np.unique(pc[:, :, 3].numpy().astype(int)):  # store files not containing a labeled tower
                # Plot predictions
                plot_pointcloud_with_labels(pc.squeeze(0).numpy(),
                                            preds,
                                            1,
                                            file_name + '_preds',
                                            path_plot=figures_path)

                for ix in ix_tower:
                    with open(os.path.join(output_folder, str(block) + '_segmented_towers.csv'), 'a') as fid:
                        # label, x, y, z, intensity
                        fid.write('15, %s, %s, %s, %s\n' % (
                            pc[:, ix, 11].item(),
                            pc[:, ix, 12].item(),
                            pc[:, ix, 13].item(),
                            tower_pts

                        ))
                for ix in ix_cable:
                    with open(os.path.join(output_folder, str(block) + '_segmented_towers.csv'), 'a') as fid:
                        # label, x, y, z, intensity
                        fid.write('14, %s, %s, %s, %s\n' % (
                            pc[:, ix, 11].item(),
                            pc[:, ix, 12].item(),
                            pc[:, ix, 13].item(),
                            tower_pts
                        ))

    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='results', help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=8, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s %(levelname)-8s %(message)s]',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    filename = 'detected-positive-sorted.txt'

    with open(os.path.join('results', filename), 'r') as f:
        test_files = f.read().splitlines()

    test(args.data_path,
         args.output_dir,
         args.number_of_workers,
         args.model_checkpoint,
         test_files)
    logging.info("--- Segmentation time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
