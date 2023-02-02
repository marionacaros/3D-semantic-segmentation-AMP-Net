import argparse
import time
from pointNet.datasets import LidarInferenceDataset
from pointNet.model.light_pointnet_256 import ClassificationPointNet
import logging
import os
import torch
from progressbar import progressbar
import glob

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cpu'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def test(dataset_folder,
         number_of_workers,
         model_checkpoint,
         test_files,
         out_file):

    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)
    name = model_checkpoint.split('/')[-1]
    print(f'Model: {name}')

    all_preds = []
    all_probs = []
    file_object = open(out_file, 'w')

    # Initialize dataset
    test_dataset = LidarInferenceDataset(dataset_folder=dataset_folder,
                                         task='classification',
                                         files=test_files)

    logging.info(f'Total samples: {len(test_dataset)}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)
    model = ClassificationPointNet(num_classes=test_dataset.NUM_CLASSIFICATION_CLASSES,
                                   point_dimension=test_dataset.POINT_DIMENSION,
                                   dataset=test_dataset,
                                   device=device)

    model.to(device)
    logging.info('Loading checkpoint')
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

    for data in progressbar(test_dataloader):

        pc, file_name = data  # [1, 2000, 9], [1]
        pc = pc.to(device)
        model = model.eval()

        log_probs, feature_transform = model(pc)  # [1,2], [1, 64, 64]  epoch=0, target=target, fileName=file_name
        probs = torch.exp(log_probs.cpu().detach())  # [1, 2]
        all_probs.append(probs.numpy().reshape(2))

        # .max(1) takes the max over dimension 1 and returns two values (the max value in each row and the column index
        # at which the max value is found)
        pred = probs.data.max(1)[1]
        all_preds.append(pred.item())

        if pred.item() == 1:
            with open(out_file, 'a') as fid:
                fid.write('%s\n' % (file_name[0].split('/')[-1]))

    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to the dataset folder')
    parser.add_argument('--output_dir', type=str, default='results', help='output folder where file containing '
                                                                          'detected towers is stored')
    parser.add_argument('--number_of_workers', type=int, default=8, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_x10_40x40')
    parser.add_argument('--filename', type=str, default='detected-positive-dataTrain.txt')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s %(levelname)-8s %(message)s]',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # with open(os.path.join(args.path_list_files, 'test_all_cls_files.txt'), 'r') as f:
    #     test_files = f.read().splitlines()

    test_files = glob.glob(os.path.join(args.data_path, '*.pkl'))

    out_file = os.path.join(args.output_dir, args.filename + '.txt')
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test(args.data_path,
         args.number_of_workers,
         args.model_checkpoint,
         test_files,
         out_file)

    logging.info("--- Classification time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))

