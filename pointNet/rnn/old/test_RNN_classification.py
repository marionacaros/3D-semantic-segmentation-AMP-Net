import argparse
import time
from torch.utils.data import random_split
import torch.nn.functional as F
from pointNet.datasets import LidarDataset
from pointNet.model.pointnetRNN import GRUPointNet
import logging
import warnings
from pointNet.collate_fns import *
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

    with open(os.path.join(path_list_files, 'test_reduced_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    logging.info(f'Dataset folder: {dataset_folder}')
    # Tensorboard location and plot names
    # now = datetime.datetime.now()
    # location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'
    writer_test = None
    # writer_test = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_testRGB')
    # logging.info(f"Tensorboard runs: {writer_test.get_logdir()}")

    # Initialize dataset
    test_dataset = LidarDataset(dataset_folder=dataset_folder,
                                task='classification', number_of_points=n_points,
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
                                                  collate_fn=collate_seq_padd)

    model = GRUPointNet(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
                        hidden_size=128,
                        point_dimension=test_dataset.POINT_DIMENSION)
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

    # print model and parameters
    # INPUT_SHAPE = (7, 2000)
    # summary(model, INPUT_SHAPE)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # print(f'{name} {parameter}')
        table.add_row([name, params])
        total_params += params
    # print(table)
    logging.info(f"Total model params: {total_params}")

    model.load_state_dict(checkpoint['model'])
    name = model_checkpoint.split('/')[-1]
    print(name)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'w+') as fid:
        fid.write('filename,prob[0],prob[1],pred,target\n')
    with open(os.path.join(output_folder, 'results-positives-%s.csv' % name), 'w+') as fid:
        fid.write('filename\n')

    all_preds = []
    all_probs = []
    targets = []

    for data in progressbar(test_dataloader):
        points, target, filename, lengths, len_w = data  # [b, n_samples, dims], [b, w_len], [32], [32]

        points, target = points.to(device), target.to(device)  # ([batch, 18802, 11]
        target = target.view(-1)

        model = model.eval()
        hidden = model.initHidden(points, device)

        # split into windows of fixed size
        pc_w, _ = split4classif_point_cloud(points, n_points, False, writer_test, filename, lengths='', targets=target,
                                            device=device)
        preds_w = []
        # forward pass
        for w in range(pc_w.shape[3]):
            in_points = pc_w[:, :, :, w]

            logits, hidden, feat_transf, local_features = model(in_points, hidden, get_preds=True)
            # [1, 2] [2, 1, 128] [1,64,64]
            log_probs = F.log_softmax(logits.detach().cpu(), dim=1)  # [1, 2]
            probs = torch.exp(log_probs)
            preds_w.append(log_probs.data.max(1)[1].item())

            if w == 0:
                probs_pc = probs
            else:
                probs_pc = torch.cat((probs_pc, probs), 0)

        # get mean probabilities across all windows
        probs_pc = probs_pc.mean(0)
        all_probs.append(probs_pc.numpy().reshape(2))
        # get most common prediction across all windows
        c = Counter(preds_w)
        value, count = c.most_common()[0]
        all_preds.append(value)
        targets.append(target[0].item())

        # if target.item() == 1 or value.item() == 1:
        #     with open(os.path.join(output_folder, 'results-positives-%s.csv' % name), 'a') as fid:
        #         fid.write('%s\n' % (filename[0].split('/')[-1]))
        #
        # if value.item() != target.item():
        #     # print(f'Wrong prediction in: {filename}')
        #     with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'a') as fid:
        #         fid.write('%s,%s,%s,%s,%s\n' % (
        #         filename[0], probs[0, 0].item(), probs[0, 1].item(), pred.item(), target.item()))
        #     # if target.item() == 0:
        #     #     with open(os.path.join(output_folder, 'files-segmentation-%s.csv' % name), 'a') as fid:
        #     #         fid.write('%s\n' % (filename[0]))

    epochs = checkpoint['epoch']
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

    # --------  considering ONLY TRANSMISSION TOWERS as correct -------

    print('--------  considering ONLY TRANSMISSION TOWERS as correct -------')
    # calculate F1 score
    lr_f1 = f1_score(targets, all_preds)
    lr_auc = 0

    # keep probabilities for the positive outcome only
    lr_probs = np.array(all_probs).transpose()[1]  # [2, len(test data)]
    lr_precision, lr_recall, thresholds = precision_recall_curve(targets, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    corrects = (np.array(all_preds) == np.array(targets))  # boolean with matched predictions
    detected_positive = (np.array(all_preds) == np.ones(len(all_preds)))  # boolean with positions of 1s
    all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN

    tp = np.logical_and(corrects, detected_positive).sum()
    fp = detected_positive.sum() - tp

    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    print('All positive: ', all_positive)
    print('TP: ', tp)
    print('FP: ', fp)

    print(f'Accuracy: {round(corrects.sum() / len(test_dataset) * 100, 2)} %')
    # Recall - Del total de torres, quin percentatge s'han trobat?
    print(f'Recall: {round(tp / all_positive * 100, 2)} %')
    # Precision - De les que s'han detectat com a torres quin percentatge era realment torre?
    print(f'Precision: {round(tp / detected_positive.sum() * 100, 2)} %')

    # data = {"lr_recall": str(list(lr_recall)),
    #         "lr_precision": str(list(lr_precision))}
    # with open('pointNet/json_files/precision-recall-%s.json' % name, 'w') as f:
    #     json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--output_folder', type=str, default='pointNet/results', help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--path_list_files', type=str,
                        default='pointNet/data/train_test_files/RGBN')
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
