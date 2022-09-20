import torch
import math
from pointNet.utils import *


def collate_segmen_padd(batch, n_points=2048):
    """
    Pads batch of variable length
    todo revisar

    :param batch: List of point clouds
    """
    lengths = [t[0].shape[0] for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    b_data = [torch.Tensor(t[0]) for t in batch]

    # get number of windows (upper bound)
    len_w = [math.ceil(l_pc / n_points) for l_pc in lengths]  # example list: [3, 7, 1, 1]

    # padding
    batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=True, padding_value=0)  # [max_length,B,D]
    pad_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)  # [max_length,B,D]

    # file names
    filenames = [t[2] for t in batch]

    return batch_data, pad_targets, filenames, torch.tensor(lengths), torch.tensor(len_w)


def collate_classif_padd(batch, num_w=5, n_points=2048):
    """
    Pads batch of variable length

    :param n_points: number of points in each window
    :param batch: List of point clouds

    """
    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]

    # get number of windows (upper bound)
    # len_w = [math.ceil(l_pc / n_points) for l_pc in lengths]  # example list: [3, 7, 1, 1]
    # list_t = [torch.LongTensor([targets[i]] * len_w[i]) for i in range(len(batch))]

    # padding
    batch_data = []
    pad_targets = []
    for pc, target in zip(b_data, targets):
        p1d = (0, num_w - pc.shape[2])  # pad last dim by 1
        batch_data.append(torch.nn.functional.pad(pc, p1d, "replicate"))
        pad_targets.append(torch.nn.functional.pad(target, p1d, "constant", -1))

    batch_data = torch.stack(batch_data, dim=0)
    pad_targets = torch.stack(pad_targets, dim=0)
    # batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=False, padding_value=0)  # [max_length,B,D]

    return batch_data, pad_targets, filenames
