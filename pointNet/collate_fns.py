import torch
import math


def collate_segmen_padd(batch, n_points=2048):
    """
    Pads batch of variable length

    :param batch: List of point clouds
    """
    lengths = [t[0].shape[0] for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    b_data = [torch.Tensor(t[0]) for t in batch]

    # get number of windows (upper bound)
    len_w = [math.ceil(l_pc/n_points) for l_pc in lengths]  # example list: [3, 7, 1, 1]

    # padding
    batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=True, padding_value=0)  # [max_length,B,D]
    pad_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)  # [max_length,B,D]

    # file names
    filenames = [t[2] for t in batch]

    return batch_data, pad_targets, filenames, torch.tensor(lengths), torch.tensor(len_w)


def collate_classif_padd(batch, n_points=2048):
    """
    Pads batch of variable length

    :param n_points: number of points in each window
    :param batch: List of point clouds

    """
    lengths = [t[0].shape[0] for t in batch]
    targets = [t[1] for t in batch]
    b_data = [torch.Tensor(t[0]) for t in batch]

    # get number of windows (upper bound)
    len_w = [math.ceil(l_pc/n_points) for l_pc in lengths]  # example list: [3, 7, 1, 1]
    list_t = [torch.LongTensor([targets[i]]*len_w[i]) for i in range(len(batch))]

    # padding
    batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=True, padding_value=0)  # [max_length,B,D]
    pad_targets = torch.nn.utils.rnn.pad_sequence(list_t, batch_first=True, padding_value=-1)  # [B, max_len_w]

    # file names
    filenames = [t[2] for t in batch]

    # compute mask
    # mask = (batch_data != 0)
    return batch_data, pad_targets, filenames, torch.tensor(lengths), torch.tensor(len_w)

