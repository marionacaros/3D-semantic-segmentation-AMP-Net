import torch
import random


def collate_seq_padd(batch):
    """
    Pads batch of variable length
    Replicate points up to desired length of windows
    Padds with -1 tensor of targets to mask values in loss during training

    :param batch: List of point clouds.
    :return: batch_data. Tensor [batch, 2048, 11, 20]
             pad_targets. Tensor [batch, 2048, 20]
             filenames. List

    """
    N_POINTS = 2048
    MAX_WINDOWS = 16

    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]

    # padding
    batch_data = []
    pad_targets = []
    for pc_w, target in zip(b_data, targets):
        if pc_w.shape[0] != N_POINTS:
            ix = random.sample(range(pc_w.shape[0]), N_POINTS)
            pc_w = pc_w[ix, :, :]
            target = target[ix, :]
        p1d = (0, MAX_WINDOWS - pc_w.shape[2])  # pad last dim
        batch_data.append(torch.nn.functional.pad(pc_w, p1d, "replicate"))
        # batch_data.append(torch.nn.functional.pad(pc_w, p1d, "constant", -1))
        pad_targets.append(torch.nn.functional.pad(target, p1d, "constant", -1))

    batch_data = torch.stack(batch_data, dim=0)
    pad_targets = torch.stack(pad_targets, dim=0)
    # batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=False, padding_value=0)  # [max_length,B,D]

    return batch_data, pad_targets, filenames

#
# def collate_seq4segmen_padd(batch, num_w=25):
#     """
#     Pads batch of variable length
#     Replicates points to achieve desired amount of windows
#     Padds with -1 tensor of targets to mask values in loss during training
#
#     :param batch: List of point clouds
#     :param num_w: max number of windows
#     :return: batch_data: [10240, b, dim],
#              batch_kmeans_data: [b, 2048, dim, seq_len],
#              pad_targets: [b, 2048, seq_len,
#              filenames
#
#     """
#     b_kmeans_pc = [torch.FloatTensor(t[0]) for t in batch]
#     targets = [torch.LongTensor(t[1]) for t in batch]
#     filenames = [t[2] for t in batch]
#
#     # padding
#     batch_kmeans_data = []
#     pad_targets = []
#     for pc_w, labels in zip(b_kmeans_pc, targets):
#         p1d = (0, num_w - pc_w.shape[2]) # pad needed windows for batch with replicated windows
#         pc_pad = torch.nn.functional.pad(pc_w, p1d, 'replicate')  # [2048, 11, 5]
#         batch_kmeans_data.append(pc_pad)
#
#         # mask = torch.zeros(pc_pad.shape)
#         # mask[:, :, pc_w.shape[2] + 1:] = True
#         # mask = mask[:, 3, :].type(torch.BoolTensor)
#         # pc_pad_labels = pc_pad[:, 3, :]
#         # pc_pad_labels[mask] = -1
#         # pc_pad[:, 3, :] = pc_pad_labels
#
#         pad_targets.append(torch.nn.functional.pad(labels, (0, p1d), "constant", -1))
#
#     batch_kmeans_data = torch.stack(batch_kmeans_data, dim=0)
#     pad_targets = torch.stack(pad_targets, dim=0)
#
#     return batch_kmeans_data, pad_targets, filenames
