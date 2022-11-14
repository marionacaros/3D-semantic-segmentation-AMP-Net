import torch


def collate_seq_padd(batch, num_w=5):
    """
    Pads batch of variable length

    :param batch: List of point clouds
    :param num_w: max number of windows
    :return: batch_data, pad_targets, filenames

    """
    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]

    # padding
    batch_data = []
    pad_targets = []
    for pc_w, target in zip(b_data, targets):
        p1d = (0, num_w - pc_w.shape[2])  # pad last dim by 1
        batch_data.append(torch.nn.functional.pad(pc_w, p1d, "replicate"))
        pad_targets.append(torch.nn.functional.pad(target, p1d, "constant", -1))

    batch_data = torch.stack(batch_data, dim=0)
    pad_targets = torch.stack(pad_targets, dim=0)
    # batch_data = torch.nn.utils.rnn.pad_sequence(b_data, batch_first=False, padding_value=0)  # [max_length,B,D]

    return batch_data, pad_targets, filenames


def collate_seq4segmen_padd(batch, num_w=5):
    """
    Pads batch of variable length

    :param batch: List of point clouds
    :param num_w: max number of windows
    :return: batch_data: [10240, b, dim],
             batch_kmeans_data: [b, 2048, dim, seq_len],
             pad_targets: [b, 2048, seq_len,
             filenames

    """
    b_kmeans_pc = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]

    # padding
    batch_kmeans_data = []
    pad_targets = []
    for pc_w, labels in zip(b_kmeans_pc, targets):
        p1d = num_w - pc_w.shape[2]  # pad needed windows for batch with replicated windows
        pc_pad = torch.nn.functional.pad(pc_w, (0, p1d), 'replicate')  # [2048, 11, 5]
        # mask = torch.zeros(pc_pad.shape)
        # mask[:, :, pc_w.shape[2] + 1:] = True
        # mask = mask[:, 3, :].type(torch.BoolTensor)
        # pc_pad_labels = pc_pad[:, 3, :]
        # pc_pad_labels[mask] = -1
        # pc_pad[:, 3, :] = pc_pad_labels

        batch_kmeans_data.append(pc_pad)
        # labels = (pc_w[:, 3, :] == 15).type(torch.LongTensor)  # [n_points, seq_len]
        pad_targets.append(torch.nn.functional.pad(labels, (0, p1d), "constant", -1))

    batch_kmeans_data = torch.stack(batch_kmeans_data, dim=0)
    pad_targets = torch.stack(pad_targets, dim=0)

    # batch_data = torch.nn.utils.rnn.pad_sequence(b_full_pc, batch_first=False, padding_value=0)  # [max_length,B,D]

    return batch_kmeans_data, pad_targets, filenames
