from pointNet.utils import *


def collate_seq_padd(batch, num_w=5, n_points=2048):
    """
    Pads batch of variable length

    :param batch: List of point clouds
    :param num_w: max number of windows
    :param n_points: number of points in each window
    :return: batch_data, pad_targets, filenames

    """
    b_data = [torch.FloatTensor(t[0]) for t in batch]
    targets = [torch.LongTensor(t[1]) for t in batch]
    filenames = [t[2] for t in batch]

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
