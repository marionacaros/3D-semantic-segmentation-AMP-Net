import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_3 = nn.Conv1d(128, 256, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256, bias=False)
        self.fc_2 = nn.Linear(256, 128, bias=False)
        self.fc_3 = nn.Linear(128, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)  # [b, 2, 2048]
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)  # [b, 256, 1]
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)

        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
            # identity_matrix = identity_matrix.to('cpu')

        x = x.view(-1, self.output_dim, self.output_dim)
        x = x + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension, return_local_features=False, dataset='', global_feat_dim=256):
        super(BasePointNet, self).__init__()
        self.global_feat_dim = global_feat_dim
        self.dataset = dataset
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(7, 64, 1, bias=False)  # 7 channels to take I, NDVI, RGB into account
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, self.global_feat_dim, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(self.global_feat_dim)

    def forward(self, x):
        num_points = x.shape[1]  # torch.Size([BATCH, SAMPLES, DIMS])

        x_tnet = x[:, :, :2]  # only apply T-NET to x and y
        input_transform = self.input_transform(x_tnet)
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product
        x_tnet = torch.cat([x_tnet, x[:, :, 2].unsqueeze(2), x[:, :, 4].unsqueeze(2)], dim=2)  # concat z and intensity
        x_tnet = torch.cat([x_tnet, x[:, :, 6].unsqueeze(2), x[:, :, 7].unsqueeze(2), x[:, :, 9].unsqueeze(2)],
                           dim=2)  # concat Green Blue NDVI
        x_tnet = x_tnet.transpose(2, 1)  # [batch, dims, n_points]

        x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 64, 2000]
        x = x.transpose(2, 1)  # [batch, 2000, 64]

        feature_transform = self.feature_transform(x)  # [batch, 64, 64]

        x = torch.bmm(x, feature_transform)
        local_point_features = x  # [batch, 2000, 64]

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, self.global_feat_dim)  # [ batch, 1024, 1]

        if self.return_local_features:
            global_feature = global_feature.view(-1, self.global_feat_dim, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class GRUPointNet(nn.Module):

    def __init__(self, point_dimension=3, hidden_size=128, global_feat_size=256, num_att_heads=2):
        super(GRUPointNet, self).__init__()
        self.hidden_size = hidden_size
        self.global_feat_size = global_feat_size
        self.att_heads = num_att_heads

        self.base_pointnet = BasePointNet(return_local_features=True, point_dimension=point_dimension,
                                          global_feat_dim=global_feat_size)
        self.gru_global = nn.GRU(self.global_feat_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x, hidden):
        local_global_features, feature_transform = self.base_pointnet(x)  # [b, np, 320] [b, 64, 64]
        local_features = local_global_features[:, :, -64:]  # [batch, n_point, 64]
        global_feature = local_global_features[:, 0, :-64]  # [batch, 256]

        global_feature = global_feature.view(-1, 1, self.global_feat_size)
        out_h, hidden = self.gru_global(global_feature, hidden)  # [batch, 1, 512] [2, b, 256]
        out_h = out_h.view(-1, self.hidden_size * self.att_heads)  # [batch, 512]
        # out_h = self.fc_hidden(out_h)  # if we want to change out_h size we add another layer

        return out_h, feature_transform, local_features

    def initHidden(self, x, device):
        """tensor of shape (D * {num_layers}, N, H_out) containing the initial hidden state for the input sequence.
        Defaults to zeros if not provided """
        return torch.zeros(2, x.shape[0], self.hidden_size, device=device)  # [layers, x.size[0], hidden]


class ClassificationWithAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes=2, dropout=0.3, num_w=5):
        super(ClassificationWithAttention, self).__init__()
        self.embed_dim = embed_dim  # todo test 2046, 3072

        self.attention = nn.MultiheadAttention(embed_dim,
                                               num_heads=num_heads,  # todo test 10
                                               dropout=dropout)

        self.conv_1 = nn.Conv1d(num_w, 1, 1, bias=True)
        self.fc_2 = nn.Linear(embed_dim, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.bn_2 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):

        attn_output, attn_output_weights = self.attention(x, x, x,
                                                          key_padding_mask=None,
                                                          need_weights=True,
                                                          attn_mask=attn_mask)
        # [w_len, b, 512] [b, w_len, w_len]
        attn_output = attn_output.view(-1, attn_output.shape[0], self.embed_dim)  # [b, w_len, 512]

        x = F.relu(self.conv_1(attn_output))  # [b, 1, 512]
        x = x.view(-1, self.embed_dim)
        x = F.relu(self.bn_2(self.fc_2(x)))
        out = self.fc_3(x)

        return out, attn_output_weights


class SegmentationWithAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes=2, dropout=0.3, num_w=5):
        super(SegmentationWithAttention, self).__init__()
        self.embed_dim = embed_dim  # todo test 2046, 3072

        self.attention = nn.MultiheadAttention(embed_dim,
                                               num_heads=num_heads,  # todo test 10
                                               dropout=dropout)

        self.conv_1 = nn.Conv1d(num_w, 1, 1, bias=True)
        self.fc_2 = nn.Linear(embed_dim, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.bn_2 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):

        attn_output, attn_output_weights = self.attention(x, x, x,
                                                          key_padding_mask=None,
                                                          need_weights=True,
                                                          attn_mask=attn_mask)
        # [w_len, b, 512] [b, w_len, w_len]
        attn_output = attn_output.view(-1, attn_output.shape[0], self.embed_dim)  # [b, w_len, 512]

        x = F.relu(self.conv_1(attn_output))  # [b, 1, 512]
        x = x.view(-1, self.embed_dim)
        x = F.relu(self.bn_2(self.fc_2(x)))
        out = self.fc_3(x)

        return out, attn_output_weights


class RNNSegmentationPointNet(nn.Module):

    def __init__(self, num_classes):
        super(RNNSegmentationPointNet, self).__init__()

        self.conv_1 = nn.Conv1d(320, 256, 1)
        self.conv_2 = nn.Conv1d(256, 128, 1)
        self.conv_3 = nn.Conv1d(128, 64, 1)

        self.conv_4 = nn.Conv1d(64, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)

        # self.fc = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_cls, local_feat):
        hid_size = hidden_cls.shape[2]
        global_feat_rnn = torch.cat((hidden_cls[0, :, :], hidden_cls[1, :, :]), 1)
        # global_feat_rnn = global_feat_rnn.view(-1, 1, global_feat_rnn.shape[1])

        # global_feat_rnn shape [b,1,hidden]
        num_points = local_feat.shape[1]

        global_feat_rnn = global_feat_rnn.view(-1, global_feat_rnn.shape[1], 1).repeat(1, 1, num_points)  # [b, h, n_points]
        local_global_rnn = torch.cat([global_feat_rnn.transpose(2, 1), local_feat], 2)

        # local_global_rnn  # [batch, n_points, 1088]
        x = local_global_rnn.transpose(2, 1)  # [batch, 1088, n_points]
        x = F.relu(self.bn_1(self.conv_1(x)))  # [batch, 512, n_points]
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 256, n_points]
        x = F.relu(self.bn_3(self.conv_3(x)))  # [batch, 128, n_points]

        x = self.conv_4(x)
        out = x.transpose(2, 1)
        # out = self.softmax(x)

        return out

    # def initHidden(self, x, device):
    #     """tensor of shape (D * {num_layers}, N, H_out) containing the initial hidden state for the input sequence.
    #     Defaults to zeros if not provided """
    #     return torch.zeros(2, x.shape[0], self.hidden_size, device=device)  # [layers, x.size[0], hidden]
