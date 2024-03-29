import torch
import torch.nn as nn
import torch.nn.functional as F
from pointNet_2.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim, device):
        super(TransformationNet, self).__init__()
        self.device = device
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
        identity_matrix = identity_matrix.to(self.device)

        x = x.view(-1, self.output_dim, self.output_dim)
        x = x + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension=2, return_local_features=False, global_feat_dim=256, device='cuda'):
        """
        point_dimension: The number of dimensions of each point in the point cloud.
        return_local_features: A boolean that determines whether the network should return local features or not.
        global_feat_dim: The dimension of the global feature vector that the network should output.
        device: The device (CPU or GPU) on which the network should run.
        """
        super(BasePointNet, self).__init__()
        self.global_feat_dim = global_feat_dim
        self.point_dimension = point_dimension
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension, device=device)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)

        self.conv_1 = nn.Conv1d(9 + point_dimension, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, 128, 1, bias=False)
        self.conv_6 = nn.Conv1d(128, self.global_feat_dim, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(128)
        self.bn_6 = nn.BatchNorm1d(self.global_feat_dim)

    def forward(self, x):
        num_points = x.shape[1]  # torch.Size([BATCH, SAMPLES, DIMS])

        x_tnet = x[:, :, :self.point_dimension]  # only apply T-NET to x and y
        input_transform = self.input_transform(x_tnet)
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product
        x_tnet = torch.cat([x_tnet, x], dim=2)  # concat z and intensity

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
        x = F.relu(self.bn_6(self.conv_6(x)))
        x = nn.MaxPool1d(num_points)(x) #+ nn.AvgPool1d(num_points)(x)

        global_feature = x.view(-1, self.global_feat_dim)  # [ batch, 1024, 1]

        if self.return_local_features:
            global_feature = global_feature.view(-1, self.global_feat_dim, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class ClassificationWithAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes=2, dropout=0.3, num_w=9):
        super(ClassificationWithAttention, self).__init__()
        self.embed_dim = embed_dim

        # self.fc1_pe = nn.Linear(2, 16)
        # self.fc2_pe = nn.Linear(16, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout)

        self.conv_1 = nn.Conv1d(num_w, 1, 1, bias=True)
        self.fc_2 = nn.Linear(embed_dim, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.bn_2 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, gl_feats, centroids, attn_mask=None):
        # positional_encoding
        # pos_enc = self.fc2_pe(F.leaky_relu_(self.fc1_pe(centroids)))
        # pos_enc = pos_enc.transpose(0, 1)
        gl_feats = gl_feats  # + pos_enc

        attn_output, attn_output_weights = self.attention(gl_feats, gl_feats, gl_feats,
                                                          key_padding_mask=attn_mask,
                                                          need_weights=True,
                                                          attn_mask=None)
        # [w_len, b, 512] [b, w_len, w_len]
        attn_output = attn_output.view(-1, attn_output.shape[0], self.embed_dim)  # [b, w_len, 512]

        x = F.relu(self.conv_1(attn_output))  # [b, 1, 512]
        x = x.view(-1, self.embed_dim)
        x = F.relu(self.bn_2(self.fc_2(x)))
        out = self.fc_3(x)

        return out, attn_output_weights


class SegmentationWithAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes=2, local_dim=128, dropout=0.3, device='cuda'):
        super(SegmentationWithAttention, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout)  # batch_first = False

        self.conv_2 = nn.Conv1d(local_dim + embed_dim, int(embed_dim / 2), 1)
        self.conv_3 = nn.Conv1d(int(embed_dim / 2), 64, 1)
        self.conv_4 = nn.Conv1d(64, num_classes, 1)

        self.dropout = nn.Dropout(dropout)

        self.bn_2 = nn.BatchNorm1d(int(embed_dim / 2))
        self.bn_3 = nn.BatchNorm1d(64)

    def forward(self, gl_feats, lo_feats, centroids, np_cluster, attn_mask=None):
        # gl_feats [n_clusters, b, 256]
        # lo_feats [b, n_points, 64]
        # centroids [b, n_clusters, 2]
        # np_cluster [2048, 2048, 2048, 2048, 2048]

        # positional_encoding
        pos_enc = self.fc2(F.leaky_relu_(self.fc1(centroids)))
        pos_enc = pos_enc.transpose(0, 1)
        gl_feats = gl_feats + pos_enc

        gl_feats, attn_output_weights = self.attention(gl_feats, gl_feats, gl_feats,
                                                       key_padding_mask=attn_mask,
                                                       need_weights=True,
                                                       attn_mask=None, )
        # [w_len, b, 512] [b, w_len, w_len]
        global_feat = torch.FloatTensor().to(self.device)
        # loop over windows to repeat global feature tensor as many times as number of points per cluster
        for i in range(gl_feats.shape[0]):
            h_cluster = gl_feats[i, :, :].view(-1, 1, gl_feats.shape[2])
            h_cluster = h_cluster.repeat(1, np_cluster[i], 1)
            global_feat = torch.cat((global_feat, h_cluster), dim=1)

        # concatenate local features and global hidden outputs
        pc_embed = torch.cat((lo_feats, global_feat), dim=2)  # [b, points, 320]
        pc_embed = pc_embed.transpose(2, 1)

        out = F.relu(self.bn_2(self.conv_2(pc_embed)))
        out = self.dropout(out)
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = self.dropout(out)
        out = self.conv_4(out)

        return out, 0  # attn_output_weights


class SegmentationWithGRU(nn.Module):

    def __init__(self, num_classes, global_feat_size, hidden_size, device):
        super(SegmentationWithGRU, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.gru_global = nn.GRU(global_feat_size, hidden_size, batch_first=True, bidirectional=False)

        self.conv_2 = nn.Conv1d(64 + 64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 64, 1)
        self.conv_4 = nn.Conv1d(64, num_classes, 1)

        self.dropout = nn.Dropout(0.3)

        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)

    def forward(self, global_seq, local_feats, np_cluster):
        # init hidden
        hidden = self.initHidden(global_seq)
        out_h, _ = self.gru_global(global_seq, hidden)  # (b,L,h_o)
        # out_h shape: [batch, 5, 256]

        global_feat = torch.FloatTensor().to(self.device)
        # loop over windows to repeat global feature tensor as many times as number of points per cluster
        for i in range(out_h.shape[1]):
            h_cluster = out_h[:, i, :].view(-1, 1, out_h.shape[2])
            h_cluster = h_cluster.repeat(1, np_cluster[i], 1)
            global_feat = torch.cat((global_feat, h_cluster), dim=1)

        # concatenate local features and global hidden outputs
        pc_embed = torch.cat((local_feats, global_feat), dim=2)  # [b, points, 320]
        pc_embed = pc_embed.transpose(2, 1)

        out = F.relu(self.bn_2(self.conv_2(pc_embed)))
        out = self.dropout(out)
        out = F.relu(self.bn_3(self.conv_3(out)))
        out = self.dropout(out)
        out = self.conv_4(out)

        return out

    def initHidden(self, x):
        """tensor of shape (D * {num_layers}, N, H_out) containing the initial hidden state for the input sequence.
        Defaults to zeros if not provided """
        return torch.zeros(1, x.shape[0], self.hidden_size, device=self.device)  # [layers, x.size[0], hidden]


class ClassificationFromGRU(nn.Module):

    def __init__(self, num_classes=2, dropout=0.3, num_w=5, embed_dim=256):
        super(ClassificationFromGRU, self).__init__()

        self.conv_1 = nn.Conv1d(num_w, 1, 1, bias=True)
        self.fc_2 = nn.Linear(embed_dim, 128)
        self.fc_3 = nn.Linear(128, num_classes)

        self.bn_2 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv_1(x))  # [b, 1, 512]
        x = x.view(-1, self.embed_dim)
        x = F.relu(self.bn_2(self.fc_2(x)))
        out = self.fc_3(x)

        return out


class pointnet_2(nn.Module):
    def __init__(self, num_classes):
        super(pointnet_2, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        # self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        num_points = xyz.shape[2]
        l0_points = xyz  # [b, n_pooints, D]
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [b, 3, 1024] [b, D, n_points]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [b, 3, 256] [b, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [b, 3, 64] [b, 256, 64]
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [b, 3, 16] [b, 512, 16]

        # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [b, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [b, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [b, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [b, 128, n_points]

        x = self.conv1(l0_points)  # [b, 128, n_points]

        x = nn.MaxPool1d(num_points)(x)  # + nn.AvgPool1d(num_points)(x)
        global_feature = x.view(-1, 128)  # [ batch, 128, 1]

        # x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return global_feature, l0_points
