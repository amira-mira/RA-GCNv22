import torch
from torch import nn
from torch.nn import functional as F

import src.utils as U


class RA_GCN(nn.Module):
    def __init__(self, data_shape, num_class, A, drop_prob, gcn_kernel_size, model_stream, subset, pretrained):
        super().__init__()

        C, T, V, M = data_shape
        self.register_buffer('A', A)

        # baseline
        self.stgcn_stream = nn.ModuleList((
            ST_GCN(data_shape, num_class, A, drop_prob, gcn_kernel_size) 
            for _ in range(model_stream)
        ))

        # load pretrained baseline
        if pretrained:
            for stgcn in self.stgcn_stream:
                checkpoint = U.load_checkpoint('baseline_NTU' + subset)
                stgcn.load_state_dict(checkpoint['model'])

        # mask
        self.mask_stream = nn.ParameterList([
            nn.Parameter(torch.ones(T * V * M)) 
            for _ in range(model_stream)
        ])

    def forward(self, inp):

        # multi stream
        out = []
        feature = []
        for stgcn, mask in zip(self.stgcn_stream, self.mask_stream):
            x = inp

            # mask
            N, C, T, V, M = x.shape
            x = x.view(N, C, -1)
            x = x * mask[None,None,:]
            x = x.view(N, C, T, V, M)

            # baseline
            temp_out, temp_feature = stgcn(x)

            # output
            out.append(temp_out.unsqueeze(-1))
            feature.append(temp_feature[0])
        return out, feature


class ST_GCN(nn.Module):
    def __init__(self, data_shape, num_class, A, drop_prob, gcn_kernel_size):
        super().__init__()

        C, T, V, M = data_shape
        self.register_buffer('A', A)

        # data normalization
        self.data_bn = nn.BatchNorm1d(C * V * M)

        # st-gcn networks
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_layer(C, 64, gcn_kernel_size, 1, A, drop_prob, residual=False),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 128, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 256, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
        ))

        # edge importance weights
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(A.shape)) for _ in self.st_gcn_networks])

        # fcn
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # extract feature
        feature = []
        _, c, t, v = x.shape
        feature.append(x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1))

        # global pooling
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(N, -1)

        return x, feature


class st_gcn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, A, drop_prob=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # spatial network
        self.gcn = SpatialGraphConv(in_channels, out_channels, kernel_size[1]+1)

        # temporal network
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0],1), (stride,1), padding),
            nn.BatchNorm2d(out_channels),
        )

        # residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        # output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        # residual
        res = self.residual(x)

        # spatial gcn
        x = self.gcn(x, A)

        # temporal 1d-cnn
        x = self.tcn(x)

        # output
        x = self.relu(x + res)
        return x


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = s_kernel_size

        # weights of different spatial classes
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel_size, kernel_size=1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.conv(x)

        # divide into different classes
        n, kc, t, v = x.shape
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()
        return x
