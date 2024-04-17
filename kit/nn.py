import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_gather, knn_points

import kit.op as op


class GraphConv(nn.Module):
    def __init__(self, in_channel, mlps, relu):
        super(GraphConv, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                    mlp_Module = nn.Sequential(
                        nn.Linear(mlps[i], mlps[i+1]),
                        nn.ReLU(inplace=True),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Linear(mlps[i], mlps[i+1]),
                    )
            self.mlp_Modules.append(mlp_Module)

    def forward(self, points):
        """
        Input:
            points: input points position data, [B, ..., N, C]
        Return:
            points: feature data, [B, ..., D]
        """

        for m in self.mlp_Modules:
            points = m(points)
    
        points = torch.max(points, -2, keepdim=False)[0]

        return points


class MiniEmbedding(nn.Module):
    def __init__(self, in_channel, local_emb_size, mlps, relu):
        super(MiniEmbedding, self).__init__()
        self.local_emb_size = local_emb_size
        self.graph_conv = GraphConv(in_channel=in_channel, mlps=mlps, relu=relu)

    def forward(self, windows):
        '''
        input: 
            windows: (M, K, 3)
        output:
            features: (M, K, C)
        '''
        M, K, _ = windows.shape

        _, _, grouped_windows = knn_points(windows, windows, K=self.local_emb_size, return_nn=True)
        # get grouped_windows: (M, K, local_emb_size, 3)
        grouped_windows = grouped_windows - windows.unsqueeze(-2)
        grouped_windows = op.n_scale_batch(grouped_windows.view(M*K, self.local_emb_size, 3)).view(M, K, self.local_emb_size, 3)

        grouped_features = self.graph_conv(grouped_windows)
        # get grouped_features: (M, K, C)
        
        return grouped_features


class SelfAttention(nn.Module):
    def __init__(self, channel):
        super(SelfAttention, self).__init__()
        self.channel = channel
        self.k_mlp = nn.Linear(channel, channel)
        self.v_mlp = nn.Linear(channel, channel)
        self.pe_multiplier, self.pe_bias = True, True
        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, channel),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, channel),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
        )
        self.residual_emb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channel, channel),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, patches, feature):
        '''
        input:
            patches: (M, K, 3)
            feature: (M, K, C)
        output:
            feature: (M, K, C)
        '''

        key = self.k_mlp(feature) # (M, K, C)
        value = self.v_mlp(feature) # (M, K, C)

        relation_qk = key - key[:, 0:1, :]
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(patches)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(patches)
            relation_qk = relation_qk + peb
            value = value + peb

        weight  = self.weight_encoding(relation_qk)
        score = self.softmax(weight)

        feature = score*value
        feature = self.residual_emb(feature)

        return feature


class PT(nn.Module):
    def __init__(self, channel, n_layers):
        super(PT, self).__init__()
        self.channel = channel
        self.n_layers = n_layers
        self.sa_ls, self.sa_emb_ls = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            self.sa_emb_ls.append(nn.Sequential(
                nn.Linear(channel, channel),
                nn.ReLU(),
            ))
            self.sa_ls.append(SelfAttention(channel))
        self.base_emb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
        )
    def forward(self, patches, patches_emb, patches_emb_base=None, weights=None):
        """
        Input:
            patches: input points position data, [M, K, 3]
            patches_emb: input points feature data, [M, K, D]
        Return:
            feature: output feature data, [M, C]
        """
        feature = patches_emb
        if patches_emb_base is not None:
            feature += self.base_emb(patches_emb_base)
        for i in range(self.n_layers):
            identity = feature
            feature = self.sa_emb_ls[i](feature) # -> (M, K, C)
            output = self.sa_ls[i](patches, feature) # -> (M, K, C)
            feature = output + identity # -> (M, K, C)
        if weights is not None:
            feature = feature * weights.unsqueeze(-1)
        feature = feature.max(dim=-2)[0] # -> (M, C)
        return feature


class DWConv(nn.Module):
    def __init__(self, channel):
        super(DWConv, self).__init__()

        self.graph_conv = GraphConv(channel+3, mlps=[channel], relu=[True])

    def forward(self, skin_features, dilated_windows, dilated_idx):
        """
        Input:
            skin_features: (M, C)
            dilated_windows: (M, k, 3)
            dilated_idx: (1, M, k)
        Return:
            enhanced_skin_fea: (M, C)
        """
        grouped_skin_features = knn_gather(skin_features.unsqueeze(0), dilated_idx)[0]
        grouped_skin_features = torch.cat((grouped_skin_features, dilated_windows), dim=-1) 
        # get grouped_skin_features: (M, k, channel+3)
        
        enhanced_skin_fea = self.graph_conv(grouped_skin_features) 
        # get skin features: (M, C)

        return enhanced_skin_fea


class Folding(nn.Module):
    def __init__(self, in_channel, fold_ratio, out_channel):
        super(Folding, self).__init__()
        self.fold_ratio = fold_ratio
        self.out_channel = out_channel
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, fold_ratio*out_channel),
        )

    def forward(self, fea):
        """
        Input:
            fea: (..., in_channel)
        Output:
            fea: (..., fold_ratio, out_channel)
        """
        output_shape = fea.shape[:-1]+(self.fold_ratio,self.out_channel,)
        fea = self.mlp(fea).reshape(output_shape)
        return fea
