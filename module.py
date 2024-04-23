from kit.op import *
from kit.nn import *


class AWDS(nn.Module):
    def __init__(self, channel):
        super(AWDS, self).__init__()

        self.mini_emb = MiniEmbedding(in_channel=3, local_emb_size=16, mlps=[channel//4, channel//2, channel], relu=[True, True, True])
        self.pt = PT(channel=128, n_layers=2)

    def forward(self, batch_x, K):

        ## Bone Sampling
        bones, local_windows = SamplingAndQuery(batch_x, K)
        # get bones: (M, 3); local_windows: (M, K, 3)

        ## Adaptive Aligning
        aligned_windows = AdaptiveAligning(local_windows, bones)
        # get aligned_windows: (M, K, 3)
        
        ## mini_emb
        skin_features = self.mini_emb(aligned_windows)
        skin_features = self.pt(aligned_windows, skin_features)
        # get skin features: (M, K, C)

        return skin_features, bones
    

class DWBuild(nn.Module):
    def __init__(self, k):
        super(DWBuild, self).__init__()
        self.k = k

    def forward(self, bones):
        '''
        input:
            bones: (M, 3)
        '''
        bones = bones.unsqueeze(0) # -> (1, M, 3)
        _, dilated_idx, dilated_windows = knn_points(bones, bones, K=self.k, return_nn=True)

        dilated_windows = dilated_windows - bones.unsqueeze(2)
        dilated_windows = dilated_windows[0] # -> (M, k, 3)
        dilated_windows = op.n_scale_batch(dilated_windows)  

        return dilated_idx, dilated_windows


class DWEM(nn.Module):
    def __init__(self, channel, bottleneck_channel):
        super(DWEM, self).__init__()
        self.bottleneck_channel = bottleneck_channel

        self.graph_conv = GraphConv(in_channel=3, mlps=[channel//4, channel//2, channel], relu=[True, True, True])
        self.regression_head = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, bottleneck_channel*2),
        )

    def forward(self, dilated_windows):
        '''
        input:
            dilated_windows: (M, k, 3)
        ''' 
        feature = self.graph_conv(dilated_windows) # -> (M, C)
        mu_sigma = self.regression_head(feature) # -> (M, c*2)
        mu, sigma = mu_sigma[:, :self.bottleneck_channel], torch.exp(mu_sigma[:, self.bottleneck_channel:])
        return mu, sigma
    

class FeatureRefinement(nn.Module):
    def __init__(self, channel, n_conv_layer):
        super(FeatureRefinement, self).__init__()
        self.n_conv_layer = n_conv_layer
        self.dw_conv_ls = nn.ModuleList()
        for i in range(n_conv_layer):
            self.dw_conv_ls.append(DWConv(channel=channel))
        self.linear = nn.Linear(channel*(n_conv_layer+1), channel)

    def forward(self, skin_features, dilated_windows, dilated_idx):
        """
        Input:
            skin_features: (M, C)
            dilated_windows: (M, k, 3)
            dilated_idx: (1, M, k)
        Return:
            refined_skin_fea: (M, C)
        """
        fea_ls = [skin_features]
        for dw_conv_block in self.dw_conv_ls:
            skin_features = dw_conv_block(skin_features, dilated_windows, dilated_idx)
            fea_ls.append(skin_features)
        fea = torch.cat(fea_ls, dim=-1)
        refined_skin_fea = self.linear(fea)
        return refined_skin_fea


class PointGenerator(nn.Module):
    def __init__(self, channel, fold_channel, R_max, r):
        super(PointGenerator, self).__init__()
        self.R_max = R_max
        self.r = r
        self.folding_base = Folding(in_channel=channel, fold_ratio=R_max, out_channel=fold_channel)
        self.folding_pro = Folding(in_channel=channel+fold_channel, fold_ratio=r, out_channel=3)

    def forward(self, skin_features, K):
        """
        Input:
            skin_features: (M, C)
        """
        M = skin_features.shape[0]

        # generate fea matrix
        fea = self.folding_base(skin_features)
        # get fea: (M, R_max, fold_channel)

        # sampling
        fea = fea[:, torch.randperm(self.R_max)[:K//self.r], :]
        # get fea: (M, K//r, fold_channel)

        # generate xyz
        skin_features = skin_features.unsqueeze(1).repeat((1, fea.shape[1], 1))
        cat_fea = torch.cat((skin_features, fea), dim=-1)
        # get cat_fea: (M, K//r, fold_channel+channel)

        xyz = self.folding_pro(cat_fea)
        # get xyz: (M, K//r, r, 3)
        xyz = xyz.view(M, -1, 3)

        return xyz


class DWUS(nn.Module):
    def __init__(self, channel, fold_channel, R_max, r):
        super(DWUS, self).__init__()
        self.fea_refine = FeatureRefinement(channel=channel, n_conv_layer=2)
        self.point_generator = PointGenerator(channel, fold_channel, R_max=R_max, r=r)

    def forward(self, skin_features, bones, dilated_windows, dilated_idx, K):
        """
        Input:
            skin_features: (M, C)
            dilated_windows: (M, k, 3)
            dilated_idx: (1, M, k)
        Return:
            rec_xyz: (1, M*K, 3)
        """
        refined_skin_fea = self.fea_refine(skin_features, dilated_windows, dilated_idx)
        # refined_skin_fea: (M, C)
        rec_windows = self.point_generator(refined_skin_fea, K)
        # get rec_windows: (M, K, 3)

        # inverse aligning
        rec_windows = InverseAligning(rec_windows, bones)
        rec_xyz = rec_windows.view(1, -1, 3)
        return rec_xyz
