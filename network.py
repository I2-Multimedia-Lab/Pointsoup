import torch
import torch.nn as nn

import module
import kit.op as op

class Pointsoup(nn.Module):
    def __init__(self, k, channel, bottleneck_channel):
        super(Pointsoup, self).__init__()

        # Encoder
        self.awds = module.AWDS(channel=channel)
        self.fea_squeeze = nn.Linear(channel, bottleneck_channel)

        # # Entropy Modeling
        self.dw_build = module.DWBuild(k=k)
        self.dwem = module.DWEM(channel=channel, bottleneck_channel=bottleneck_channel)

        # Decoder
        self.fea_stretch = nn.Linear(bottleneck_channel, channel)
        self.dwus = module.DWUS(channel=channel, fold_channel=8, R_max=256, r=4)

    def forward(self, batch_x, K):
        '''
        input: 
            batch_x: (1, N, 3)
            K: int, local window size
        output:
            rec_batch_x: (1, *, 3)
            bitrate: bpp for skin features
        '''
        N = batch_x.shape[1]

        ##################### Encoder #####################
        ## AWDS
        skin_fea, bones = self.awds(batch_x, K)
        # get skin features: (M, C)

        ## Feature Squeezing
        compact_fea = self.fea_squeeze(skin_fea)
        # get compact features: (M, c)

        ##################### Entropy Model #####################
        ## Quantization
        quantized_compact_fea = compact_fea + torch.nn.init.uniform_(torch.zeros_like(compact_fea), -0.5, 0.5)

        ## DW-Build & DWEM
        dilated_idx, dilated_windows = self.dw_build(bones)
        mu, sigma = self.dwem(dilated_windows)
        bitrate, _ = op.feature_probs_based_mu_sigma(quantized_compact_fea, mu, sigma)
        bitrate = bitrate / N

        ##################### Decoder #####################
        rec_skin_fea = self.fea_stretch(quantized_compact_fea)
        # get rec_skin_fea: (M, C)
        rec_batch_x = self.dwus(rec_skin_fea, bones, dilated_windows, dilated_idx, K)
        # get rec_batch_x: (M*K, 3)

        return rec_batch_x, bitrate
