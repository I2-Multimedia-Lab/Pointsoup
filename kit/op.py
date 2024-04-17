import os
import math
import time
import subprocess

import numpy as np

import torch
import torchac

from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.ops.sample_farthest_points import sample_farthest_points


class Recoder:
    def __init__(self):
        self.ls = []

    def update(self, value):
        self.ls.append(value)
    
    def dump_avg(self, precision=5):
        avg_value = round(np.array(self.ls).mean(), precision)
        self.ls = []
        return avg_value


class Ticker:
    def __init__(self):
        self.dict = {}
    
    def start_count(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time()
    
    def end_count(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time() - self.dict[label]

    def set_time(self, label, value):
        self.dict[label] = value
    
    def get_time(self, label, precision=3):
        return round(self.dict[label], precision)

    def dump_sum(self, precision=3):
        t = 0
        for key in self.dict.keys():
            t += self.dict[key]
        t = round(t, precision)
        self.dict = {}
        return t


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##################################################
##################################################


def get_self_cd(pos):
    '''
    input:
        pos: (B, N, 3)
    output:
        dist: (B, N)
    '''
    dist = knn_points(pos, pos, K=2, return_nn=False).dists[:,:,1]
    dist = torch.sqrt(dist)
    return dist


def n_scale_batch(x_windows, margin=0.01):

    device = x_windows.device
    M, K, _ = x_windows.shape

    x, y, z = x_windows[:, :, 0], x_windows[:, :, 1], x_windows[:, :, 2]
    x_max, x_min, y_max, y_min, z_max, z_min = x.max(dim=1)[0], x.min(dim=1)[0], y.max(dim=1)[0], y.min(dim=1)[0], z.max(dim=1)[0], z.min(dim=1)[0]
    x_max, x_min, y_max, y_min, z_max, z_min = x_max.unsqueeze(-1), x_min.unsqueeze(-1), y_max.unsqueeze(-1), y_min.unsqueeze(-1), z_max.unsqueeze(-1), z_min.unsqueeze(-1)
    
    longest = torch.max(torch.cat([x_max-x_min, y_max-y_min, z_max-z_min], dim=1), dim=1)[0].to(device)
    scaling = (1-margin) / longest
    
    x_windows = x_windows * scaling.view(M, 1, 1)

    return x_windows


def reorder(points, ref_points):
    '''
    Input:
        points: 
        ref_points: 
    '''

    dist = torch.cdist(points.cpu(), ref_points.cpu())
    cloest_idx = torch.argmin(dist, dim=0).cuda()

    return cloest_idx

##################################################
##################################################


def SamplingAndQuery(batch_x, K):
    '''
    input:
        batch_x: (1, N, 3)
        K: local window size
    output:
        bones: (M, 3)
        local_windows: (M, K, 3)
    '''
    _, N, _ = batch_x.shape
    M = N*2//K

    # Sampling
    if N < 10000:
        bones = sample_farthest_points(batch_x, K=M)[0] # (1, M, 3)
    else:
        sample_anchor = batch_x.clone()[:, torch.randperm(N)[:M*16], :]
        bones = sample_farthest_points(sample_anchor, K=M)[0] # (1, M, 3)

    # Query
    _, _, local_windows = knn_points(bones, batch_x, K=K, return_nn=True)
    bones, local_windows = bones[0], local_windows[0]

    return bones, local_windows
    

def AdaptiveAligning(local_windows, bones):
    n_local_windows = local_windows - bones.unsqueeze(-2)
    sampled_self_dist = get_self_cd(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows / sampled_self_dist # -> (M, K, 3)
    return n_local_windows


def InverseAligning(n_local_windows, bones):
    sampled_self_dist = get_self_cd(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows * sampled_self_dist # -> (M, K, 3)
    local_windows = n_local_windows + bones.unsqueeze(-2)
    return local_windows


##################################################
##################################################


def feature_probs_based_mu_sigma(feature, mu, sigma):
    sigma = sigma.clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, probs


def get_cdf_min_max_v(mu, sigma, L):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, L)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, L).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, L).to(sigma.device).view(1, 1, L).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    # print(cdf_with_0.shape)
    return cdf_with_0


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
  """
  From torchac
  """
  Lp = cdf_float.shape[-1]
  factor = torch.tensor(
    2, dtype=torch.float32, device=cdf_float.device).pow_(16)
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float.mul(new_max_value)
  cdf_float = cdf_float.round()
  cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
  if needs_normalization:
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
  return cdf


def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8


##################################################
##################################################


def tmc_compress(tmc_path, input_file, output_file):
    """
        Compress point cloud losslessly using MPEG G-PCCv23 PredTree. 
    """
    cmd = (tmc_path+ 
            ' --mode=0' + 
            ' --geomTreeType=1' + # 0 for octree, 1 for predtree
            ' --mergeDuplicatedPoints=1' +
            f' --positionQuantizationScale=1' +
            ' --neighbourAvailBoundaryLog2=8' +
            ' --intra_pred_max_node_size_log2=6' +
            ' --inferredDirectCodingMode=0' +
            f' --uncompressedDataPath='+input_file +
            f' --compressedStreamPath='+output_file)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    slice_number = int(str(output).split('Slice number:')[1].split('\\n')[0])

    xyz_steam_size, xyz_time = 0, 0
    for i in range(slice_number):
        xyz_steam_size += float(str(output).split('positions bitstream size')[i+1].split('B')[0]) * 8
        xyz_time += float(str(output).split('positions processing time (user):')[i+1].split('s')[0])

    return xyz_steam_size, xyz_time


def tmc_decompress(tmc_path, input_file, output_file):
    """
        Decompress point cloud using MPEG G-PCCv23. 
    """
    cmd = (tmc_path+ 
        ' --mode=1'+ 
        ' --compressedStreamPath='+input_file+ 
        ' --reconstructedDataPath='+output_file)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    xyz_time = float(str(output).split('positions processing time (user):')[1].split('s')[0])
    return xyz_time

