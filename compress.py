import os
import random
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

import kit.io as io
import kit.op as op
import network

import warnings
warnings.filterwarnings("ignore")

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress point clouds.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/example_pcs/*.ply')
parser.add_argument('--compressed_path', type=str, help='Path to save .bin files.', default='./data/compressed/')
parser.add_argument('--model_load_path', type=str, help='Directory where to load trained models.', default=f'./model/exp/ckpt.pt')
parser.add_argument('--tmc_path', type=str, help='TMC to compress bone points.', default='./tmc3')

parser.add_argument('--local_window_size', type=int, help='Local window size.', default=128)
parser.add_argument('--verbose', type=bool, help='Print compression details.', default=False)

parser.add_argument('--dilated_window_size', type=int, help='Dilated window size. (Same value with train.py)', default=8)
parser.add_argument('--channel', type=int, help='Network channel. (Same value with train.py)', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel. (Same value with train.py)', default=16)

args = parser.parse_args()

if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

model = network.Pointsoup(k=args.dilated_window_size,
                          channel=args.channel, 
                          bottleneck_channel=args.bottleneck_channel)
model.load_state_dict(torch.load(args.model_load_path))
model = torch.compile(model)
model = model.cuda().eval()

# warm up our model, since the first step of the model is very slow
model(torch.randn(1, 1024, 3).cuda(), 128)

files = np.array(glob(args.input_glob, recursive=True))

time_recoder, bpp_recoder = op.Recoder(), op.Recoder()
ticker = op.Ticker()

with torch.no_grad():
    for file_path in tqdm(files):

        pc = io.read_point_cloud(file_path)
        batch_x = torch.tensor(pc).unsqueeze(0).cuda()
        N = batch_x.shape[1]

        filename_w_ext = os.path.split(file_path)[-1]
        compressed_head_path = os.path.join(args.compressed_path, filename_w_ext+'.h.bin')
        compressed_bones_path = os.path.join(args.compressed_path, filename_w_ext+'.b.bin')
        compressed_skin_path = os.path.join(args.compressed_path, filename_w_ext+'.s.bin')
        
        if args.verbose:
            print('*'*30)
            print('Processing:', filename_w_ext, f'({N} points)')

        ######################################################
        ######################## AWDS ########################
        ######################################################

        ############## 🚩 SamplingAndQuery ##############

        ticker.start_count('SamplingAndQuery') # 🕒 ⏳

        bones, local_windows = op.SamplingAndQuery(batch_x, K=args.local_window_size)
        # we get bones: (M, 3); local windows: (M, K, 3)

        ticker.end_count('SamplingAndQuery') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Sampling and query:', ticker.get_time('SamplingAndQuery'), 's')

        ############## 🚩 Bone Compression ##############
        # (io time is omitted since the tmc process can be done in RAM in practial applications)
        cache_file_path = os.path.join(args.compressed_path, '__cache__.ply')
        io.save_point_cloud(bones, cache_file_path)
        bone_steam_size, bone_enc_time = op.tmc_compress(args.tmc_path, cache_file_path, compressed_bones_path)
        bone_dec_time = op.tmc_decompress(args.tmc_path, compressed_bones_path, cache_file_path)
        rec_bones = torch.tensor(io.read_point_cloud(cache_file_path)).float().cuda() # -> (M, 3)
        
        ticker.set_time('TMCEncTime', bone_enc_time) # 🕒 ✔️
        ticker.set_time('TMCDecTime', bone_dec_time) # 🕒 ✔️
        if args.verbose:
            print('[TMC] Enc:', ticker.get_time('TMCEncTime'), 's')
            print('[TMC] Dec:', ticker.get_time('TMCDecTime'), 's')

        ############## 🚩 Adaptive Aligning ##############
        
        ticker.start_count('Aligning') # 🕒 ⏳

        # we have to re-sort the bones&windows since decoded bones has different point order
        # otherwise the bones and window cannot one-by-one match
        cloest_idx = op.reorder(bones, rec_bones)
        bones, local_windows = bones[cloest_idx], local_windows[cloest_idx]
        # we get bones: (M, 3); local windows: (M, K, 3)
        aligned_windows = op.AdaptiveAligning(local_windows, bones)
        # aligned_windows: (M, K, 3)

        ticker.end_count('Aligning') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Aligning:', ticker.get_time('Aligning'), 's')

        ############## 🚩 Attention-based Aggregation ##############

        ticker.start_count('Aggregation') # 🕒 ⏳

        # due to the GPU memory limit
        # the attention block cannot deal with all the windows in one shot
        # we group the windows into mini batches
        mini_batch_size = 524288 // args.local_window_size
        cursor = 0
        skin_fea_ls = []
        while cursor < aligned_windows.shape[0]:
            skin_fea_item = model.awds.mini_emb(aligned_windows[cursor:cursor+mini_batch_size])
            skin_fea_item = model.awds.pt(aligned_windows[cursor:cursor+mini_batch_size], skin_fea_item)
            skin_fea_ls.append(skin_fea_item)
            cursor = cursor + mini_batch_size
        skin_fea = torch.cat(skin_fea_ls, dim=0)
        # get skin_fea: (M, C)

        ticker.end_count('Aggregation') # 🕒 ✔️
        if args.verbose:
            print('[AWDS] Aggregation:', ticker.get_time('Aggregation'), 's')


        ######################################################
        ################## Entropy Modeling ##################
        ######################################################
            
        ############## 🚩 DW-Build ##############
            
        ticker.start_count('DWBuild') # 🕒 ⏳

        dilated_idx, dilated_windows = model.dw_build(rec_bones)

        ticker.end_count('DWBuild') # 🕒 ✔️
        if args.verbose:
            print('[DWBuild]:', ticker.get_time('DWBuild'), 's')

        ############## 🚩 DWEM ##############
            
        ticker.start_count('DWEM') # 🕒 ⏳
        
        mu, sigma = model.dwem(dilated_windows)
        
        ticker.end_count('DWEM') # 🕒 ✔️
        if args.verbose:
            print('[DWEM]:', ticker.get_time('DWEM'), 's')

        ############## 🚩 Feature Squeezeing & Quantization ##############
            
        ticker.start_count('Squeezeing_Q') # 🕒 ⏳
            
        compact_fea = model.fea_squeeze(skin_fea)
        quantized_compact_fea = torch.round(compact_fea)
        # get quantized_compact_fea: (M, c)

        ticker.end_count('Squeezeing_Q') # 🕒 ✔️
        if args.verbose:
            print('[Squeezeing]+[Q]:', ticker.get_time('Squeezeing_Q'), 's')

        ############## 🚩 Arithmetic Encoding ##############
            
        ticker.start_count('AE') # 🕒 ⏳
        min_v_value, max_v_value = quantized_compact_fea.min().to(torch.int16), quantized_compact_fea.max().to(torch.int16)
        bytestream = torchac.encode_int16_normalized_cdf(
            op._convert_to_int_and_normalize(op.get_cdf_min_max_v(mu-min_v_value, sigma, L=max_v_value-min_v_value+1), needs_normalization=True).cpu(), 
            (quantized_compact_fea-min_v_value).cpu().to(torch.int16)
        )

        ticker.end_count('AE') # 🕒 ✔️
        if args.verbose:
            print('[AE]:', ticker.get_time('AE'), 's')

        ############## Saving ##############
        with open(compressed_head_path, 'wb') as fout:
            fout.write(np.array(args.local_window_size, dtype=np.uint16).tobytes())
            fout.write(np.array(min_v_value.item(), dtype=np.int16).tobytes())
            fout.write(np.array(max_v_value.item(), dtype=np.int16).tobytes())

        with open(compressed_skin_path, 'wb') as fin:
            fin.write(bytestream)

        ############## Calc Enc Time and Bitrate ##############

        # Calc Bpp
        total_bits = op.get_file_size_in_bits(compressed_bones_path) + op.get_file_size_in_bits(compressed_skin_path) + op.get_file_size_in_bits(compressed_head_path)
        bpp = total_bits/N

        # Calc Enc Time
        enc_time = ticker.dump_sum()
        
        # Add to recoder
        time_recoder.update(enc_time)
        bpp_recoder.update(bpp)

        if args.verbose:
            print(f'{filename_w_ext} done. Encoding time: {enc_time}s | Bpp: {round(bpp, 3)}.')

        # remove cache file
        # but it is ok not to clean it up, it won't affect the code running...
        output = subprocess.check_output(f'rm {cache_file_path}', shell=True, stderr=subprocess.STDOUT)
    
print(f'Done. Avg. Encoding time: {time_recoder.dump_avg(precision=3)} | Bpp: {bpp_recoder.dump_avg(precision=3)}')
