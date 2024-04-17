import os
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='eval_PSNR.py',
    description='Eval Geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/example_pcs/conferenceRoom_1.ply')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/decompressed/')
parser.add_argument('--pcc_metric_path', type=str, help='Path for PccAppMetrics.', default='./PccAppMetrics')
parser.add_argument('--resolution', type=int, help='Point cloud resolution (peak signal).', default=1023)

args = parser.parse_args()

files = np.array(glob(args.input_glob))

def process(input_f):
    filename_w_ext = os.path.split(input_f)[-1]
    dec_f = os.path.join(args.decompressed_path, filename_w_ext+'.bin.ply')

    cmd = f'{args.pcc_metric_path} \
    --uncompressedDataPath={input_f} --reconstructedDataPath={dec_f} \
    --resolution={args.resolution}   --frameCount=1'
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    d1_psnr = float(str(output).split('mseF,PSNR (p2point):')[1].split('\\n')[0])
    
    return np.array([filename_w_ext, d1_psnr])

f_len = len(files)
with Pool(4) as p:
    arr = list(tqdm(p.imap(process, files), total=f_len))

arr = np.array(arr)
fnames, p2pPSNRs = arr[:, 0], arr[:, 1].astype(float)
    
print('Avg. D1 PSNR:', round(p2pPSNRs.mean(), 3))
