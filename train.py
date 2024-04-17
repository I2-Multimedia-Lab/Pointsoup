import os
import random
import argparse

import numpy as np
from glob import glob
from datetime import datetime

import torch
import torch.utils.data as Data
from pytorch3d.loss import chamfer_distance

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
    prog='train.py',
    description='Training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--model_save_folder', help='Directory where to save trained models.', default=f'./model/exp/')
parser.add_argument('--train_glob', help='Glob pattern to load point clouds.', default='/mnt/hdd/datasets_yk/ShapeNet/ShapeNet_pc_01_8192p_colorful/train/*.ply')

parser.add_argument('--local_window_size', type=int, help='Local window size $K$.', default=128)
parser.add_argument('--dilated_window_size', type=int, help='Dilated window size $k$.', default=8)
parser.add_argument('--channel', type=int, help='Network channel.', default=128)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel.', default=16)
parser.add_argument('--λ_R', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-04)

parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=5000)
parser.add_argument('--batch_size', type=int, help='Batch size (must be 1).', default=1)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate at x step.', default=[70000, 120000])
parser.add_argument('--max_step', type=int, help='Train up to this number of steps.', default=140000)

args = parser.parse_args()

# Create Model Save Path
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

files = np.array(glob(args.train_glob, recursive=True))[:10000]
pcs = io.read_point_clouds(files)

loader = Data.DataLoader(
    dataset = pcs,
    batch_size = args.batch_size,
    shuffle = True,
)

model = network.Pointsoup(k=args.dilated_window_size,
                          channel=args.channel, 
                          bottleneck_channel=args.bottleneck_channel)
model = model.cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

global_step = 0
cd_recoder, bpp_recoder, loss_recoder = op.Recoder(), op.Recoder(), op.Recoder()

for epoch in range(1, 9999):
    print(datetime.now())
    for batch_x in loader:
        batch_x = batch_x.cuda()

        rec_batch_x, bitrate = model(batch_x, args.local_window_size)

        # Get Loss
        chamfer_dist, _ = chamfer_distance(rec_batch_x, batch_x)
        loss = chamfer_dist
        if global_step > args.rate_loss_enable_step:
            loss += args.λ_R * bitrate
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        cd_recoder.update(chamfer_dist.item())
        bpp_recoder.update(bitrate.item())
        loss_recoder.update(loss.item())

        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | \
                  Step:{global_step} | \
                  Ave CD: {cd_recoder.dump_avg()} | \
                  Bpp: {bpp_recoder.dump_avg()} | \
                  Loss: {loss_recoder.dump_avg()}')

            # save model
            torch.save(model.state_dict(), os.path.join(args.model_save_folder, 'ckpt.pt'))
        
        # Learning Rate Decay
        if global_step in args.lr_decay_steps:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')
        
        if global_step > args.max_step:
            break
    
    if global_step > args.max_step:
        break
