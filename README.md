# Pointsoup: High-Performance and Extremely Low-Decoding-Latency Learned Geometry Codec for Large-Scale Point Cloud Scenes

## News
- [2024.4.21] The manuscript is uploaded to Arxiv, which will be publicly accessible soon.
- [2024.4.21] The supplementary material is uploaded to [Google Drive](https://drive.google.com/file/d/113PvrVBll9frY1k6OC3QDA5PugnDcHdV/view?usp=sharing).
- [2024.4.17] Our paper has been accepted by [IJCAI 2024](https://ijcai24.org/)!

## Overview
> Despite considerable progress being achieved in point cloud geometry compression, there still remains a challenge in effectively compressing large-scale scenes with sparse surfaces. Another key challenge lies in reducing decoding latency, a crucial requirement in real-world application. In this paper, we propose Pointsoup, an efficient learning-based geometry codec that attains high-performance and extremely low-decoding-latency simultaneously. Inspired by conventional Trisoup codec, a point model-based strategy is devised to characterize local surfaces. Specifically, skin features are embedded from local windows via an attention-based encoder, and dilated windows are introduced as cross-scale priors to infer the distribution of quantized features in parallel. During decoding, features undergo fast refinement, followed by a folding-based point generator that reconstructs point coordinates with fairly fast speed. Experiments show that Pointsoup achieves state-of-the-art performance on multiple benchmarks with significantly lower decoding complexity, i.e., up to 90$\sim$160$\times$ faster than the G-PCCv23 Trisoup decoder on a comparatively low-end platform (e.g., one RTX 2080Ti). Furthermore, it offers variable-rate control with a single neural model (2.9MB), which is attractive for industrial practitioners.

## Environment

The environment we use is as follows：

- Python 3.10.14
- Pytorch 2.0.1 with CUDA 11.7
- Pytorch3d 0.7.5
- Torchac 0.9.3

For the convenience of reproduction, we provide three different ways to help create the environment:

#### Option 1: Using yml

```
conda env create -f=environment.yml
```

#### Option 2: Using .sh

```
source ./env_create.sh
```

#### Option 3: CodeWithGPU (AutoDL image)

Will be released soon.

## Data

In our paper, point clouds with the coordinate range of [0, 1023] are used as input.

Example point clouds are saved in ``./data/example_pc_1023/``, trained model is saved in ``./model/exp/``.

## Compression
First and foremost, the `tmc3` is need to perform predtree coding on bone points. If the `tmc3` file we provided cannot work on your platform, please refer to [MPEGGroup/mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for manual building.

```
chmod +x ./tmc3
```

You can adjust the compression ratio by simply adjusting the parameter `local_window_size`. In our paper, we use `local_window_size` in the range of 2048~128.

```
python ./compress.py \
    --input_glob='./data/example_pc_1023/*.ply' \
    --compressed_path='./data/compressed/' \
    --model_load_path='./model/exp/ckpt.pt'\
    --local_window_size=200 \
    --tmc_path='./tmc3'\
    --verbose=True
```

## Decompression

```
python ./decompress.py \
    --compressed_path='./data/compressed/' \
    --decompressed_path='./data/decompressed/' \
    --model_load_path='./model/exp/ckpt.pt'\
    --tmc_path='./tmc3'\
    --verbose=True
```

## Evaluation

We use `PccAppMetrics` for D1 PSNR calculation. You can refer to [MPEGGroup/mpeg-pcc-tmc2](https://github.com/MPEGGroup/mpeg-pcc-tmc2) if the provided `PccAppMetrics` file does not fit your platform.

```
python ./eval_PSNR.py \
    --input_glob='./data/example_pc_1023/*.ply' \
    --decompressed_path='./data/decompressed/' \
    --pcc_metric_path='./PccAppMetrics' \
    --resolution=1023
```

## Disucssion
Merits:

- High Performance - SOTA efficiency on multiple large-scale benchmarks.
- Low Decoding Latency - 90~160× faster than the conventional Trisoup decoder.
- Robust Generalizability - Applicable to large-scale samples once trained on small objects.
- High Flexibility - Variable-rate control with a single neural model.
- Light Weight - Fairly small with 761k parameters (about 2.9MB).

Limitations:

- Rate-distortion performance is inferior to G-PCC Octree codec at high bitrates (e.g., bpp>1). The surface approximation-based approaches (Pointsoup and Trisoup) seem hard to characterize accurate point positions even if given enough bitrate budget.

- Naive outdoor LiDAR frame coding efficacy is unsatisfactory. Due to the used sampling&grouping strategy, the pointsoup is limited to point clouds with relatively uniform distributed points, such as [S3DIS](http://buildingparser.stanford.edu/dataset.html), [ScanNet](https://github.com/ScanNet/ScanNet), [dense point cloud map](https://github.com/PRBonn/deep-point-map-compression), [8iVFB (human body)](https://plenodb.jpeg.org/pc/8ilabs), [Visionair (objects)](https://github.com/yulequan/PU-Net), etc.


## Citation

```
Waiting for publication...
```
