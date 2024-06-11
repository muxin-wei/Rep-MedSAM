# Solution for CVPR 2024 MedSAM on Laptop Challenge

**Rep-MedSAM: Towards Real-time and Universal Medical Image Segmentation** \
*Muxin Wei, Shuqing Chen, Silin Wu, Dabin Xu*

Built upon [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) and [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT/tree/main/model). This repository provides the solution of our team for [CVPR 2024 MedSAM on Laptop Challenge](https://www.codabench.org/competitions/1847/#/pages-tab). Details of our method are described in our [paper](https://openreview.net/forum?id=yqf77n9Kfw&noteId=yqf77n9Kfw).

Model weight is available at [ckpts](./ckpts).
## Requirements and Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.9` | `CUDA 12.2` | `Pytorch 2.0`

1. git clone https://github.com/mxWe1/Rep-MedSAM.git
2. Enter Rep-MedSAM folder `cd Rep-MedSAM` and run `pip install -e .`


## Training

To train Rep-MedSAM, run:
```bash
python train_one_gpu.py \
-data_root /path/to/train_npy \
-pretrained_checkpoint ckpts/rep_medsam.pth \
-work_dir work_dir \
-num_epochs 10 \
-batch_size 16 \
-num_workers 12 
```
### Pretraining Distillation

To perform pretraining distillation, the first thing to do is get image embeddings from images using MedSAM Image Encoder.
You can use infer script from MedSAM or our distil script to get image embeddings.
Run the following command to distil from MedSAM.

```bash
python train_one_gpu.py \
-data_root /path/to/train_npy \
-embedding_path /path/to/embeddings\ 
-mask_decoder teacher_model/mask_decoder.pth \ 
-prompt_encoder teacher_model/prompt_encoder.pth \
-work_dir work_dir/distillation \
-num_epochs 5 \
-lr 5E-4 \ 
-distillation True \
-num_workers 12 
```

#### Data Structure for Training
    images, masks and corresponding embeddings share the same file name
    train_npy
    ├─embeddings
    ├─gts
    └─imgs
