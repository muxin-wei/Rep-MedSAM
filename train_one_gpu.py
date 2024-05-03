# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename, dirname
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from repvit import RepViT
from repvit_cfgs import repvit_m1_5_cfgs, repvit_m0_9_cfgs, repvit_m1_0_cfgs
from functools import partial
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer, ImageEncoderViT
import cv2
import torch.nn.functional as F
from find_lr import LRFinder
from sklearn.model_selection import KFold
from math import nan
from matplotlib import pyplot as plt
import argparse
import torch.multiprocessing as mp
from CVPR24_LiteMedSAM_infer import validate


# %%

parser = argparse.ArgumentParser()
# npy data root
parser.add_argument(
    "-data_root", type=str, default="/cvpr-data/train_npy/",
    help="Path to the npy data root."
)
parser.add_argument(
    "-data_root_1", type=str, default='/train_ct/train_npy/'
)

parser.add_argument(
    '-data_root_2',type=str, default='/mnt/others/'
)
# pre-trained checkpoint
parser.add_argument(
    "-pretrained_checkpoint", type=str, default="workdir/finetune/finetune_weights.pth",
    help="Path to the pretrained RepViT-SAM checkpoint."
)

parser.add_argument(
    "-resume", type=str, default='',
    help="Path to the checkpoint to continue training."
)

parser.add_argument(
    "-work_dir", type=str, default="./workdir/finetune/",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=20,
    help="Number of epochs to train."
)

parser.add_argument(
    "-batch_size", type=int, default=16,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=12,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=5E-4,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=0.85,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.15,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)
parser.add_argument(
    "-distillation", type=bool, default=False, 
    help="Enable distillation"
)
parser.add_argument(
    "-teacher", type=str, default=None
)
parser.add_argument(
    "-mse_loss_weight", type=float, default=1.0,
    help="Weight of mean squared error for distillation"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument(
    "-embedding_path", type=str, default=None
)
parser.add_argument(
    '-find_lr', type=bool, default=False,
    help='find lr for training'
)
parser.add_argument(
    "-val_root", type=str, default='imgs/'
)

args = parser.parse_args()
# %%

device = torch.device(args.device)
do_sancheck = args.sanity_check

makedirs(args.work_dir, exist_ok=True)
# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
        # get random color for bbox edge and opacity
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45]) #(r,g,b, alpha(opacity))
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image) 
    
def show_box(box, ax):
    x0, y0 = box[0], box[1] # bbox left-top 
    w, h = box[2] - box[0], box[3] - box[1] # get bbox width, height 
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
    # showing bbox with blue and transparent edge  on image, width = 2px
    
def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)]) # gt&inf
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)]) # gt|inf
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)

# %%
def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 


# %%
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path_files = []
        for root in self.data_root:
            gt_path = join(root, 'gts')
            img_path = join(root, 'imgs')
            gt_files = sorted(glob(join(gt_path, '*.npy'), recursive=True))
            gt_files = [
                file for file in gt_files
                if isfile(join(img_path, basename(file)))
            ]
            self.gt_path_files.extend(gt_files)
        
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = args.bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        image_path = dirname(dirname(self.gt_path_files[index]))
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(join(image_path, 'imgs/'+ img_name), 'r', allow_pickle=True) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_256 = np.transpose(img_3c, (2, 0, 1)) # (3, 256, 256)
        assert (
            img_256.shape[1] == 256 and img_256.shape[2] == 256
        ), f'image {img_name} shape should be 256'
        assert (
            np.max(img_256)<=1.0 and np.min(img_256)>=0.0
            ), 'image should be normalized to [0, 1]'
        
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
                
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return {
            "ori_image":img_3c,
            "image": torch.tensor(img_256).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_256.shape[0], img_256.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long(),
        }

    def resize_longest_side(self, image, target_length = 256):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image, target_length=256):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = target_length - h
        padw = target_length - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded


#%% sanity test of dataset class

if do_sancheck:
    tr_dataset = NpyDataset(args.data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(args.work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break

# %% MedSAM_Lite Module Design
class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder  
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = self.prompt_encoder( 
            points=None,
            boxes=boxes,
            masks=None,
        ) # get sparse_embeddings (one-point based and bbox) and z()
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        if args.distillation:
            return image_embedding
        return low_res_masks, iou_predictions


    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks =  F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

# %%
def main():
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project="RepSAM",
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.data_root,
                "model_type": "repvit",
                "epochs": args.num_epochs,
            },
        )

    medsam_lite_image_encoder = RepViT(
        cfgs=repvit_m1_0_cfgs,
        img_size=256
        )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )


    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    if args.pretrained_checkpoint is not None:
        if isfile(args.pretrained_checkpoint):
            print(f"Finetuning with pretrained weights {args.pretrained_checkpoint}")
            medsam_lite_ckpt = torch.load(
                args.pretrained_checkpoint,
                map_location="cpu"
            )
            medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
        else:
            print(f"Pretained weights {args.pretrained_checkpoint} not found, training from scratch")

    if args.distillation:
        medsam_lite_model.mask_decoder.requires_grad_(False)
        medsam_lite_prompt_encoder.requires_grad_(False)
        med_enc = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        ).to(device=device)
        med_enc.load_state_dict(torch.load(args.teacher))
        med_enc.eval()
        print(f"MedSAM Image Encoder size:{sum(p.numel() for p in med_enc.parameters())}")

    medsam_lite_model = medsam_lite_model.to(device)
    medsam_lite_model.train()
        
    # %%
    print(f"Rep-MedSAM  size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # %%
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        cooldown=0
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    if args.distillation:
        mse_loss = nn.MSELoss(reduction='mean')
        
    # %%
    data_root = []
    if args.data_root is not None:
        data_root.append(args.data_root)
    if args.data_root_1 is not None:
        data_root.append(args.data_root_1)
    if args.data_root_2 is not None:
        data_root.append(args.data_root_2)
    train_dataset = NpyDataset(data_root=data_root, data_aug=True) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.resume and isfile(args.resume):
        print(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10
    
    if args.find_lr:
        lr_finder = LRFinder(model=medsam_lite_model, optimizer=optimizer,device=args.device)
        lr_finder.range_test(train_loader, end_lr=1, num_iter=int(len(train_loader)/2), step_mode="linear")
        lr_finder.plot(suggest_lr=True, log_lr=False)
        lr_finder.reset() 
        return
        
    # %%
    train_losses = []
    iou_losses = []
    mask_losses = []
    grad_history = []
    for epoch in range(start_epoch + 1, args.num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        iou_epoch_loss = [1e10 for _ in range(len(train_loader))]
        mask_epoch_loss = [1e10 for _ in range(len(train_dataset))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            image_names = batch["image_name"]
            
            if args.distillation:
                embedding_list = []
                for img_name in image_names:
                    embedding = torch.from_numpy(np.load(join(args.embedding_path,img_name), allow_pickle=True, mmap_mode='r'))
                    embedding_list.append(embedding.squeeze(0).to(device))
                embeddings = torch.stack(embedding_list)
                
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            #mask_loss = l_seg + l_ce
            mask_loss = args.seg_loss_weight * l_seg + args.ce_loss_weight * l_ce
            
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            #loss = mask_loss + l_iou
            loss = mask_loss + args.iou_loss_weight * l_iou
            
            if args.distillation:
                l_mse = mse_loss(embeddings_pred, embeddings)
                loss = l_mse * args.mse_loss_weight
            epoch_loss[step] = loss.item()
            iou_epoch_loss[step] = l_iou.item()
            mask_epoch_loss[step] = mask_loss.item()
            loss.backward()

            obs_grad_norm = _get_grad_norm(medsam_lite_model)
            grad_history.append(obs_grad_norm)
            clip_value = np.percentile(grad_history, 90)
            torch.nn.utils.clip_grad_norm_(medsam_lite_model.parameters(), clip_value)        
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
        
        epoch_end_time = time()
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        iou_epoch_loss_reduced = sum(iou_epoch_loss) / len(epoch_loss)
        mask_epoch_loss_reduced = sum(mask_epoch_loss) / len(epoch_loss)
        
        
        
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss_reduced})
            wandb.log({"mask loss": mask_epoch_loss_reduced})
            wandb.log({"iou loss": iou_epoch_loss_reduced})
        train_losses.append(epoch_loss_reduced)
        iou_losses.append(iou_epoch_loss_reduced)
        mask_losses.append(mask_epoch_loss_reduced)
        lr_scheduler.step(epoch_loss_reduced)
        model_weights = medsam_lite_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss_reduced,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, join(args.work_dir, f"medsam_lite_{epoch}.pth"))
        if epoch_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, join(args.work_dir, "medsam_lite_best.pth"))

        epoch_loss_reduced = 1e10
        mask_epoch_loss_reduced = 1e10
        iou_epoch_loss_reduced = 1e10
        # %% plot loss
        plt.plot(train_losses)
        if args.distillation:
            plt.title("MSE Loss")
        else:
            plt.title("Dice + Binary Cross Entropy + IoU Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(args.work_dir, "train_loss.png"))
        plt.close()
        
        plt.plot(iou_losses)
        plt.title("IoU Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(args.work_dir, "iou_loss.png"))
        
        plt.close()
        plt.plot(mask_losses)
        plt.title("Mask Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(args.work_dir, "mask_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()