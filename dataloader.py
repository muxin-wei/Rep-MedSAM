# %%
import os
import random
from os.path import join, exists, isfile, isdir, basename, dirname
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import torchvision.transforms.v2.functional as aug
from torchvision.transforms import v2
import cv2
# from torchvision import models, datasets, tv_tensors
import PIL.Image


#%%
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

#%%

transforms = v2.ColorJitter(brightness=.4, contrast=.5, saturation=.4, hue= .2)

#%%
class NpyDataset(Dataset): 
    def __init__(self, data_root,image_size=256, bbox_shift=5, data_aug=True):
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
        self.bbox_shift = bbox_shift
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
        if len(gt.shape) > 2:
            gt_slice_idx = random.randint(0, gt.shape[0] - 1)
            gt = gt[gt_slice_idx,:,:]
        assert gt.shape == (256, 256), f'gt shape (256, 256) expected, but got {gt.shape} instead'
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
        gt2D = np.uint8(gt2D > 0)
        img_256 = torch.tensor(img_256).float()
        gt2D = torch.tensor(gt2D[np.newaxis,:]).long()
        if self.data_aug:
            img_256 = transforms(img_256)
            if random.random() > .3:
                img_256 = aug.horizontal_flip_image_tensor(img_256)
                gt2D = aug.horizontal_flip_mask(gt2D)
            if random.random() > .3:
                img_256 = aug.vertical_flip_image_tensor(img_256)
                gt2D = aug.vertical_flip_mask(gt2D)
        gt2D = gt2D.squeeze(0)
        assert len(gt2D.shape) == 2, f"gt for {img_name} is not 2D"
        assert (
            torch.max(gt2D)<=1.0 and torch.min(gt2D)>=0.0
            ), f'mask {img_name} should be normalized to [0, 1]'
        # assert max(gt2)
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
        
        # bboxes

        # bboxes = torch.tensor(bboxes[None,None, ...]).float()

        return {
            "ori_image":img_3c,
            "image": img_256,
            "gt2D": gt2D[None, :,:],
            "bboxes": torch.tensor(bboxes[None,None, ...]).float(), # (B, 1, 4)
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


#%% 
batch_size = 48
data_root=['/train_ct/CT-Organs']
tr_dataset = NpyDataset(data_root, data_aug=True)
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers = 12)
#%%
for step, batch in enumerate(tqdm(tr_dataloader)):
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(10, 10))
    idx = random.randint(0, 6)

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
    idx = random.randint(6, 11)
    axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
    show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
    show_box(bboxes[idx].numpy().squeeze(), axs[1])
    axs[1].axis('off')
    # set title
    axs[1].set_title(names_temp[idx])
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(
        join('workdir/sanity_check', 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    if step == 2:
        continue
        # break

# %%
