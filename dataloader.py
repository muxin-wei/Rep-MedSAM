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

import cv2


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
            np.max(img_256)<=1.0 and np.min(img_256)>=0.0
            ), 'image should be normalized to [0, 1]'
        
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) 
        # multiple labels [0, 1,4,5...], (256,256)

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



#%% 
data_root=['/mnt/train_npy', '/cvpr-data/train_npy', '/train_ct/train_npy']
tr_dataset = NpyDataset(data_root, data_aug=True)
tr_dataloader = DataLoader(tr_dataset, batch_size=32, shuffle=True, num_workers = 12)

#%%
for step, batch in enumerate(tqdm(tr_dataloader)):
    # show the example

    image = batch["image"]
    gt = batch["gt2D"]
    bboxes = batch["bboxes"]
    names_temp = batch["image_name"]


# %%
