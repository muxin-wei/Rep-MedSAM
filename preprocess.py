#%%
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from skimage import io
join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d
from os.path import join, isfile, dirname, basename
from glob import glob
from os import makedirs
import pydicom as dic
import matplotlib.colors as mcolor
import random

# %%
def get_box(gt2D, label,bbox_shift=3):
    y_indices, x_indices = np.where(gt2D == label)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
        # get random color for bbox edge and opacity
    else:
        color = np.array([251/255, 252/255, 30/255, 0.35]) #(r,g,b, alpha(opacity))
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image) 
    
def show_box(box, ax):
    x0, y0 = box[0], box[1] # bbox left-top 
    w, h = box[2] - box[0], box[3] - box[1] # get bbox width, height 
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
    # showing bbox with blue and transparent edge  on image, width = 2px
    
#%%
data_path = 'paper_example/path'
gt_path = join(data_path, 'gts')
img_path = join(data_path, 'imgs')
makedirs(gt_path, exist_ok=True)
makedirs(img_path, exist_ok=True)
gts = sorted(glob(join(gt_path, '*.nii.gz'), recursive=True))

WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 1500  # only for CT images

#%%
for gt in gts:
    # gt_name = basename(gt).split('_labels.nii.gz')
    img = join(img_path, basename(gt).split('.nii')[0])+'.png'
    gt_sitk = sitk.ReadImage(gt)
    img_sitk = sitk.ReadImage(img)
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    img_data = sitk.GetArrayFromImage(img_sitk)
    spacing = np.array(gt_sitk.GetSpacing())
    
    lower_bound, upper_bound = np.percentile(image_data[image_data>0], 0.5), np.percentile(image_data[image_data>0], 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre[image_data==0] = 0
    image_data_pre = np.uint8(image_data_pre)
    
    print(np.unique(gt_data))
    for i in range(gt_data.shape[0]):
        boxes = []
        gt_i = gt_data[i, :, :]
        img_i = image_data_pre[i,:,:]
        
        unique_labels = np.unique(gt_i)
        mask_labels = unique_labels[unique_labels>0]
        
        if len(mask_labels) == 0 :
            continue
        
        # _, axs = plt.subplots(1, 3, figsize = (10, 5))
        # axs[0].imshow(img_i, cmap = 'gray')
        for idx, label in enumerate(mask_labels, start=1):
            bbox = get_box(gt_i, label)
            mask = np.zeros_like(gt_i)
            mask[gt_i == label] = label 
            # show_box(bbox, axs[0])
            # show_mask(mask, axs[0])
            boxes.append(bbox)

        boxes = np.array(boxes, dtype=np.int64)
    #     # print(boxes.shape)
    #     # axs[1].imshow(img_i, cmap = 'gray')
    #     # axs[2].imshow(gt_i)
    #     # plt.subplots_adjust(wspace=0.01, hspace=0)
    #     # plt.show()
    #     # _, ax = plt.subplots(figsize=(10,10))
    #     # ax.imshow(img_3c, cmap = 'gray')
        img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
        np.savez(join(img_path, f'2Dbox_MS  _Case029_slice#{i}.npz'), imgs = img_3c, boxes = boxes)
        np.savez(join(gt_path,f'2Dbox_Ms_Case029_slice#{i}.npz'), gts = gt_i)

# %%

def slice_number(filename):
    return int(basename(filename).split('.npz')[0].split('#')[-1])

seg_root = join(data_path, 'segs')
segs_data = []
segs = sorted(glob(join(seg_root, '*.npz'),recursive=True), key=slice_number)

slice_0 = slice_number(segs[0])
slice_end = slice_number(segs[-1])
empty = np.zeros_like(gt_data[i, :, :])

for idx in range(1, slice_0):
    segs_data.append(empty)
print(len(segs_data))

for seg in segs:
    print(basename(seg))
    seg_data = np.load(seg, allow_pickle=True, mmap_mode='r+')
    segs_data.append(seg_data['segs'])


seg_data = np.array(segs_data)
print(seg_data.shape)  
seg_sitk = sitk.GetImageFromArray(segs_data)
seg_sitk.SetSpacing(spacing=gt_sitk.GetSpacing())
seg_sitk.SetOrigin(origin=gt_sitk.GetOrigin())
seg_sitk.SetDirection(direction=gt_sitk.GetDirection())


# %%
sitk.WriteImage(seg_sitk,join(data_path, 'US_seg.nii.gz'))
# %%

img = np.load('test_demo\imgs\\2DBox_OCT_demo.npz', 'r', allow_pickle=True)
img_data = img['imgs']
boxes = img['boxes']
print(boxes)

img_sitk = sitk.GetImageFromArray(img_data.transpose())
boxes_img = np.zeros_like(gt_data)
for idx, box in enumerate(boxes, start=1):
    x_min, y_min, x_max, y_max = box
    boxes_img[x_min:x_max+1, y_max] = idx
    boxes_img[x_max, y_min:y_max+1] = idx
    boxes_img[x_min:x_max-1, y_min] = idx
    boxes_img[x_min, y_min:y_max-1] = idx


# box_sitk = sitk.GetImageFromArray()
# sitk.WriteImage(img_sitk, 'paper_example\\2DBox_OCT_demo_img.nii.gz')
# sitk.WriteImage(gt_sitk, 'paper_example\\2DBox_OCT_demo_gt.nii.gz')
# %%
boxes_itk = sitk.GetImageFromArray(boxes_img.transpose())
sitk.WriteImage(boxes_itk, 'paper_example\\2DBox_OCT_demo_bx.nii.gz')

# %%
gt = np.load('test_demo\\rep_med_seg\\2DBox_OCT_demo.npz', 'r', allow_pickle=True)
gt_data = gt['segs']
gt_sitk = sitk.GetImageFromArray(gt_data)
sitk.WriteImage(gt_sitk, 'paper_example\\2DBox_OCT_demo_seg.nii.gz')

# %%
