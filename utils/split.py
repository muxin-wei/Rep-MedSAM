# -*- coding: utf-8 -*-
import os
join = os.path.join
import random
from tqdm import tqdm

path_npy = '/mnt/FIVES/Fundus_FIVES' # npy dataset path
train_path = '/mnt/finetune/train' 
validation_path = '/mnt/finetune/val'
# testing_path = '/mnt/finetune/test'

##% split npy files
if path_npy is not None:
    img_path = join(path_npy, 'imgs')
    gt_path = join(path_npy, 'gts')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '.npy'
    gt_suffix = '.npy'
    
    os.makedirs(join(train_path, 'imgs'), exist_ok=True)
    os.makedirs(join(train_path, 'gts'), exist_ok=True)
    os.makedirs(join(validation_path, 'imgs'), exist_ok=True)
    os.makedirs(join(validation_path, 'gts'), exist_ok=True)
    
    # split 20% data for validation
    validation_names = random.sample(gt_names, int(len(gt_names)*0.2))
    train_names = [name for name in gt_names if name not in validation_names]
    # move validation and train data to corresponding folders
    for name in tqdm(validation_names):
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(validation_path, 'imgs', img_name))
        os.rename(join(gt_path, name), join(validation_path, 'gts', name))
        
    # Move the training data to the corresponding folders
    for name in tqdm(train_names):
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(train_path, 'imgs', img_name))
        os.rename(join(gt_path, name), join(train_path, 'gts', name))
