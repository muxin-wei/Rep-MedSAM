#%%
from segment_anything.modeling import ImageEncoderViT
from glob import glob
from os.path import join, isfile, basename
from os import makedirs
from functools import partial
import torch
import numpy as np

#%%
device = torch.device('cpu') # device
image_encoder_ckpt = 'teacher_model/MedSAM_Enc.pth' # path to medsam image encoder weights.
img_path = 'distil_demo/' #path to npy files (1024, 1024)
embedding_path = 'distil_demo/embeddings/' #path to save image embeddings
imgs = sorted(glob(join(img_path, '*.npy'), recursive=True))
makedirs(embedding_path, exist_ok=True)
#%%
medsam_image_encoder = ImageEncoderViT(
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
medsam_image_encoder.load_state_dict(torch.load(image_encoder_ckpt,map_location=device))
medsam_image_encoder.eval()

#%%
for img in imgs:
    img_name = basename(img)
    img_3c = np.load(img, 'r', allow_pickle=True) # (1024, 1024, 3)
    img_1024 = np.transpose(img_3c, (2, 0, 1)) # (3, 1024, 1024)
    assert(
        img_1024.shape[1] == 1024 and img_1024.shape[2] == 1024
    ), f'image {img_name} shape should be 256'
    assert(
        np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0
    ), f'image {img_name} should be normalized to [0, 1]'
    tensor_1024 = torch.tensor(img_1024).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = medsam_image_encoder(tensor_1024).squeeze(0)
    np.save(join(embedding_path, img_name), embedding.numpy())

# %%