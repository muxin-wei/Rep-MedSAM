#%%
import numpy as np
import torch
import torch.nn as nn
from repvit import RepViT, RepViTBlock
from segment_anything.modeling import  PromptEncoder, MaskDecoder, TwoWayTransformer, ImageEncoderViT
from utils import replace_batchnorm
from collections import OrderedDict
from repvit_cfgs import repvit_m1_0_cfgs
from torchsummary import  summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count
from functools import partial
from time import time
from tiny_vit_sam import TinyViT
#%%
device = torch.device('cpu')

#%%
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
        
        self.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe().to(device), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings.to(device), # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings.to(device), # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        # return low_res_masks, iou_predictions
    
#%%
medsam_lite_image_encoder = RepViT(repvit_m1_0_cfgs)

medsam_image_encoder = ImageEncoderViT(
    depth=12,
    embed_dim=768,
    img_size=256,
    mlp_ratio=4,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    num_heads=12,
    patch_size=16,
    qkv_bias=True,
    use_rel_pos=True,
    global_attn_indexes=[2, 5, 8, 11],
    window_size=14,
    out_chans=256,
).to(device)

tiny_medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
).to(device)



medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
).to(device)

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
).to(device)

rep_medsam =  MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
).to(device)

medsam = MedSAM_Lite(
    image_encoder = medsam_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

tinysam = MedSAM_Lite(
    image_encoder = tiny_medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

#%% 
image = torch.rand(1, 3, 256, 256).to(device)
boxes = torch.randint(low=0, high=256, size=(1, 1, 4)).to(device)
#%%
medsam_lite_image_encoder.eval()
replace_batchnorm(medsam_lite_image_encoder)

start_vit = time()
vit_out = medsam_image_encoder(image)
end_vit = time()
cost_vit = end_vit - start_vit
print(f'vit time consume: {cost_vit}')

start_tinyvit = time()
tinyvit_out = tiny_medsam_lite_image_encoder(image)
end_tinyvit = time()
cost_tinyvit = end_tinyvit - start_tinyvit
print(f'tinyvit time consume: {cost_tinyvit}')

start_repvit = time()
repvit_out = medsam_lite_image_encoder(image)
end_repvit = time()
cost_tinyvit = end_repvit - start_repvit
print(f'rep-vit time consume: {cost_tinyvit}')

tiny_flops = FlopCountAnalysis(model=tiny_medsam_lite_image_encoder, inputs= (image))
print(tiny_flops.total())

rep_flops = FlopCountAnalysis(model=medsam_lite_image_encoder, inputs= (image))
print(rep_flops.total())


# %%
