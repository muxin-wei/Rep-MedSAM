# %%
import torch
import argparse
import os
join = os.path.join


# %%
from_pth = '/data/Rep-MedSAM/workdir/finetune/checkpoint.pth'
to_pth = '/data/Rep-MedSAM/workdir/finetune/weights.pth'

# %%
from_pth = torch.load(from_pth, map_location='cpu')
assert "model" in from_pth.keys(), "The .pth file does not contain the model weights"
weights = from_pth["model"]
torch.save(weights, to_pth)
print("Weights are saved to {}".format(to_pth))
# %%
