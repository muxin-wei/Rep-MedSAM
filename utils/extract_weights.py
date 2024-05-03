# %%
import torch
import argparse
import os
join = os.path.join

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-from_pth", type=str, default='',
                    help="Path to the .pth file from which the weights will be extracted")
parser.add_argument("-to_pth", type=str, default= '',
                    help="Path to the .pth file to which the weights will be saved")
args = parser.parse_args()

# %%
from_pth = '/data/Rep-MedSAM/workdir/finetune/medsam_lite_latest.pth'
to_pth = '/data/Rep-MedSAM/workdir/finetune/refine_weights.pth'

# %%
from_pth = torch.load(from_pth, map_location='cpu')
assert "model" in from_pth.keys(), "The .pth file does not contain the model weights"
weights = from_pth["model"]
torch.save(weights, to_pth)
print("Weights are saved to {}".format(to_pth))
# %%
