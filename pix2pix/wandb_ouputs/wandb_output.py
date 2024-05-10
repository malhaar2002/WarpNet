import wandb
import os
from secret import wandb_api

os.environ['WANDB_API_KEY'] = wandb_api

api = wandb.Api()

# download only images from media in wandb
run = api.run("/CycleGAN-and-pix2pix/runs/bsfa75j6")

for file in run.files():
    if file.name.endswith('.png'):
        file.download(replace=True)
        print(f"Downloaded {file.name}")