import wandb
import os

os.environ['WANDB_API_KEY'] = 'd8aa8d5b3f2d3edfcdc12240ab2387d3d9399fe4'

api = wandb.Api()

# download only images from media in wandb
run = api.run("/CycleGAN-and-pix2pix/runs/bsfa75j6")

for file in run.files():
    if file.name.endswith('.png'):
        file.download(replace=True)
        print(f"Downloaded {file.name}")