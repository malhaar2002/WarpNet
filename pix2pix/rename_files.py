# iterate through D:\WarpNET\media\fake_B and rename all files
import os

path = r"D:\WarpNET\media\train"
real_dir = r"D:\WarpNET\media\real_A"
fake_dir = r"D:\WarpNET\media\fake_B"

for i, file in enumerate(os.listdir(path)):
    split = file.split("_")
    name = split[0]
    if split[1] == "real":
        # move to real directory
        os.rename(os.path.join(path, file), os.path.join(real_dir, f"{name}.png"))
    elif split[1] == "fake":
        # move to fake directory
        os.rename(os.path.join(path, file), os.path.join(fake_dir, f"{name}.png"))