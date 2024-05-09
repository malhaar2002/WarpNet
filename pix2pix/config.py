import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 2e-4
batch_size = 1
num_workers = 2
image_size = 256
channels_img = 3
l1_lambda = 100
num_epochs = 100
load_model = False
save_model = True

transform_both = A.Compose(
    [
        A.Resize(width=image_size, height=image_size),
        A.HorizontalFlip(p=0.5),],
        additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)


