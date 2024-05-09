import torch
from utils import save_examples, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataloader import CocoDataset
import torch.nn as nn
import torch.optim as optim
import config
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm


def train_fn(discriminator, generator, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_fake = discriminator(x, y_fake.detach())
            D_loss = (bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))) / 2


        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            gen_loss_fake = bce(D_fake, torch.ones_like(D_fake))
            l1 = l1_loss(y_fake, y) * config.l1_lambda
            gen_loss = gen_loss_fake + l1

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # update tqdm loop
        loop.set_postfix(d_loss=D_loss.item(), g_loss=gen_loss_fake.item())


def main():
    discriminator = Discriminator(in_channels=3).to(config.device)
    generator = Generator(in_channels=3).to(config.device)
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.load_model:
        load_checkpoint(
            config.checkpoint_file, generator, opt_gen, config.learning_rate,
        )
        load_checkpoint(
            config.checkpoint_file, discriminator, opt_disc, config.learning_rate,
        )

    dataset = CocoDataset(root="D:\WarpNET\coco_persons")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()



    for epoch in range(config.num_epochs):
        train_fn(discriminator, generator, loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.save_model and epoch % 5 == 0:
            save_checkpoint(generator, opt_gen, filename="gen.pth.tar")
            save_checkpoint(discriminator, opt_disc, filename="disc.pth.tar")

        save_examples(generator, loader, epoch, folder="saved_images")


if __name__ == "__main__":
    main()