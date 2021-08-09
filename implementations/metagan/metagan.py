import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="PACS", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images_m/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models_m/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# a/b are meta-S/meta-T images
# A/B are S/T images

# Initialize generator and discriminator
G_AB = MultiInputGeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = MultiInputGeneratorResNet(input_shape, opt.n_residual_blocks)
D_a = Discriminator(input_shape)
D_b = Discriminator(input_shape)
D_A = MultiInputDiscriminator(input_shape)
D_B = MultiInputDiscriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    D_a = D_a.cuda()
    D_b = D_b.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models_m/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models_m/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models_m/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models_m/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_a.load_state_dict(torch.load("saved_models_m/%s/D_a_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_b.load_state_dict(torch.load("saved_models_m/%s/D_b_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    D_a.apply(weights_init_normal)
    D_b.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_a = torch.optim.Adam(D_a.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_b = torch.optim.Adam(D_b.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_a = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_a, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_b = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_b, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_a_buffer = ReplayBuffer()
fake_b_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(opt.img_height, Image.BICUBIC),
#     transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
#     transforms.RandomCrop((opt.img_height, opt.img_width)),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_gray_ = [
    transforms.Resize(opt.img_height, Image.BICUBIC),
#     transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
#     transforms.RandomCrop((opt.img_height, opt.img_width)),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

# Training data loader
dataloader = DataLoader(
    PACS_Dataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, transforms_gray_=transforms_gray_),
#     ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    PACS_Dataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test", transforms_gray_=transforms_gray_),
#     ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=4,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    real_gray_A = Variable(imgs["gray_A"].type(Tensor))
    fake_B, fake_gray_B = G_AB(real_A, real_gray_A)
    real_B = Variable(imgs["B"].type(Tensor))
    real_gray_B = Variable(imgs["gray_B"].type(Tensor))
    fake_A, fake_gray_A = G_BA(real_B, real_gray_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    real_gray_A = make_grid(real_gray_A, nrow=5, normalize=True)
    fake_gray_B = make_grid(fake_gray_B, nrow=5, normalize=True)
    real_gray_B = make_grid(real_gray_B, nrow=5, normalize=True)
    fake_gray_A = make_grid(fake_gray_A, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, real_gray_A, fake_B, fake_gray_B, real_B, real_gray_B, fake_A, fake_gray_A), 1)
    save_image(image_grid, "images_m/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        real_gray_A = Variable(batch["gray_A"].type(Tensor))
        real_gray_B = Variable(batch["gray_B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        i_fake_A, i_fake_gray_A = G_BA(real_A, real_gray_A)
        i_fake_B, i_fake_gray_B = G_AB(real_B, real_gray_B)

        loss_id_A = criterion_identity(i_fake_A, real_A)
        loss_id_B = criterion_identity(i_fake_B, real_B)
        loss_id_a = criterion_identity(i_fake_gray_A, real_gray_A)
        loss_id_b = criterion_identity(i_fake_gray_B, real_gray_B)

        loss_identity = (loss_id_A + loss_id_B) / 2
        loss_identity_meta = (loss_id_a + loss_id_a) / 2

        # GAN loss
        fake_B, fake_gray_B = G_AB(real_A, real_gray_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B, fake_gray_B), valid)
        loss_GAN_AB_meta = criterion_GAN(D_b(fake_gray_B), valid)
        fake_A, fake_gray_A = G_BA(real_B, real_gray_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A, fake_gray_A), valid)
        loss_GAN_BA_meta = criterion_GAN(D_a(fake_gray_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        loss_GAN_meta = (loss_GAN_AB_meta + loss_GAN_BA_meta) / 2

        # Cycle loss
        recov_A, recov_gray_A = G_BA(fake_B, fake_gray_B)
        loss_cycle_A, loss_cycle_a = criterion_cycle(recov_A, real_A), criterion_cycle(recov_gray_A, real_gray_A)
        recov_B, recov_gray_B = G_AB(fake_A, fake_gray_A)
        loss_cycle_B, loss_cycle_b = criterion_cycle(recov_B, real_B), criterion_cycle(recov_gray_B, real_gray_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        loss_cycle_meta = (loss_cycle_a + loss_cycle_b) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
        loss_G_meta = loss_GAN_meta + opt.lambda_cyc * loss_cycle_meta + opt.lambda_id * loss_identity_meta
        loss_G_all = loss_G + loss_G_meta

        loss_G_all.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real, loss_real_meta = criterion_GAN(D_A(real_A, real_gray_A), valid), criterion_GAN(D_a(real_gray_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        fake_a_ = fake_a_buffer.push_and_pop(fake_gray_A)
        loss_fake, loss_fake_meta = criterion_GAN(D_A(fake_A_.detach(), fake_a_.detach()), fake), criterion_GAN(D_a(fake_a_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A_meta = (loss_real_meta + loss_fake_meta) / 2
        loss_D_A_all = loss_D_A + loss_D_A_meta

        loss_D_A_all.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real, loss_real_meta = criterion_GAN(D_B(real_B, real_gray_B), valid), criterion_GAN(D_b(real_gray_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        fake_b_ = fake_b_buffer.push_and_pop(fake_gray_B)
        loss_fake, loss_fake_meta = criterion_GAN(D_B(fake_B_.detach(), fake_b_.detach()), fake), criterion_GAN(D_b(fake_b_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B_meta = (loss_real_meta + loss_fake_meta) / 2
        loss_D_B_all = loss_D_B + loss_D_B_meta

        loss_D_B_all.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        loss_D_meta = (loss_D_A_meta + loss_D_B_meta) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: (%.3f, %.3f)] [G loss: (%.3f, %.3f), adv: (%.3f, %.3f), cycle: (%.3f, %.3f), identity: (%.3f, %.3f)] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_D_meta.item(),
                loss_G.item(),
                loss_G_meta.item(),
                loss_GAN.item(),
                loss_GAN_meta.item(),
                loss_cycle.item(),
                loss_cycle_meta.item(),
                loss_identity.item(),
                loss_identity_meta.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_D_a.step()
    lr_scheduler_D_b.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models_m/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models_m/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models_m/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models_m/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_a.state_dict(), "saved_models_m/%s/D_a_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_b.state_dict(), "saved_models_m/%s/D_b_%d.pth" % (opt.dataset_name, epoch))
