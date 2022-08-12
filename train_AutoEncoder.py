import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import get_dataloader
from utils import (
    VidCenterCrop,
    VidPad,
    VidResize,
    VidNormalize,
    VidReNormalize,
    VidCrop,
    VidRandomHorizontalFlip,
    VidRandomVerticalFlip,
    VidToTensor,
)
from utils import (
    visualize_batch_clips,
    save_ckpt,
    load_ckpt,
    set_seed,
    AverageMeters,
    init_loss_dict,
    write_summary,
    resume_training,
)
from utils import set_seed

import os, sys

set_seed(2021)

use_l1_loss = False


def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    # print(fake_imgs.shape, real_imgs.shape)
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0, 1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real


def cal_lossG(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
    loss_G_gan = gan_loss(pred_fake, True)

    if use_l1_loss == True:
        AE_MSE_loss = l1_loss(fake_imgs, real_imgs)
    else:
        AE_MSE_loss = mse_loss(fake_imgs, real_imgs)
    AE_GDL_loss = gdl_loss(real_imgs, fake_imgs)

    loss_G = lam_gan * loss_G_gan + AE_MSE_loss + AE_GDL_loss

    return loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss


def single_iter(
    VPTR_Enc,
    VPTR_Dec,
    VPTR_Disc,
    optimizer_G,
    optimizer_D,
    sample,
    device,
    train_flag=True,
    img_channels=1,
    pred_channels=1,
):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    # print("past, future: ", past_frames.shape, future_frames.shape)
    x = torch.cat([past_frames, future_frames], dim=1)
    x_in = x[:, :, 0:img_channels, ...]
    x_out = x[:, :, 0:pred_channels, ...]

    if train_flag:
        VPTR_Enc = VPTR_Enc.train()
        VPTR_Enc.zero_grad()
        VPTR_Dec = VPTR_Dec.train()
        VPTR_Dec.zero_grad()

        rec_feats = VPTR_Enc(x_in)
        # print(rec_feats.shape)
        rec_frames = VPTR_Dec(rec_feats)

        # update discriminator
        VPTR_Disc = VPTR_Disc.train()
        for p in VPTR_Disc.parameters():
            p.requires_grad_(True)
        VPTR_Disc.zero_grad(set_to_none=True)
        # print(train_flag, rec_frames.shape, x_out.shape)
        loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, rec_frames, x_out, lam_gan)
        loss_D.backward()
        optimizer_D.step()

        # update autoencoder (generator)
        for p in VPTR_Disc.parameters():
            p.requires_grad_(False)
        loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss = cal_lossG(VPTR_Disc, rec_frames, x_out, lam_gan)
        loss_G.backward()
        optimizer_G.step()
    else:
        VPTR_Enc = VPTR_Enc.eval()
        VPTR_Dec = VPTR_Dec.eval()
        VPTR_Disc = VPTR_Disc.eval()
        with torch.no_grad():
            rec_frames = VPTR_Dec(VPTR_Enc(x_in))
            # print(train_flag, rec_frames.shape, x_out.shape)
            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, rec_frames, x_out, lam_gan)
            loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss = cal_lossG(VPTR_Disc, rec_frames, x_out, lam_gan)

    iter_loss_dict = {
        "AEgan": loss_G_gan.item(),
        "AE_MSE": AE_MSE_loss.item(),
        "AE_GDL": AE_GDL_loss.item(),
        "AE_total": loss_G.item(),
        "Dtotal": loss_D.item(),
        "Dfake": loss_D_fake.item(),
        "Dreal": loss_D_real.item(),
    }

    return iter_loss_dict


def show_samples(
    VPTR_Enc,
    VPTR_Dec,
    sample,
    save_dir,
    renorm_transform,
    img_channels=1,
    pred_channels=1,
):
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()
    with torch.no_grad():
        past_frames, future_frames = sample
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        pred_past_frames = VPTR_Dec(VPTR_Enc(past_frames[:, :, 0:img_channels, ...]))
        pred_future_frames = VPTR_Dec(VPTR_Enc(future_frames[:, :, 0:img_channels, ...]))
        rec_frames = torch.cat((pred_past_frames, pred_future_frames), dim=1)

        N = past_frames.shape[0]
        idx = min(N, 4)
        visualize_batch_clips(
            past_frames[0:idx, :, 0:pred_channels, ...],
            future_frames[0:idx, :, 0:pred_channels, ...],
            rec_frames[0:idx, :, ...],
            save_dir,
            renorm_transform,
            desc="ae",
        )


if __name__ == "__main__":
    if sys.argv[1] == "nsbd":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path("/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_tensorboard")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-12000/pngs"
        img_channels = 1  # 3 channels for BAIR datset
    elif sys.argv[1] == "kth":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path("/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_tensorboard")
        data_set_name = "KTH"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/kth_action/pngs"
        device = torch.device("cuda:0")
        img_channels = 1  # 3 channels for BAIR datset
    elif sys.argv[1] == "mnist":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0725_MNIST_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path("/kky/VPTR/VPTR_ckpts/0725_MNIST_ResNetAE_MSEGDLgan_tensorboard")
        data_set_name = "MNIST"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/MovingMNIST"
        img_channels = 1  # 3 channels for BAIR datset
        device = torch.device("cuda:0")
    elif sys.argv[1] == "nsbd-field":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path("/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_tensorboard")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-field-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        img_channels = 2  # 3 channels for BAIR datset
    elif sys.argv[1] == "nsbd-total":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_tensorboard"
        )
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-total-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-total-12000"
        img_channels = 1  # 3 channels for BAIR datset
        pred_channels = 4  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
        transform_norm = "min_max"
        visc_list = "total"

    # resume_ckpt = ckpt_save_dir.joinpath("epoch_10.tar")
    # print(resume_ckpt)
    resume_ckpt = None
    start_epoch = 0

    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    num_past_frames = 10
    num_future_frames = 40
    encH, encW, encC = 8, 8, 528

    epochs = 1000
    N = 16
    AE_lr = 2e-4
    D_lr = 2e-4
    lam_gan = 0.01

    #####################Init Dataset ###########################

    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(
        data_set_name,
        N,
        dataset_dir,
        num_past_frames,
        num_future_frames,
        transform_norm=transform_norm,
        visc_list=visc_list,
    )

    #####################Init Models and Optimizer ###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim=encC, n_downsampling=3).to(device)
    VPTR_Dec = VPTRDec(pred_channels, feat_dim=encC, n_downsampling=3, out_layer="Tanh").to(
        device
    )  # Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Disc = VPTRDisc(pred_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    optimizer_G = torch.optim.Adam(
        params=list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()),
        lr=AE_lr,
        betas=(0.5, 0.999),
    )
    optimizer_D = torch.optim.Adam(params=VPTR_Disc.parameters(), lr=D_lr, betas=(0.5, 0.999))

    Enc_parameters = sum(p.numel() for p in VPTR_Enc.parameters() if p.requires_grad)
    Dec_parameters = sum(p.numel() for p in VPTR_Dec.parameters() if p.requires_grad)
    Disc_parameters = sum(p.numel() for p in VPTR_Disc.parameters() if p.requires_grad)
    print(f"Encoder num_parameters: {Enc_parameters}")
    print(f"Decoder num_parameters: {Dec_parameters}")
    print(f"Discriminator num_parameters: {Disc_parameters}")

    #####################Init Criterion ###########################
    loss_name_list = [
        "AE_MSE",
        "AE_GDL",
        "AE_total",
        "Dtotal",
        "Dfake",
        "Dreal",
        "AEgan",
    ]
    # gan_loss = GANLoss("vanilla", target_real_label=1.0, target_fake_label=0.0).to(device)
    gan_loss = GANLoss("lsgan", target_real_label=1.0, target_fake_label=0.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    l1_loss = L1Loss()
    gdl_loss = GDL(alpha=1)

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training(
            {"VPTR_Enc": VPTR_Enc, "VPTR_Dec": VPTR_Dec, "VPTR_Disc": VPTR_Disc},
            {"optimizer_G": optimizer_G, "optimizer_D": optimizer_D},
            resume_ckpt,
            loss_name_list,
        )

    #####################Training loop ###########################
    # import gc

    # gc.collect()
    # torch.cuda.empty_cache()

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        epoch_st = datetime.now()

        # Train
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(train_loader, 0):
            iter_loss_dict = single_iter(
                VPTR_Enc,
                VPTR_Dec,
                VPTR_Disc,
                optimizer_G,
                optimizer_D,
                sample,
                device,
                train_flag=True,
                img_channels=img_channels,  # 3 channels for BAIR datset
                pred_channels=pred_channels,  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
            )
            EpochAveMeter.iter_update(iter_loss_dict)

        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag=True)
        write_summary(summary_writer, loss_dict, train_flag=True)

        show_samples(
            VPTR_Enc,
            VPTR_Dec,
            sample,
            ckpt_save_dir.joinpath(f"train_gifs_epoch{epoch}"),
            renorm_transform,
            img_channels=img_channels,
            pred_channels=pred_channels,
        )

        # validation
        EpochAveMeter = AverageMeters(loss_name_list)
        for idx, sample in enumerate(val_loader, 0):
            iter_loss_dict = single_iter(
                VPTR_Enc,
                VPTR_Dec,
                VPTR_Disc,
                optimizer_G,
                optimizer_D,
                sample,
                device,
                train_flag=False,
                img_channels=img_channels,  # 3 channels for BAIR datset
                pred_channels=pred_channels,  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
            )
            EpochAveMeter.iter_update(iter_loss_dict)
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag=False)
        write_summary(summary_writer, loss_dict, train_flag=False)

        # if epoch == 50:
        # VPTR_Enc = VPTR_Enc.to("cpu")
        # VPTR_Dec = VPTR_Dec.to("cpu")
        # VPTR_Disc = VPTR_Disc.to("cpu")
        save_ckpt(
            {"VPTR_Enc": VPTR_Enc, "VPTR_Dec": VPTR_Dec, "VPTR_Disc": VPTR_Disc},
            {"optimizer_G": optimizer_G, "optimizer_D": optimizer_D},
            epoch,
            loss_dict,
            ckpt_save_dir,
        )
        # VPTR_Enc = VPTR_Enc.to(device)
        # VPTR_Dec = VPTR_Dec.to(device)
        # VPTR_Disc = VPTR_Disc.to(device)

        for idx, sample in enumerate(test_loader, random.randint(0, len(test_loader) - 1)):
            show_samples(
                VPTR_Enc,
                VPTR_Dec,
                sample,
                ckpt_save_dir.joinpath(f"test_gifs_epoch{epoch}"),
                renorm_transform,
                img_channels=img_channels,
                pred_channels=pred_channels,
            )
            break

        epoch_time = datetime.now() - epoch_st
        print(f"epoch {epoch}", EpochAveMeter.meters["AE_total"])
        print(
            f"Estimated remaining training time: {epoch_time.total_seconds()/3600. * (start_epoch + epochs - epoch)} Hours"
        )
