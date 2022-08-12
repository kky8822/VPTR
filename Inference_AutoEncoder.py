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
    visualize_batch_clips_inference,
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
from utils.metrics import PSNR, SSIM, LPIPS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

PSNR_metric = PSNR
SSIM_metric = SSIM()
LPIPS_metric = LPIPS

set_seed(2021)

use_l1_loss = False


def show_samples(
    idx,
    VPTR_Enc,
    VPTR_Dec,
    sample,
    save_dir,
    renorm_transform,
    device=torch.device("cuda:0"),
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
        pred_future_frames = VPTR_Dec(
            VPTR_Enc(future_frames[:, :, 0:img_channels, ...])
        )
        rec_frames = torch.cat((pred_past_frames, pred_future_frames), dim=1)

    # psnr_sol_t = torch.zeros(pred_frames.shape[1])
    # psnr_w_t = torch.zeros(pred_frames.shape[1])
    # ssim_sol_t = torch.zeros(pred_frames.shape[1])
    # ssim_w_t = torch.zeros(pred_frames.shape[1])
    # lpips_sol_t = torch.zeros(pred_frames.shape[1])
    # lpips_w_t = torch.zeros(pred_frames.shape[1])
    gt_frames = torch.cat(
        (
            past_frames[:, :, 0:pred_channels, ...],
            future_frames[:, :, 0:pred_channels, ...],
        ),
        dim=1,
    )
    N, T, C, X, Y = gt_frames.shape
    psnr = torch.zeros((T, C))
    ssim = torch.zeros((T, C))
    for c in range(C):
        for t in range(T):
            psnr[t, c] = PSNR_metric(
                gt_frames[:, t, c, ...].unsqueeze(1),
                rec_frames[:, t, c, ...].unsqueeze(1),
            )
            ssim[t, c] = SSIM_metric(
                gt_frames[:, t, c, ...].unsqueeze(1),
                rec_frames[:, t, c, ...].unsqueeze(1),
            )

    psnr = psnr.cpu()
    ssim = ssim.cpu()

    # N = rec_frames.shape[0]
    # # idx = min(N, 4)
    # if idx == 0:
    #     visualize_batch_clips_inference(
    #         idx,
    #         past_frames[0:N, :, 0:pred_channels, ...],
    #         future_frames[0:N, :, 0:pred_channels, ...],
    #         rec_frames[0:N, :, ...],
    #         save_dir,
    #         renorm_transform,
    #         desc="AE",
    #     )

    del past_frames
    del future_frames
    del rec_frames
    torch.cuda.empty_cache()  # GPU 캐시 데이터 삭제

    # return psnr_sol_t, psnr_w_t, ssim_sol_t, ssim_w_t, lpips_sol_t, lpips_w_t
    return psnr, ssim


if __name__ == "__main__":
    if sys.argv[1] == "nsbd":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_tensorboard"
        )
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-12000/pngs"
        img_channels = 1  # 3 channels for BAIR datset
    elif sys.argv[1] == "kth":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_tensorboard"
        )
        data_set_name = "KTH"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/kth_action/pngs"
        device = torch.device("cuda:0")
        img_channels = 1  # 3 channels for BAIR datset
    elif sys.argv[1] == "mnist":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0725_MNIST_ResNetAE_MSEGDLgan_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0725_MNIST_ResNetAE_MSEGDLgan_tensorboard"
        )
        data_set_name = "MNIST"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/MovingMNIST"
        img_channels = 1  # 3 channels for BAIR datset
        device = torch.device("cuda:0")
    elif sys.argv[1] == "nsbd-field":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_tensorboard"
        )
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-field-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        img_channels = 2  # 3 channels for BAIR datset
    elif sys.argv[1] == "nsbd-total":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_tensorboard"
        )
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-total-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-total-12000"
        img_channels = 7  # 3 channels for BAIR datset
        visc_list = [sys.argv[2]]
        # visc_list = "total"
        transform_norm = "min_max"
        img_channels = 1  # 3 channels for BAIR datset
        pred_channels = 4  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy

    resume_ckpt = ckpt_save_dir.joinpath("epoch_20.tar")
    # print(resume_ckpt)
    # resume_ckpt = None
    start_epoch = 0

    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    num_past_frames = 10
    num_future_frames = 40
    encH, encW, encC = 8, 8, 528

    epochs = 1000
    N = 64
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
        visc_list=visc_list,
        transform_norm=transform_norm,
    )

    #####################Init Models and Optimizer ###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim=encC, n_downsampling=3).to(device)
    VPTR_Dec = VPTRDec(
        pred_channels, feat_dim=encC, n_downsampling=3, out_layer="Tanh"
    ).to(
        device
    )  # Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Disc = VPTRDisc(
        pred_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d
    ).to(device)
    init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    optimizer_G = torch.optim.Adam(
        params=list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()),
        lr=AE_lr,
        betas=(0.5, 0.999),
    )
    optimizer_D = torch.optim.Adam(
        params=VPTR_Disc.parameters(), lr=D_lr, betas=(0.5, 0.999)
    )

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

    ##################### Inference ################################

    psnr = 0
    ssim = 0
    # lpips_sol_t = 0
    # lpips_w_t = 0
    for idx, sample in enumerate(test_loader):
        (
            b_psnr,
            b_ssim,
            # b_lpips_sol_t,
            # b_lpips_w_t,
        ) = show_samples(
            idx,
            VPTR_Enc,
            VPTR_Dec,
            sample,
            ckpt_save_dir.joinpath("inference_gifs" + visc_list[0]),
            renorm_transform,
            device=device,
            img_channels=img_channels,  # 3 channels for BAIR datset
            pred_channels=pred_channels,  # 1: w, 4: w, dw/dx, dw/dy, d2w/dxdy
        )
        psnr += b_psnr / len(test_loader)
        ssim += b_ssim / len(test_loader)

        print(f"{idx}/{len(test_loader)}")
        # break

    columns = (
        ["t"]
        + ["psnr_" + str(i) for i in range(psnr.shape[1])]
        + ["ssim_" + str(i) for i in range(psnr.shape[1])]
    )
    data = np.hstack((psnr.numpy(), ssim.numpy()))
    data = np.hstack((np.arange(len(psnr)).reshape(-1, 1), data))

    df = pd.DataFrame(columns=columns, data=data)
    df.to_csv(
        ckpt_save_dir.joinpath("AE_test_metrics_" + visc_list[0] + ".csv"), index=False
    )
