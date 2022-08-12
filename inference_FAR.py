import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR, VPTRFormerFAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import get_dataloader
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

import logging

from utils.metrics import PSNR, SSIM, LPIPS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

PSNR_metric = PSNR
SSIM_metric = SSIM()
LPIPS_metric = LPIPS


def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0, 1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real


def cal_lossT(fake_imgs, real_imgs, VPTR_Disc, lam_gan):
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)

    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss

    return loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan


def single_iter(
    VPTR_Enc,
    VPTR_Dec,
    VPTR_Disc,
    VPTR_Transformer,
    optimizer_T,
    optimizer_D,
    sample,
    device,
    lam_gan,
    train_flag=True,
):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)

    # print("past, future: ", past_frames.shape, future_frames.shape)

    with torch.no_grad():
        x = torch.cat([past_frames, future_frames[:, 0:-1, ...]], dim=1)
        # print("x: ", past_frames.shape, future_frames[:, 0:-1, ...].shape, x.shape)
        gt_feats = VPTR_Enc(x)
        # print(gt_feats.shape)

    if train_flag:
        VPTR_Transformer = VPTR_Transformer.train()
        VPTR_Transformer.zero_grad(set_to_none=True)
        VPTR_Dec.zero_grad(set_to_none=True)

        pred_future_feats = VPTR_Transformer(gt_feats)
        pred_frames = VPTR_Dec(pred_future_feats)

        # print("pred_frames: ", pred_frames.shape)

        if optimizer_D is not None:
            assert lam_gan is not None, "Input lam_gan"
            # update discriminator
            VPTR_Disc = VPTR_Disc.train()
            for p in VPTR_Disc.parameters():
                p.requires_grad_(True)
            VPTR_Disc.zero_grad(set_to_none=True)
            loss_D, loss_D_fake, loss_D_real = cal_lossD(
                VPTR_Disc, pred_frames, future_frames, lam_gan
            )
            loss_D.backward()
            optimizer_D.step()

            for p in VPTR_Disc.parameters():
                p.requires_grad_(False)

        # update Transformer (generator)
        # print("gt: ", past_frames[:, 1:, ...].shape, future_frames.shape, torch.cat([past_frames[:, 1:, ...], future_frames], dim=1).shape)
        loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(
            pred_frames,
            torch.cat([past_frames[:, 1:, ...], future_frames], dim=1),
            VPTR_Disc,
            lam_gan,
        )
        loss_T.backward()
        nn.utils.clip_grad_norm_(
            VPTR_Transformer.parameters(), max_norm=max_grad_norm, norm_type=2
        )
        optimizer_T.step()

    else:
        if optimizer_D is not None:
            VPTR_Disc = VPTR_Disc.eval()
        VPTR_Transformer = VPTR_Transformer.eval()
        with torch.no_grad():
            pred_future_feats = VPTR_Transformer(gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
            if optimizer_D is not None:
                loss_D, loss_D_fake, loss_D_real = cal_lossD(
                    VPTR_Disc, pred_frames, future_frames, lam_gan
                )
            loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(
                pred_frames,
                torch.cat([past_frames[:, 1:, ...], future_frames], dim=1),
                VPTR_Disc,
                lam_gan,
            )

    if optimizer_D is None:
        loss_D, loss_D_fake, loss_D_real = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )

    iter_loss_dict = {
        "T_total": loss_T.item(),
        "T_MSE": T_MSE_loss.item(),
        "T_GDL": T_GDL_loss.item(),
        "T_gan": loss_T_gan.item(),
        "Dtotal": loss_D.item(),
        "Dfake": loss_D_fake.item(),
        "Dreal": loss_D_real.item(),
    }

    return iter_loss_dict


def FAR_show_sample(
    idx,
    VPTR_Enc,
    VPTR_Dec,
    VPTR_Transformer,
    num_pred,
    sample,
    save_dir,
    test_phase=True,
    device=torch.device("cuda:0"),
):
    VPTR_Transformer = VPTR_Transformer.eval()
    # print(device)
    with torch.no_grad():
        past_frames, future_frames = sample
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        past_gt_feats = VPTR_Enc(past_frames)  # 0...9
        future_gt_feats = VPTR_Enc(future_frames)

        if test_phase:
            pred_feats = VPTR_Transformer(past_gt_feats)  # 0,...,9 -> 1,...10
            for i in range(num_pred - 1):
                if i == 0:
                    input_feats = torch.cat(
                        [past_gt_feats, pred_feats[:, -1:, ...]], dim=1
                    )  # 0,...,9 + 10
                else:
                    pred_future_frame = VPTR_Dec(pred_feats[:, -1:, ...])  # 11
                    pred_future_feat = VPTR_Enc(pred_future_frame)  # 11
                    input_feats = torch.cat(
                        [input_feats, pred_future_feat], dim=1
                    )  # 0, ..., 11

                pred_feats = VPTR_Transformer(input_feats)  # 0,...,10 -> 1,...,19
        else:
            input_feats = torch.cat(
                [past_gt_feats, future_gt_feats[:, 0:-1, ...]], dim=1
            )
            pred_feats = VPTR_Transformer(input_feats)

        pred_feats = torch.cat((past_gt_feats[:, 0:1, ...], pred_feats), dim=1)
        pred_frames = VPTR_Dec(pred_feats)
    # pred_past_frames = pred_frames[:, 0:-num_pred, ...]
    # pred_future_frames = pred_frames[:, -num_pred:, ...]
    # pred_frames = torch.cat((pred_past_frames, pred_future_frames), dim=1)
    # print("pred frames: ", pred_frames.shape, torch.max(pred_frames))

    psnr_sol_t = torch.zeros(pred_frames.shape[1])
    psnr_w_t = torch.zeros(pred_frames.shape[1])
    ssim_sol_t = torch.zeros(pred_frames.shape[1])
    ssim_w_t = torch.zeros(pred_frames.shape[1])
    lpips_sol_t = torch.zeros(pred_frames.shape[1])
    lpips_w_t = torch.zeros(pred_frames.shape[1])
    for i, (true, pred) in enumerate(
        zip(
            torch.cat((past_frames, future_frames), dim=1).permute(1, 0, 2, 3, 4),
            pred_frames.permute(1, 0, 2, 3, 4),
        )
    ):
        # start = time.time()  # 시작 시간 저장
        psnr_sol_t[i] = PSNR_metric(true[:, 0:1, ...], pred[:, 0:1, ...])
        psnr_w_t[i] = PSNR_metric(true[:, -1:, ...], pred[:, -1:, ...])
        # print("psnr_time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

        # start = time.time()  # 시작 시간 저장
        ssim_sol_t[i] = SSIM_metric(true[:, 0:1, ...], pred[:, 0:1, ...])
        ssim_w_t[i] = SSIM_metric(true[:, -1:, ...], pred[:, -1:, ...])
        # print("ssim_time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

        # start = time.time()  # 시작 시간 저장
        # lpips_sol_t[i] = LPIPS_metric(
        #     true[:, 0:1, ...].tile((1, 3, 1, 1)),
        #     pred[:, 0:1, ...].tile((1, 3, 1, 1)),
        #     device=device,
        # )
        # lpips_w_t[i] = LPIPS_metric(
        #     true[:, -1:, ...].tile((1, 3, 1, 1)),
        #     pred[:, -1:, ...].tile((1, 3, 1, 1)),
        #     device=device,
        # )
        # print("lpips_time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    psnr_sol_t = psnr_sol_t.cpu()
    psnr_w_t = psnr_w_t.cpu()
    ssim_sol_t = ssim_sol_t.cpu()
    ssim_w_t = ssim_w_t.cpu()
    # lpips_sol_t = lpips_sol_t.cpu()
    # lpips_w_t = lpips_w_t.cpu()

    # N = pred_future_frames.shape[0]
    N = pred_frames.shape[0]
    # idx = min(N, 4)
    if idx == 0:
        visualize_batch_clips_inference(
            idx,
            past_frames[0:N, :, ...],
            future_frames[0:N, :, ...],
            pred_frames[0:N, :, ...],
            save_dir,
            renorm_transform,
            desc="FAR",
        )

    del past_frames
    del future_frames
    del pred_frames
    torch.cuda.empty_cache()  # GPU 캐시 데이터 삭제

    # return psnr_sol_t, psnr_w_t, ssim_sol_t, ssim_w_t, lpips_sol_t, lpips_w_t
    return psnr_sol_t, psnr_w_t, ssim_sol_t, ssim_w_t

    # visualize_batch_clips(
    #     past_frames[0:idx, 1:, ...],
    #     pred_past_frames[0:idx, :, ...],
    #     pred_future_frames[0:idx, :-1, ...],
    #     save_dir,
    #     renorm_transform,
    #     desc="pred_past",
    # )


if __name__ == "__main__":
    set_seed(2021)

    if sys.argv[1] == "ns":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0722_NS_10_FAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0722_NS_10_FAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0722_NS_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_100.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255/pngs"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
    elif sys.argv[1] == "nsbd":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0723_NSBD_10_FAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_FAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0723_NSBD_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_20.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-12000/pngs"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
    elif sys.argv[1] == "kth":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0721_KTH_FAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0721_KTH_FAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0721_KTH_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_50.tar")
        data_set_name = "KTH"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/kth_action/pngs"
        device = torch.device("cuda:1")
        num_past_frames = 10
        num_future_frames = 40
        test_past_frames = 10
        test_future_frames = 40

    elif sys.argv[1] == "nsbd-field":
        ckpt_save_dir = Path("/kky/VPTR/VPTR_ckpts/0729_NSBDField_40_FAR_MSEGDL_ckpt")
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0729_NSBDField_40_FAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0728_NSBDField_40_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_23.tar")
        device = torch.device("cuda:0")
        data_set_name = "Navier-Stokes-field-BD"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        num_past_frames = 10
        num_future_frames = 40
        test_past_frames = 10
        test_future_frames = 40
        img_channels = 2  # 3 channels for BAIR datset
        window_size = 4
        resume_ckpt = ckpt_save_dir.joinpath("epoch_47.tar")
        rk = False

    elif sys.argv[1] == "nsbd-field-test":
        ckpt_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_FAR_MSEGDL_ckpt"
        )
        tensorboard_save_dir = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_FAR_MSEGDL_tensorboard"
        )
        resume_AE_ckpt = Path(
            "/kky/VPTR/VPTR_ckpts/0727_test_NSBDField_10_ResNetAE_MSEGDLgan_ckpt"
        ).joinpath("epoch_1.tar")
        device = torch.device("cuda:1")
        data_set_name = "Navier-Stokes-field-BD-test"  # see utils.dataset
        dataset_dir = "/kky/VPTR/dataset/navier-stokes-255-field-12000"
        num_past_frames = 10
        num_future_frames = 10
        test_past_frames = 10
        test_future_frames = 10
        img_channels = 2  # 3 channels for BAIR datset
        window_size = 4
        resume_ckpt = None

    #############Set the logger#########
    if not Path(ckpt_save_dir).exists():
        Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%a, %d %b %Y %H:%M:%S",
        format="%(asctime)s - %(message)s",
        filename=ckpt_save_dir.joinpath("train_log.log").absolute().as_posix(),
        filemode="a",
    )

    start_epoch = 0
    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())

    encH, encW, encC = 8, 8, 528
    # img_channels = 1  # 3 channels for BAIR
    epochs = 1
    N = 64
    # AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0
    rpe = False
    lam_gan = 0.001
    dropout = 0.1
    val_per_epochs = 1

    #####################Init Dataset ###########################

    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(
        data_set_name, N, dataset_dir, test_past_frames, test_future_frames
    )

    #####################Init model###########################
    VPTR_Enc = VPTREnc(img_channels, feat_dim=encC, n_downsampling=3).to(device)
    VPTR_Dec = VPTRDec(
        img_channels, feat_dim=encC, n_downsampling=3, out_layer="Tanh"
    ).to(
        device
    )  # Tanh for KTH and BAIR
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    # VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    # VPTR_Disc = VPTR_Disc.eval()
    # init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    VPTR_Transformer = VPTRFormerFAR(
        num_past_frames,
        num_future_frames,
        encH=encH,
        encW=encW,
        d_model=encC,
        nhead=8,
        num_encoder_layers=12,
        dropout=dropout,
        window_size=window_size,
        Spatial_FFN_hidden_ratio=4,
        rpe=rpe,
        device=device,
        rk=rk,
    ).to(device)

    optimizer_D = None
    # optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
    optimizer_T = torch.optim.AdamW(
        params=VPTR_Transformer.parameters(), lr=Transformer_lr
    )

    Transformer_parameters = sum(
        p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad
    )
    print(f"FAR Transformer num_parameters: {Transformer_parameters}")

    #####################Init loss function###########################
    loss_name_list = ["T_MSE", "T_GDL", "T_gan", "T_total", "Dtotal", "Dfake", "Dreal"]
    # gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha=1)

    # load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training(
        {"VPTR_Enc": VPTR_Enc, "VPTR_Dec": VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list
    )

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training(
            {"VPTR_Transformer": VPTR_Transformer},
            {"optimizer_T": optimizer_T},
            resume_ckpt,
            loss_name_list,
        )

    ##################### Inference ################################

    psnr_sol_t = 0
    psnr_w_t = 0
    ssim_sol_t = 0
    ssim_w_t = 0
    # lpips_sol_t = 0
    # lpips_w_t = 0
    for idx, sample in enumerate(test_loader):
        (
            b_psnr_sol_t,
            b_psnr_w_t,
            b_ssim_sol_t,
            b_ssim_w_t,
            # b_lpips_sol_t,
            # b_lpips_w_t,
        ) = FAR_show_sample(
            idx,
            VPTR_Enc,
            VPTR_Dec,
            VPTR_Transformer,
            num_future_frames,
            sample,
            ckpt_save_dir.joinpath("inference_gifs"),
            test_phase=True,
            device=device,
        )
        psnr_sol_t += b_psnr_sol_t / len(test_loader)
        psnr_w_t += b_psnr_w_t / len(test_loader)
        ssim_sol_t += b_ssim_sol_t / len(test_loader)
        ssim_w_t += b_ssim_w_t / len(test_loader)
        # lpips_sol_t += b_lpips_sol_t / len(test_loader)
        # lpips_w_t += b_lpips_w_t / len(test_loader)
        print(f"{idx}/{len(test_loader)}")
        # break

    columns = [
        "t",
        "psnr_sol",
        "psnr_w",
        "ssim_sol",
        "ssim_w",
        # "lpips_sol",
        # "lpips_w",
    ]

    data = np.asarray(
        [
            np.arange(len(psnr_sol_t)),
            psnr_sol_t.numpy(),
            psnr_w_t.numpy(),
            ssim_sol_t.numpy(),
            ssim_w_t.numpy(),
            # lpips_sol_t.numpy(),
            # lpips_w_t.numpy(),
        ]
    ).T

    df = pd.DataFrame(columns=columns, data=data)
    df.to_csv(ckpt_save_dir.joinpath("FAR_test_metrics.csv"), index=False)
