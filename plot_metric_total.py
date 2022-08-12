import os
import matplotlib.pyplot as plt
import pandas as pd

ckpt = "VPTR_ckpts"
TOTAL_AE = os.path.join(
    ckpt,
    "0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt/AE_test_metrics_total.csv",
)
V100_AE = os.path.join(
    ckpt,
    "0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt/AE_test_metrics_V100.csv",
)
V1000_AE = os.path.join(
    ckpt,
    "0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt/AE_test_metrics_V1000.csv",
)
V10000_AE = os.path.join(
    ckpt,
    "0811_NSBDTotalMinMax_1to4_40_ResNetAE_MSEGDLlsgan_ckpt/AE_test_metrics_V10000.csv",
)

f_list = [TOTAL_AE, V100_AE, V1000_AE, V10000_AE]

label_list = ["w", "f", "w0", "ux", "uy", "uxy", "v"]
fig, ax = plt.subplots(ncols=2, nrows=7)
fig.set_size_inches(12, 5)
plt.subplots_adjust(
    left=0.1, bottom=None, right=0.8, top=None, wspace=None, hspace=None
)
lines = ["solid", "dashed", "dotted", "dashdot"]
colors = ["k", "r", "g", "b", "m", "y", "c"]
for i, metric in enumerate(("psnr", "ssim")):
    for idx, f in enumerate(f_list):
        df = pd.read_csv(f)
        for j in range(4):
            ax[j, i].plot(
                df.t,
                df[f"{metric}_{j}"],
                label=label_list[j],
                linestyle=lines[idx],
                color=colors[j],
            )

for i, metric in enumerate(("psnr", "ssim")):
    for j in range(7):
        ax[j, i].set_xticks([0, 10, 20, 30, 40, 50])
        ax[j, i].grid()
        if metric == "ssim":
            ax[j, i].set_ylim([0.3, 1.0])
        else:
            ax[j, i].set_ylim([0, 100])
ax[0, 0].set_title("PSNR")
ax[0, 1].set_title("SSIM")
ax[0, 1].text(60.0, 0.05, "\n".join(label_list))

plt.savefig("metric_AE_1to4_minmax.png")
