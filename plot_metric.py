import os
import matplotlib.pyplot as plt
import pandas as pd

ckpt = "VPTR_ckpts"
FAR_Euler = os.path.join(ckpt, "0729_NSBDField_40_FAR_MSEGDL_ckpt/FAR_test_metrics.csv")
FAR_RK2 = os.path.join(
    ckpt, "0805_NSBDField_RK2_40_FAR_MSEGDL_ckpt/FAR_test_metrics.csv"
)
FAR_RK4 = os.path.join(
    ckpt, "0805_NSBDField_RK4_40_FAR_MSEGDL_ckpt/FAR_test_metrics.csv"
)
NAR_Euler = os.path.join(ckpt, "0730_NSBDField_40_NAR_MSEGDL_ckpt/NAR_test_metrics.csv")
NAR_RK2 = os.path.join(
    ckpt, "0804_NSBDField_RK2_40_NAR_MSEGDL_ckpt/NAR_test_metrics.csv"
)
NAR_RK4 = os.path.join(
    ckpt, "0804_NSBDField_RK4_40_NAR_MSEGDL_ckpt/NAR_test_metrics.csv"
)
NAR_Euler_V100 = os.path.join(
    ckpt, "0730_NSBDField_40_NAR_MSEGDL_ckpt/NAR_test_metrics_V100.csv"
)
NAR_Euler_V1000 = os.path.join(
    ckpt, "0730_NSBDField_40_NAR_MSEGDL_ckpt/NAR_test_metrics_V1000.csv"
)
NAR_Euler_V10000 = os.path.join(
    ckpt, "0730_NSBDField_40_NAR_MSEGDL_ckpt/NAR_test_metrics_V10000.csv"
)

# t,psnr_sol,psnr_w,ssim_sol,ssim_w
Flist = [
    FAR_Euler,
    FAR_RK2,
    FAR_RK4,
    NAR_Euler,
    NAR_RK2,
    NAR_RK4,
    NAR_Euler_V100,
    NAR_Euler_V1000,
    NAR_Euler_V10000,
]
Llist = [
    "FAR_Euler",
    "FAR_RK2",
    "FAR_RK4",
    "NAR_Euler",
    "NAR_RK2",
    "NAR_RK4",
    "NAR_Euler_V100",
    "NAR_Euler_V1000",
    "NAR_Euler_V10000",
]
Clist = ["r", "r", "r", "b", "b", "b", "g", "g", "g"]
Dlist = [
    "solid",
    "dashed",
    "dotted",
    "solid",
    "dashed",
    "dotted",
    "solid",
    "dashed",
    "dotted",
]
fig, ax = plt.subplots(ncols=2, nrows=1)
fig.set_size_inches(12, 5)
plt.subplots_adjust(
    left=0.1, bottom=None, right=0.8, top=None, wspace=None, hspace=None
)
for f, l, c, d in zip(Flist, Llist, Clist, Dlist):
    df = pd.read_csv(f)
    ax[0].plot(df.t, df.psnr_sol, color=c, linestyle=d, label=l)
    ax[1].plot(df.t, df.ssim_sol, color=c, linestyle=d, label=l)
ax[0].set_xticks([0, 10, 20, 30, 40, 50])
ax[1].set_xticks([0, 10, 20, 30, 40, 50])
ax[0].grid()
ax[1].grid()
ax[0].set_title("PSNR")
ax[1].set_title("SSIM")
ax[1].legend(bbox_to_anchor=(1.1, 1.05))
plt.savefig("metric.png")
