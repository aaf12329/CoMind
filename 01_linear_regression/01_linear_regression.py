# -*- coding: utf-8 -*-
r"""
01 线性回归 —— 一切从这里开始
==============================
模型:  y_hat = w*x + b
损失:  L = mean( 1/2*(y_hat - y)^2 )          乘 1/2 是为了让导数干净
手推梯度(反向传播的全过程, 就这么多):
    dL/d_yhat = (y_hat - y)        <- 反向传播的"种子", 整个系列里它反复出现
    dL/d_w    = mean( (y_hat - y) * x )
    dL/d_b    = mean( (y_hat - y) )
数值梯度检查:
    用 (L(w+e)-L(w-e)) / 2e 直接对损失做差分, 和手推结果对照。
    手推错了当场现形 —— 这套质检流程后面每个模型都会用。

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 01_linear_regression\01_linear_regression.py
预期:  梯度检查误差 < 1e-6;  w, b 收敛到 3.0 和 1.0
产物:  本文件夹 figs/ 下生成 4 张参考图:
       fig1 数据与拟合线 | fig2 loss曲线 | fig3 损失面等高线+GD轨迹 | fig4 反向传播计算图
"""
import os
import numpy as np

rng = np.random.default_rng(0)

# ---------- 造数据: y = 3x + 1 + 噪声 ----------
N = 200
x = rng.uniform(-2, 2, N)
y = 3.0 * x + 1.0 + rng.normal(0, 0.3, N)

w, b = 0.0, 0.0
lr = 0.1


def loss(w, b):
    return 0.5 * np.mean((w * x + b - y) ** 2)


def grads(w, b):
    """手推的解析梯度。误差项 e = y_hat - y 就是反向传播的种子。"""
    e = w * x + b - y
    return np.mean(e * x), np.mean(e)


# ---------- 数值梯度检查: 质检员上岗 ----------
eps = 1e-5
num_dw = (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps)
num_db = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
gw, gb = grads(w, b)
print("grad check  dw: manual %.8f | numeric %.8f" % (gw, num_dw))
print("grad check  db: manual %.8f | numeric %.8f" % (gb, num_db))
assert abs(gw - num_dw) < 1e-6, "手推的梯度错了!"
print("grad check passed\n")

# ---------- 梯度下降(顺手记录轨迹, 画图用) ----------
hist = []                       # 每步的 (w, b, loss)
for step in range(1, 301):
    gw, gb = grads(w, b)
    w -= lr * gw
    b -= lr * gb
    hist.append((w, b, loss(w, b)))
    if step % 50 == 0 or step == 1:
        print("step %3d  loss %.4f  w %.4f  b %.4f" % (step, loss(w, b), w, b))

print("\nfinal:  w = %.4f (true 3.0)   b = %.4f (true 1.0)" % (w, b))
print("注意: loss 不会降到 0 —— 数据自带 0.3 的噪声, 模型只能学到趋势, 吃不掉噪声。")


# ================= 参考图 =================
def save_figures(w, b, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figdir, exist_ok=True)

    # ---- 图1: 数据与拟合 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y, s=14, alpha=0.45, color="#4878b0", label="数据（含噪声）")
    xs = np.linspace(-2.2, 2.2, 50)
    ax.plot(xs, 3.0 * xs + 1.0, "g--", lw=2, label="真实规律  y = 3x + 1")
    ax.plot(xs, w * xs + b, "r-", lw=2, label="模型拟合  ŷ = %.2fx + %.2f" % (w, b))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("线性回归：模型学到的就是这条直线")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(figdir, "fig1_data_vs_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: loss 曲线 ----
    steps = np.arange(1, len(hist) + 1)
    losses = np.array([h[2] for h in hist])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(steps, losses, color="#c44e52", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("loss（对数轴）")
    ax.set_title("loss 曲线：每一步都沿负梯度下山")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig2_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 损失面等高线 + GD 轨迹 ----
    ws = np.linspace(-0.5, 3.6, 220)
    bs = np.linspace(-1.5, 2.5, 220)
    WW, BB = np.meshgrid(ws, bs)
    Z = 0.5 * ((WW[:, :, None] * x + BB[:, :, None] - y) ** 2).mean(-1)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    cs = ax.contourf(WW, BB, np.log10(Z + 1e-8), levels=32, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10(loss)")
    ax.plot([h[0] for h in hist], [h[1] for h in hist], ".-",
            color="#ff9d3b", ms=3, lw=1.2, label="GD 轨迹 (w, b)")
    ax.plot(3.0, 1.0, "g*", ms=16, label="真实值 (3, 1)")
    ax.set_xlabel("参数 w")
    ax.set_ylabel("参数 b")
    ax.set_title("损失面等高线：反向传播告诉我们往哪边下山")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(figdir, "fig3_loss_surface_gd.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 反向传播计算图（示意） ----
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def rbox(cx, cy, bw, bh, text, fc):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.06", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10)

    def arr(p1, p2, color, lw=1.6, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    # 前向（蓝，从左到右）
    ax.text(0.45, 2.0, "x", fontsize=13, ha="center", va="center")
    rbox(2.3, 2.0, 2.6, 1.15, "线性块\nŷ = w·x + b", "#dbe9ff")
    rbox(5.9, 2.0, 2.6, 1.15, "损失块\nL = ½(ŷ − y)²", "#ffd9d2")
    ax.text(8.6, 2.0, "L（一个数）", fontsize=11, ha="center", va="center")
    arr((0.62, 2.0), (0.95, 2.0), "#3b6fd4")
    arr((3.62, 2.0), (4.55, 2.0), "#3b6fd4")
    arr((7.22, 2.0), (8.0, 2.0), "#3b6fd4")
    # 参数（要学的东西）
    ax.text(2.3, 3.55, "参数 w, b（要学的东西）", fontsize=10, ha="center", color="#3b6fd4")
    arr((1.85, 3.3), (1.85, 2.62), "#3b6fd4")
    arr((2.75, 3.3), (2.75, 2.62), "#3b6fd4")
    # 反向（红，从右到左）：反向传播从损失开始
    arr((5.9, 1.15), (2.3, 1.15), "#d4402e", lw=2)
    ax.text(4.1, 1.32, "反向的起点：种子  ∂L/∂ŷ = ŷ − y", fontsize=10,
            ha="center", color="#d4402e")
    arr((2.3, 0.85), (2.3, 0.45), "#d4402e", lw=2)
    ax.text(2.4, 0.28, "∂L/∂w = (ŷ−y)·x,   ∂L/∂b = (ŷ−y)", fontsize=10,
            ha="center", color="#d4402e")
    ax.set_title("反向传播：前向（蓝）算损失，反向（红）从损失出发把责任分摊到参数",
                 fontsize=12)
    fig.savefig(os.path.join(figdir, "fig4_backprop_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


save_figures(w, b, hist)
