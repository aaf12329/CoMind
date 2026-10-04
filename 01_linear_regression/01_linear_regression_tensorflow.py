# -*- coding: utf-8 -*-
r"""
01 线性回归 · TensorFlow 版 —— 同一道题，换第二个框架
================================================================
模型:  y_hat = w*x + b
损失:  L = mean( 1/2*(y_hat - y)^2 )          乘 1/2 是为了让导数干净

框架替你做了什么（与数学公式逐条对应）：
    数学 / 手搓                         TensorFlow
    -------------------------------    --------------------------------------
    参数 w, b（两个浮点数）            tf.Variable(0.0)（可训练变量）
    前向  y_hat = w*x + b              w * x + b（写在 GradientTape 上下文里）
    损失  手写 0.5*mean((y_hat-y)^2)   0.5 * tf.reduce_mean(...)
    梯度  手推 dL/dw、dL/db            tape.gradient(loss, [w, b])  ← 记录带自动微分
    更新  w -= lr*gw                   optimizer.apply_gradients([(gw,w),(gb,b)])
    质检  数值梯度检查                 照做！tape 的梯度 vs 差分对照

与 PyTorch 版的一个机制差异（记住这个就懂两大框架的脾气）：
    PyTorch 的 .grad 会累加，所以每轮必须先 optimizer.zero_grad()；
    TensorFlow 的 tape 每次 with 都新建一张记录带，天然不累加，没有清零这一步。

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_tensorflow.py
预期:  梯度检查误差 < 1e-6;  w, b 收敛到 3.0 和 1.0（与 PyTorch 版同一份数据、同一随机种子）
产物:  本文件夹 figs_tensorflow/ 下生成 4 张参考图（与 PyTorch 版同名，方便并排对照）
备注:  Windows 原生 TensorFlow 只有 CPU 版（GPU 版仅 WSL2）——这些题很小，CPU 足够。
"""
import os
import numpy as np
import tensorflow as tf

# ---------- 造数据: y = 3x + 1 + 噪声（与 PyTorch 版逐位相同：同一 numpy 种子） ----------
rng = np.random.default_rng(0)
N = 200
x = rng.uniform(-2, 2, N)
y = 3.0 * x + 1.0 + rng.normal(0, 0.3, N)

# 演示用的题很小，float64 让数值梯度检查能卡到 1e-6 级别（工程里默认 float32）
x_t = tf.constant(x, dtype=tf.float64)
y_t = tf.constant(y, dtype=tf.float64)

# ---------- 参数与损失 ----------
w = tf.Variable(0.0, dtype=tf.float64)   # 起点 (0, 0)：与 PyTorch 版同起点，轨迹才能并排对照
b = tf.Variable(0.0, dtype=tf.float64)
lr = 0.1
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)


def compute_loss():
    """0.5 * MSE —— 与手搓公式完全一致（PyTorch 版同样补了这个 1/2）。"""
    y_hat = w * x_t + b
    return 0.5 * tf.reduce_mean((y_hat - y_t) ** 2)


# ---------- 数值梯度检查: 质检员上岗（tape 的梯度 vs 差分） ----------
print("grad check: tape.gradient vs numeric")
with tf.GradientTape() as tape:            # ← 记录带：这段上下文里的运算会被记下来
    loss = compute_loss()
gw_auto, gb_auto = tape.gradient(loss, [w, b])   # ← 反向传播一趟，取回两个梯度
gw_auto = float(gw_auto.numpy())
gb_auto = float(gb_auto.numpy())

eps = 1e-6                                  # float64 上差分可以很小
Lp = float(compute_loss().numpy())          # 说明：下面用"临时变量"做差分，不动真参数
w_old, b_old = w.numpy(), b.numpy()
w.assign(w_old + eps); Lp_w = float(compute_loss().numpy())
w.assign(w_old - eps); Lm_w = float(compute_loss().numpy())
w.assign(w_old)
b.assign(b_old + eps); Lp_b = float(compute_loss().numpy())
b.assign(b_old - eps); Lm_b = float(compute_loss().numpy())
b.assign(b_old)
gw_num = (Lp_w - Lm_w) / (2 * eps)
gb_num = (Lp_b - Lm_b) / (2 * eps)
print("grad check  dw: tape %.8f | numeric %.8f" % (gw_auto, gw_num))
print("grad check  db: tape %.8f | numeric %.8f" % (gb_auto, gb_num))
assert abs(gw_auto - gw_num) < 1e-6, "tape 的梯度错了?!"
assert abs(gb_auto - gb_num) < 1e-6, "tape 的梯度错了?!"
print("grad check passed\n")

# ---------- 梯度下降（顺手记录轨迹，画图用） ----------
hist = []                          # 每步的 (w, b, loss)
for step in range(1, 301):
    with tf.GradientTape() as tape:    # 每轮新建记录带（不像 PyTorch 需要先清零）
        loss = compute_loss()
    grads = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(grads, [w, b]))   # 更新：w -= lr*gw
    wv, bv = float(w.numpy()), float(b.numpy())
    hist.append((wv, bv, float(loss.numpy())))
    if step % 50 == 0 or step == 1:
        print("step %3d  loss %.4f  w %.4f  b %.4f" % (step, float(loss.numpy()), wv, bv))

wv, bv = float(w.numpy()), float(b.numpy())
print("\nfinal:  w = %.4f (true 3.0)   b = %.4f (true 1.0)" % (wv, bv))
print("注意: loss 不会降到 0 —— 数据自带 0.3 的噪声, 模型只能学到趋势, 吃不掉噪声。")


# ================= 参考图 =================
def save_figures(w, b, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_tensorflow")
    os.makedirs(figdir, exist_ok=True)

    # ---- 图1: 数据与拟合 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y, s=14, alpha=0.45, color="#4878b0", label="数据（含噪声）")
    xs = np.linspace(-2.2, 2.2, 50)
    ax.plot(xs, 3.0 * xs + 1.0, "g--", lw=2, label="真实规律  y = 3x + 1")
    ax.plot(xs, w * xs + b, "r-", lw=2, label="TensorFlow 拟合  ŷ = %.2fx + %.2f" % (w, b))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("线性回归（TensorFlow）：模型学到的就是这条直线")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(figdir, "fig1_data_vs_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: loss 曲线 ----
    steps = np.arange(1, len(hist) + 1)
    losses = np.array([h[2] for h in hist])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(steps, losses, color="#2e8b57", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("loss（对数轴）")
    ax.set_title("loss 曲线（TensorFlow）：每一步都沿负梯度下山")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig2_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 损失面等高线 + GD 轨迹（损失面本身用 numpy 算，与框架无关） ----
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
    ax.set_title("损失面等高线：tape.gradient 告诉我们往哪边下山")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(figdir, "fig3_loss_surface_gd.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 反向传播计算图（蓝=前向，红=GradientTape 自动完成） ----
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

    ax.text(0.45, 2.0, "x", fontsize=13, ha="center", va="center")
    rbox(2.3, 2.0, 2.6, 1.15, "线性块\nŷ = w·x + b", "#dbe9ff")
    rbox(5.9, 2.0, 2.6, 1.15, "损失块\nL = ½(ŷ − y)²", "#ffd9d2")
    ax.text(8.6, 2.0, "L（一个标量）", fontsize=11, ha="center", va="center")
    arr((0.62, 2.0), (0.95, 2.0), "#3b6fd4")
    arr((3.62, 2.0), (4.55, 2.0), "#3b6fd4")
    arr((7.22, 2.0), (8.0, 2.0), "#3b6fd4")
    ax.text(2.3, 3.55, "参数 w, b（要学的东西）", fontsize=10, ha="center", color="#3b6fd4")
    arr((1.85, 3.3), (1.85, 2.62), "#3b6fd4")
    arr((2.75, 3.3), (2.75, 2.62), "#3b6fd4")
    arr((5.9, 1.15), (2.3, 1.15), "#d4402e", lw=2)
    ax.text(4.1, 1.32, "GradientTape 记录 + tape.gradient：整条链自动求导", fontsize=10,
            ha="center", color="#d4402e")
    arr((2.3, 0.85), (2.3, 0.45), "#d4402e", lw=2)
    ax.text(2.4, 0.28, "取回 (gw, gb)，apply_gradients 用它更新", fontsize=10,
            ha="center", color="#d4402e")
    ax.set_title("TensorFlow 的反向传播：前向（蓝）手写，反向（红）交给记录带", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig4_backprop_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


save_figures(wv, bv, hist)
