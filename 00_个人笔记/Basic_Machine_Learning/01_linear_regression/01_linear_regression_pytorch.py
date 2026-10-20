# -*- coding: utf-8 -*-
r"""
01 线性回归 · PyTorch 版 —— 手搓版换成框架，一行一行都有对应
================================================================
模型:  y_hat = w*x + b
损失:  L = mean( 1/2*(y_hat - y)^2 )          乘 1/2 是为了让导数干净

框架替你做了什么（与数学公式逐条对应）：
    数学 / 手搓                         PyTorch
    -------------------------------    --------------------------------------
    参数 w, b（两个浮点数）            nn.Linear(1, 1)（内部就是 w、b 两个参数）
    前向  y_hat = w*x + b              model(x)（调用 __call__ 走前向）
    损失  手写 0.5*mean((y_hat-y)^2)   0.5 * nn.MSELoss()（PyTorch 的 MSELoss 少个 0.5 因子）
    梯度  手推 dL/dw、dL/db            loss.backward()  ← autograd 自动做链式法则
                                        （反向传播的种子 dL/dy_hat = y_hat - y
                                          就藏在 MSE 的反向实现里，不用再手推）
    更新  w -= lr*gw                   optimizer.step()
    质检  数值梯度检查                 照做！用 autograd 的梯度 vs 差分对照

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_pytorch.py
预期:  梯度检查误差 < 1e-6;  w, b 收敛到 3.0 和 1.0（与 TensorFlow 版同一份数据、同一随机种子）
产物:  本文件夹 figs_pytorch/ 下生成 4 张参考图:
       fig1 数据与拟合线 | fig2 loss曲线 | fig3 损失面等高线+GD轨迹 | fig4 反向传播计算图
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ---------- 造数据: y = 3x + 1 + 噪声（与 TF 版逐位相同：同一 numpy 种子） ----------
rng = np.random.default_rng(0)
N = 200
x = rng.uniform(-2, 2, N)
y = 3.0 * x + 1.0 + rng.normal(0, 0.3, N)

# 演示用的题很小，float64 让数值梯度检查能卡到 1e-6 级别（工程里默认 float32）
x_t = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
y_t = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)

# ---------- 模型与损失 ----------
model = nn.Linear(1, 1).double()   # 一个线性层 = w*x + b；.double() 与 float64 数据对齐
with torch.no_grad():              # 起点定为 (0, 0)：与 TensorFlow 版同起点，轨迹才能并排对照
    model.weight.zero_()           # （工程里一般用默认随机初始化；这里是为教学对照）
    model.bias.zero_()
loss_fn = nn.MSELoss()             # 均值平方误差（注意：PyTorch 版本不带 1/2）
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


def compute_loss():
    """0.5 * MSE —— 补上 1/2 与人手搓的公式完全一致（对梯度来说只差个常数因子）。"""
    return 0.5 * loss_fn(model(x_t), y_t)


# ---------- 数值梯度检查: 质检员上岗（autograd vs 差分） ----------
print("grad check: autograd vs numeric")
loss = compute_loss()
loss.backward()                                   # ← autograd 一趟反向，梯度进 .grad
gw_auto = model.weight.grad.item()
gb_auto = model.bias.grad.item()

eps = 1e-6                                        # float64 上差分可以很小
with torch.no_grad():                             # 差分采样时不要污染 autograd 图
    model.weight += eps; Lp = compute_loss().item()
    model.weight -= 2 * eps; Lm = compute_loss().item()
    model.weight += eps                           # 恢复
    model.bias += eps; Lp_b = compute_loss().item()
    model.bias -= 2 * eps; Lm_b = compute_loss().item()
    model.bias += eps
gw_num = (Lp - Lm) / (2 * eps)
gb_num = (Lp_b - Lm_b) / (2 * eps)
print("grad check  dw: autograd %.8f | numeric %.8f" % (gw_auto, gw_num))
print("grad check  db: autograd %.8f | numeric %.8f" % (gb_auto, gb_num))
assert abs(gw_auto - gw_num) < 1e-6, "autograd 的梯度错了?!"
assert abs(gb_auto - gb_num) < 1e-6, "autograd 的梯度错了?!"
print("grad check passed\n")

# ---------- 梯度下降（顺手记录轨迹，画图用） ----------
hist = []                          # 每步的 (w, b, loss)
for step in range(1, 301):
    optimizer.zero_grad()          # 清掉上一步的 .grad（梯度会累加，不清会滚雪球）
    loss = compute_loss()
    loss.backward()                # 反向传播：autograd 按计算图逐节点求导
    optimizer.step()               # 用 .grad 更新参数：w -= lr*gw
    w = model.weight.item()
    b = model.bias.item()
    hist.append((w, b, loss.item()))
    if step % 50 == 0 or step == 1:
        print("step %3d  loss %.4f  w %.4f  b %.4f" % (step, loss.item(), w, b))

w = model.weight.item()
b = model.bias.item()
print("\nfinal:  w = %.4f (true 3.0)   b = %.4f (true 1.0)" % (w, b))
print("注意: loss 不会降到 0 —— 数据自带 0.3 的噪声, 模型只能学到趋势, 吃不掉噪声。")


# ================= 参考图 =================
def save_figures(w, b, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_pytorch")
    os.makedirs(figdir, exist_ok=True)

    # ---- 图1: 数据与拟合 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y, s=14, alpha=0.45, color="#4878b0", label="数据（含噪声）")
    xs = np.linspace(-2.2, 2.2, 50)
    ax.plot(xs, 3.0 * xs + 1.0, "g--", lw=2, label="真实规律  y = 3x + 1")
    ax.plot(xs, w * xs + b, "r-", lw=2, label="PyTorch 拟合  ŷ = %.2fx + %.2f" % (w, b))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("线性回归（PyTorch）：模型学到的就是这条直线")
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
    ax.set_title("loss 曲线（PyTorch）：每一步都沿负梯度下山")
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
    ax.set_title("损失面等高线：backward() 告诉我们往哪边下山")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(figdir, "fig3_loss_surface_gd.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 反向传播计算图（蓝=前向，红=autograd 自动完成） ----
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
    rbox(2.3, 2.0, 2.6, 1.15, "nn.Linear(1,1)\nŷ = w·x + b", "#dbe9ff")
    rbox(5.9, 2.0, 2.6, 1.15, "nn.MSELoss\nL = ½(ŷ − y)²", "#ffd9d2")
    ax.text(8.6, 2.0, "L（一个标量）", fontsize=11, ha="center", va="center")
    arr((0.62, 2.0), (0.95, 2.0), "#3b6fd4")
    arr((3.62, 2.0), (4.55, 2.0), "#3b6fd4")
    arr((7.22, 2.0), (8.0, 2.0), "#3b6fd4")
    ax.text(2.3, 3.55, "参数 w, b（要学的东西）", fontsize=10, ha="center", color="#3b6fd4")
    arr((1.85, 3.3), (1.85, 2.62), "#3b6fd4")
    arr((2.75, 3.3), (2.75, 2.62), "#3b6fd4")
    arr((5.9, 1.15), (2.3, 1.15), "#d4402e", lw=2)
    ax.text(4.1, 1.32, "loss.backward()：autograd 自动完成整条链", fontsize=10,
            ha="center", color="#d4402e")
    arr((2.3, 0.85), (2.3, 0.45), "#d4402e", lw=2)
    ax.text(2.4, 0.28, "梯度落进 w.grad / b.grad，optimizer.step() 用它更新", fontsize=10,
            ha="center", color="#d4402e")
    ax.set_title("PyTorch 的反向传播：前向（蓝）手写，反向（红）交给 autograd", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig4_backprop_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


save_figures(w, b, hist)
