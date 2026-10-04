# -*- coding: utf-8 -*-
r"""
02 多层感知机 MLP · TensorFlow 版 —— 同一道题，第二个框架的层封装
==================================================================
任务: 两月牙二分类 —— 线性回归画一条直线分不开的两个月牙,
      中间夹一个 tanh 隐层就能分开。这就是"神经网络"三个字的全部含义:
      线性变换 -> 非线性 -> 再线性变换, 深度产生了非线性边界。

框架替你做了什么（与数学公式逐条对应）：
    数学 / 手搓                         TensorFlow / Keras
    -------------------------------    --------------------------------------
    参数 W1,b1,W2,b2（四个数组）      keras.layers.Dense(32) + Dense(2)（封装成层；
                                       注意：Keras 的 kernel 形状就是 (入,出)，与手写 W 一致，
                                       不像 PyTorch 内部存转置）
    前向  z1=X@W1+b1 ... softmax      model(x) 一层层走；softmax 打包进损失
    损失  -mean(log P[正确类])        keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                                       （= 手写的减 max + softmax + 负对数似然，数值稳定内部做了）
    梯度  手推 6 条链式法则           tape.gradient(loss, model.trainable_variables)
                                       ← 种子 (P − onehot)/N 藏在交叉熵的反向里
    更新  手写 Adam（m/v 两本账）     keras.optimizers.Adam  ← 同一套公式，免写
    质检  数值梯度检查                照做！tape 的梯度 vs 差分（相对误差 <1e-6）

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 02_mlp\02_mlp_tensorflow.py
预期:  梯度检查 < 1e-6;  训练后准确率 > 95%（与 PyTorch 版同一份数据、同一初始权重）
产物:  本文件夹 figs_tensorflow/ 下生成 4 张参考图（与 PyTorch 版同名，可并排对照）
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

rng = np.random.default_rng(0)

# ---------- 造数据: 两个交错的半圆（与 PyTorch 版逐位相同：同一 numpy 种子） ----------
def make_moons(n=400, noise=0.15):
    t = rng.uniform(0, np.pi, n // 2)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    Y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return X, Y

X, Y = make_moons()
X = X - X.mean(0)                       # 中心化, 让梯度下降好走
N, DX = X.shape
H = 32

# ---------- 初始权重：按手搓版的随机流顺序抽，各版同起点（教学对照用） ----------
W1 = rng.normal(0, 0.5, (DX, H))        # 注意抽取顺序：先 W1 后 W2，不能反
W2 = rng.normal(0, 0.5, (H, 2))

X_t = tf.constant(X, dtype=tf.float64)
Y_t = tf.constant(Y, dtype=tf.int32)

model = keras.Sequential([
    keras.Input(shape=(DX,)),
    keras.layers.Dense(H, activation="tanh", dtype="float64"),
    keras.layers.Dense(2, dtype="float64"),
])
model(X_t[:1])                          # 跑一个样本把层"建"出来，才能 set_weights
model.layers[0].set_weights([W1, np.zeros(H)])   # Keras kernel 形状 (入,出)，与 W1 直接对上
model.layers[1].set_weights([W2, np.zeros(2)])

def compute_loss():
    """交叉熵 = softmax + 负对数似然（等价于 PyTorch 的 F.cross_entropy）。

    这里用底层算子而不是 keras.losses.SparseCategoricalCrossentropy：
    Keras 的损失类默认把计算降到 float32（dtype 策略），微小梯度在数值差分里会被
    量化成 0，梯度检查过不了；底层算子按输入 dtype（float64）原样算，精度够。
    """
    logits = model(X_t, training=True)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_t, logits=logits))


def accuracy():
    logits = model(X_t, training=False)
    correct = tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == Y_t, tf.float64)
    return float(tf.reduce_mean(correct))


# ---------- 数值梯度检查: 质检员上岗（tape 的梯度 vs 差分） ----------
with tf.GradientTape() as tape:
    loss = compute_loss()
grads = tape.gradient(loss, model.trainable_variables)   # ← 6 步链式法则，自动走完

eps = 1e-5
worst = 0.0
names = ["W1", "b1", "W2", "b2"]        # trainable_variables 的顺序 = 层的顺序
for name, var, g in zip(names, model.trainable_variables, grads):
    flat = tf.reshape(var, [-1]).numpy().copy()
    gflat = tf.reshape(g, [-1]).numpy()
    n = flat.size
    for j in [0, n // 2, n - 1]:         # 每个参数张量抽 3 个元素查
        old = float(flat[j])
        flat[j] = old + eps
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        lp = float(compute_loss())
        flat[j] = old - eps
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        lm = float(compute_loss())
        flat[j] = old                    # 恢复
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        num = (lp - lm) / (2 * eps)
        ana = float(gflat[j])
        rel = abs(ana - num) / max(abs(ana), abs(num), 1e-8)
        worst = max(worst, rel)
print("grad check  max relative error = %.2e" % worst)
assert worst < 1e-6, "tape 的梯度错了?!"
print("grad check passed\n")


# ---------- Adam(论文 5.3 节同款——框架一行搞定) ----------
opt = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
hist = []                               # (loss, accuracy), 画图用
for step in range(1, 1501):
    with tf.GradientTape() as tape:
        loss = compute_loss()
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    hist.append((float(loss), accuracy()))
    if step % 250 == 0 or step == 1:
        print("step %4d  loss %.4f  accuracy %.1f%%" % (step, float(loss), 100 * accuracy()))

print("\nfinal accuracy: %.1f%%   (线性模型的极限约 85%%, 隐层把边界掰弯了)" % (100 * accuracy()))


# ================= 参考图 =================
def save_figures(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_tensorflow")
    os.makedirs(figdir, exist_ok=True)

    # ---- 图1: 月牙数据 ----
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=14, color="#4878b0", label="类 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=14, color="#d65f5f", label="类 1")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("两月牙：一条直线分不开（去问问 01 的线性回归）")
    ax.legend(); ax.set_aspect("equal")
    fig.savefig(os.path.join(figdir, "fig1_moons_data.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 决策边界: MLP(弯) vs 线性基线(直) ----
    Wl = rng.normal(0, 0.5, (2, 2))     # 与手搓版同随机流位置抽的初始值
    lin = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(2, dtype="float64"),
    ])
    lin(X_t[:1])
    lin.layers[0].set_weights([Wl, np.zeros(2)])
    optl = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    for _ in range(400):
        with tf.GradientTape() as tape:
            lloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=Y_t, logits=lin(X_t, training=True)))
        optl.apply_gradients(zip(tape.gradient(lloss, lin.trainable_variables),
                                 lin.trainable_variables))

    gx = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 300)
    gy = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 300)
    XX, YY = np.meshgrid(gx, gy)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)
    G_t = tf.constant(G, dtype=tf.float64)

    def prob_mlp(Gg):                    # 属于类1的概率
        return tf.nn.softmax(model(Gg, training=False), axis=1)[:, 1].numpy()

    def prob_lin(Gg):
        return tf.nn.softmax(lin(Gg, training=False), axis=1)[:, 1].numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.contourf(XX, YY, prob_mlp(G_t).reshape(XX.shape), levels=[0, 0.5, 1],
                colors=["#aec7e8", "#ff9896"], alpha=0.35)
    ax.contour(XX, YY, prob_mlp(G_t).reshape(XX.shape), levels=[0.5],
               colors="crimson", linewidths=2.2)
    ax.contour(XX, YY, prob_lin(G_t).reshape(XX.shape), levels=[0.5],
               colors="gray", linestyles="--", linewidths=2.0)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=12, color="#4878b0", label="类 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=12, color="#d65f5f", label="类 1")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="crimson", lw=2.2, label="MLP 边界（弯的）"),
               Line2D([], [], color="gray", ls="--", lw=2.0, label="线性基线边界（直的）")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], loc="upper left", fontsize=9)
    ax.set_title("决策边界：隐层把直线掰弯了（TensorFlow）")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(figdir, "fig2_decision_boundary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 网络结构 + 反向传播链（红段是 GradientTape 替你走的） ----
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def node(cx, cy, r=0.13, fc="#ffffff"):
        ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec="#444444", lw=1.1))

    def arr(p1, p2, color="#3b6fd4", lw=1.1, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    ins = [(1.2, 2.3), (1.2, 3.3)]
    hids = [(4.2, yy) for yy in np.linspace(0.9, 4.3, 6)]
    outs = [(7.2, 2.55), (7.2, 3.05)]
    for ix, iy in ins:
        node(ix, iy, fc="#dbe9ff")
    for hx, hy in hids:
        node(hx, hy, fc="#fff3d6")
    for ox, oy in outs:
        node(ox, oy, fc="#ffd9d2")
    ax.text(1.2, 0.35, "输入 x1,x2", ha="center", fontsize=10)
    ax.text(4.2, 0.35, "隐层 tanh（共32个，画出6个）", ha="center", fontsize=10)
    ax.text(7.2, 0.35, "输出 softmax", ha="center", fontsize=10)
    for ix, iy in ins:
        for hx, hy in hids:
            arr((ix + 0.15, iy), (hx - 0.15, hy))
    for hx, hy in hids:
        for ox, oy in outs:
            arr((hx + 0.15, hy), (ox - 0.15, oy))
    ax.text(8.6, 4.6, "GradientTape 记录 + tape.gradient 自动走完（红段）:", fontsize=10,
            color="#d4402e", va="top")
    ax.text(8.6, 4.0,
            "种子  dlogits = (P − y)/N\n"
            "→ dW2 = h^T · dlogits\n"
            "→ dh  = dlogits · W2^T\n"
            "→ dz1 = dh⊙(1 − h²)   ← tanh′\n"
            "→ dW1 = X^T · dz1\n"
            "（公式还是这套，只是不用手写）",
            fontsize=9, color="#d4402e", va="top", linespacing=1.5)
    arr((7.2, 0.85), (1.2, 0.85), "#d4402e", lw=2, ls="--")
    ax.set_title("MLP（TensorFlow）：前向（蓝）一层层算，反向（红）记录带一层层退", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig3_network_backprop.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: loss / accuracy 曲线 ----
    steps = np.arange(1, len(hist) + 1)
    losses = [h[0] for h in hist]
    accs = [h[1] for h in hist]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(steps, losses, color="#c44e52", label="loss（左轴）")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss", color="#c44e52")
    ax2 = ax.twinx()
    ax2.plot(steps, accs, color="#2e8b57", label="accuracy（右轴）")
    ax2.set_ylabel("accuracy", color="#2e8b57")
    ax.set_title("训练曲线（TensorFlow）：loss 降、accuracy 升")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig4_loss_acc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


save_figures(hist)
