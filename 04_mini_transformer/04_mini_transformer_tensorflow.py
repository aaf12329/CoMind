# -*- coding: utf-8 -*-
r"""
04 迷你 Transformer · TensorFlow 版 —— 《Attention Is All You Need》最小可训练版
================================================================================
同一个任务(字符级语言模型, 语料同 03), 换成论文的架构:

    token embedding + 正弦位置编码(论文 3.5 节, 连公式都一样)
    x2 个 block:
        x = x + MultiHeadAttention(LN(x))     论文 3.2: 缩放点积注意力 x 4 头
        x = x + FFN(LN(x))                    论文 3.3: 512->2048->512, 这里等比缩小
    最后 Linear -> softmax -> 交叉熵

和论文的三处有意差异(都是现代惯例, 为了手写简单):
    1. pre-LN(归一化放在子层前面): 论文的 post-LN 必须配学习率预热才稳,
       pre-LN 直接训得动 —— 现在的 LLM 几乎全用 pre-LN
    2. 去掉所有偏置和 LN 的缩放平移参数
    3. 模型缩小: D=64, 4头x2层 (论文是 D=512, 8头x6层)

对照 03 最重要的一个观察:
    LSTM 里有个 for t in range(...) 的时间循环, 一步算一帧;
    这里没有时间循环 —— 注意力把整段序列的交互一次矩阵乘算完。
    这就是论文 Table 1 里 Self-Attention 那一行 O(1) 串行操作的代价。

框架替你做了什么（与手搓版逐条对应）：
    手搓版                            TensorFlow / Keras
    ------------------------------    --------------------------------------
    手写 layernorm + 反向              keras.layers.LayerNormalization(center=False,
                                      scale=False) ← "无仿射"就是关掉这两个开关
    权重摆法                          手搓版 Wq 是 (入,出)（前向 xn@Wq），
                                      Keras kernel 同为 (入,出) → 直接灌
    手写的多头注意力（拆分/拼回）      还是手写这几步（教学版），但全部换 tf 张量：
    手推 softmax 反向                  tf.nn.softmax 的反向由 GradientTape 处理
    手推整棵计算图的反向（40 行）      tape.gradient
    Adam 手写 / 裁剪 np.clip          keras.optimizers.Adam / tf.clip_by_value

    （工程里注意力会直接用 keras.layers.MultiHeadAttention——
      本教学版把矩阵乘一步步写开，图和公式才对得上。）

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 04_mini_transformer\04_mini_transformer_tensorflow.py
预期:  梯度检查 < 1e-4(含 ReLU, 阈值放宽);  loss 从 2.56 降到 0.1 以下;  逐字默写
产物:  本文件夹 figs_tensorflow/ 下 4 张参考图（与 PyTorch 版同名，可并排对照）
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

rng = np.random.default_rng(0)

# ---------- 数据(与 03 完全一致, 保证对照公平；也与 PyTorch 版逐位相同) ----------
SENT = "attention is all you need. "
data = SENT * 400
chars = sorted(set(data))
V = len(chars)
stoi = {ch: k for k, ch in enumerate(chars)}
itos = {k: ch for ch, k in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]

D, NH, NL, T = 64, 4, 2, 32      # 模型维度 / 注意力头数 / block 层数 / 上下文长度
DH = D // NH                     # 每个头的维度


def positional_encoding(T, D):
    """论文 3.5 节的正弦位置编码, 公式原样照抄。"""
    pe = np.zeros((T, D))
    pos = np.arange(T)[:, None]
    i2 = np.arange(0, D, 2)[None, :]
    pe[:, 0::2] = np.sin(pos / 10000 ** (i2 / D))
    pe[:, 1::2] = np.cos(pos / 10000 ** (i2 / D))
    return pe


PE = positional_encoding(T, D)
PE_t = tf.constant(PE, dtype=tf.float64)

# ---------- 初始权重：按手搓版的随机流顺序抽（两版同起点） ----------
def init_weights_numpy():
    s = 0.02
    w = {'Emb': rng.normal(0, s, (V, D))}
    for l in range(NL):
        for nm, sh in [('Wq', (D, D)), ('Wk', (D, D)), ('Wv', (D, D)),
                       ('Wo', (D, D)), ('W1', (D, 4 * D)), ('W2', (4 * D, D))]:
            w['%s%d' % (nm, l)] = rng.normal(0, s, sh)
    w['Wout'] = rng.normal(0, s, (D, V))
    return w


NW = init_weights_numpy()

MASK = tf.constant(np.triu(np.ones((T, T)), 1)[None, None, :, :] * (-1e9), dtype=tf.float64)


class Block:
    """pre-LN 的一个 block：LN→注意力→残差；LN→FFN→残差。全部无偏置、LN 无仿射。"""

    def __init__(self, l):
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5, center=False,
                                                   scale=False, dtype="float64")
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5, center=False,
                                                   scale=False, dtype="float64")
        self.q = keras.layers.Dense(D, use_bias=False, dtype="float64")
        self.k = keras.layers.Dense(D, use_bias=False, dtype="float64")
        self.v = keras.layers.Dense(D, use_bias=False, dtype="float64")
        self.o = keras.layers.Dense(D, use_bias=False, dtype="float64")
        self.f1 = keras.layers.Dense(4 * D, use_bias=False, dtype="float64")
        self.f2 = keras.layers.Dense(D, use_bias=False, dtype="float64")

    def set_weights_from(self, l):
        # 手搓版这章的权重本来就是 (入,出)（前向是 xn @ W），与 Keras kernel 同形 → 直接灌。
        # （对照：PyTorch nn.Linear 存 (出,入)，那边才需要 .T——同一个"摆法"问题，两家相反）
        self.q.set_weights([NW['Wq%d' % l]])
        self.k.set_weights([NW['Wk%d' % l]])
        self.v.set_weights([NW['Wv%d' % l]])
        self.o.set_weights([NW['Wo%d' % l]])
        self.f1.set_weights([NW['W1%d' % l]])
        self.f2.set_weights([NW['W2%d' % l]])


class MiniTransformer:
    def __init__(self):
        self.emb = keras.layers.Embedding(V, D, dtype="float64")
        self.blocks = [Block(l) for l in range(NL)]
        self.out = keras.layers.Dense(V, use_bias=False, dtype="float64")
        # 先跑一个样本让所有层建权重，再灌手搓版抽的初始值
        dummy = tf.zeros((1, 8), dtype=tf.int32)
        self.emb(dummy)
        self.emb.set_weights([NW['Emb']])
        xd = self.emb(dummy) + PE_t[:8]
        for blk, l in zip(self.blocks, range(NL)):
            xd, _ = self.forward_block(blk, xd, 8)     # 注意解包 (x, A)
            blk.set_weights_from(l)
        self.out(xd)
        self.out.set_weights([NW['Wout']])          # 同上：直接灌，不转置

    @staticmethod
    def forward_block(blk, x, L):
        """一个 block 的前向（含因果 mask 的多头注意力 + FFN），训练/推理共用。"""
        xn = blk.ln1(x)                                   # pre-LN: 先归一化再进子层
        qh = tf.transpose(tf.reshape(blk.q(xn), (1, L, NH, DH)), (0, 2, 1, 3))  # (1,NH,L,DH)
        kh = tf.transpose(tf.reshape(blk.k(xn), (1, L, NH, DH)), (0, 2, 1, 3))
        vh = tf.transpose(tf.reshape(blk.v(xn), (1, L, NH, DH)), (0, 2, 1, 3))
        scores = qh @ tf.transpose(kh, (0, 1, 3, 2)) / (DH ** 0.5)   # 缩放点积, 论文式(1)
        scores = scores + MASK[:, :, :L, :L]              # 因果mask: 未来=负无穷
        A = tf.nn.softmax(scores, axis=-1)                # 每行和为1的注意力权重
        ah = A @ vh                                       # 按权重加权 V
        concat = tf.reshape(tf.transpose(ah, (0, 2, 1, 3)), (1, L, D))
        x = x + blk.o(concat)                             # 残差连接(论文 3.1)
        x = x + blk.f2(tf.nn.relu(blk.f1(blk.ln2(x))))    # FFN + 残差(论文 3.3)
        return x, A

    def forward(self, xs, need_attn=False):
        L = len(xs)
        x = self.emb(tf.constant([xs], dtype=tf.int32)) + PE_t[:L]   # 词义 + 位置(论文 3.5)
        attns = []
        for blk in self.blocks:
            x, A = self.forward_block(blk, x, L)
            if need_attn:
                attns.append(A)
        logits = self.out(x)
        return (logits, attns) if need_attn else logits

    @property
    def trainable_variables(self):
        """固定顺序：[Emb, (每层 q,k,v,o,f1,f2 的 kernel)×NL, Wout]。"""
        vs = [self.emb.embeddings]
        for blk in self.blocks:
            for layer in (blk.q, blk.k, blk.v, blk.o, blk.f1, blk.f2):
                vs.append(layer.kernel)
        vs.append(self.out.kernel)
        return vs


model = MiniTransformer()
VAR_NAMES = ["Emb"] + ["%s%d" % (nm, l) for l in range(NL)
                       for nm in ("Wq", "Wk", "Wv", "Wo", "W1", "W2")] + ["Wout"]


def loss_on(xs, ys):
    logits = model.forward(xs)                            # (1, L, V)
    logits = tf.reshape(logits, (-1, V))
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.constant(ys, dtype=tf.int32), logits=logits))


# ---------- 数值梯度检查: 质检员上岗（tape 的梯度 vs 差分） ----------
GC_S = 100                                            # 固定一段语料，两版用同一段
chunk = encode(data[GC_S:GC_S + T + 1])
gc_xs, gc_ys = chunk[:-1], chunk[1:]
with tf.GradientTape() as tape:
    loss = loss_on(gc_xs, gc_ys)
grads = tape.gradient(loss, model.trainable_variables)

eps = 1e-5
worst = 0.0
for name, var, g in zip(VAR_NAMES, model.trainable_variables, grads):
    flat = tf.reshape(var, [-1]).numpy().copy()
    gflat = tf.reshape(g, [-1]).numpy()
    n = flat.size
    for j in [0, n // 2, n - 1]:                      # 每个参数张量抽 3 个元素查
        old = float(flat[j])
        flat[j] = old + eps
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        lp = float(loss_on(gc_xs, gc_ys))
        flat[j] = old - eps
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        lm = float(loss_on(gc_xs, gc_ys))
        flat[j] = old                                 # 恢复
        var.assign(tf.reshape(tf.constant(flat, dtype=var.dtype), var.shape))
        num = (lp - lm) / (2 * eps)
        ana = float(gflat[j])
        scaled = abs(ana - num) / max(1.0, abs(ana), abs(num))    # 大梯度看相对, 小梯度看绝对
        worst = max(worst, scaled)
print("grad check  max scaled error = %.2e" % worst)
assert worst < 1e-4, "tape 反向推错了?!"            # 含 ReLU，阈值按惯例放宽
print("grad check passed\n")


# ---------- 训练(和 03 同一套流程) ----------
def train(iters=4000):
    opt = keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    print("V=%d, 瞎猜下限 ln(V) = %.3f" % (V, np.log(V)))
    losses = []
    for it in range(1, iters + 1):
        s = rng.integers(0, len(data) - T - 1)        # 与 PyTorch 版同一随机流抽起点
        chunk = encode(data[s:s + T + 1])
        xs, ys = chunk[:-1], chunk[1:]
        with tf.GradientTape() as tape:
            loss = loss_on(xs, ys)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_value(g, -5.0, 5.0) for g in grads]   # 与手搓版同款逐元素裁剪
        opt.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss))
        if it % 250 == 0 or it == 1:
            print("step %5d  loss %.4f  ppl %.2f" % (it, float(loss), np.exp(float(loss))))
    return losses


# ---------- 生成: 逐字符自回归, 每次只取最后一个位置的预测 ----------
def sample(seed="a", n=41):
    ctx = encode(seed)
    out = []
    for _ in range(n):
        logits = model.forward(ctx[-T:])
        nxt = int(tf.argmax(logits[0, -1, :]))
        ctx.append(nxt)
        out.append(itos[nxt])
    return seed + "".join(out)


# ================= 参考图 =================
def save_figures(losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_tensorflow")
    os.makedirs(figdir, exist_ok=True)
    labels = list(SENT)                  # 27 个字符

    # ---- 图1: 注意力热图（2 层 x 4 头, 论文 Figure 3-5 的自制品） ----
    _, attns = model.forward(encode(SENT), need_attn=True)
    maps = [A.numpy()[0] for A in attns]     # 每层 (NH, L, L)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8))
    for l in range(NL):
        for hh in range(NH):
            axx = axes[l][hh]
            im = axx.imshow(maps[l][hh], cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
            axx.set_xticks(range(len(labels)))
            axx.set_yticks(range(len(labels)))
            axx.set_xticklabels(labels, rotation=90, fontsize=5)
            axx.set_yticklabels(labels, fontsize=5)
            axx.set_title("layer %d  head %d" % (l, hh), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.62, label="注意力权重 0~1（每行和为1）")
    fig.suptitle("训练后的注意力权重（TensorFlow）：行=当前字符(Q)，列=它看的字符(K)；"
                 "右上角全黑 = 因果mask禁看未来", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig1_attention_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 架构图 + 因果mask ----
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")

    def rbox(cx, cy, bw, bh, text, fc, fs=10):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.08", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arr(p1, p2, color="#3b6fd4", lw=1.8):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    ax.text(3.4, 12.4, "输入：字符序列（一次性整段进入）", ha="center", fontsize=10)
    rbox(3.4, 11.2, 4.6, 0.9, "Embedding 查表  13 → 64", "#dbe9ff")
    rbox(3.4, 9.9, 4.6, 0.9, "+ 正弦位置编码（论文 3.5）", "#d6f0f7")
    rbox(3.4, 7.6, 5.2, 2.6, "Block × 2\n\nLN → 多头注意力(4头, 因果mask) → ⊕残差\n\nLN → FFN(64→256→64, ReLU) → ⊕残差",
         "#fff3d6", fs=9.5)
    rbox(3.4, 5.3, 4.6, 0.9, "Linear  64 → 13", "#e4d9f7")
    rbox(3.4, 4.0, 4.6, 0.9, "softmax → 每个位置的概率", "#ffd9d2")
    rbox(3.4, 2.7, 4.6, 0.9, "取 argmax → 下一个字符", "#d9f2d9")
    for y1, y2 in [(10.7, 10.42), (9.42, 9.05), (6.2, 5.85), (4.8, 4.52), (3.5, 3.22)]:
        arr((3.4, y1), (3.4, y2))
    arr((5.75, 7.6), (6.9, 7.6), "#3b6fd4", lw=1.4)
    ax.text(7.0, 7.6, "对照 03_lstm:\n没有时间循环, 整段一次\n矩阵乘算完\n(论文 Table 1: 串行 O(1))",
            fontsize=9, va="center", linespacing=1.5)
    axm = fig.add_axes([0.68, 0.62, 0.2, 0.24])
    mask = np.triu(np.ones((len(labels), len(labels))), 1)
    axm.imshow(1 - mask, cmap="gray", interpolation="nearest")
    axm.set_title("因果mask\n白=可见 黑=禁看未来", fontsize=8)
    axm.set_xticks([]); axm.set_yticks([])
    ax.set_title("迷你 Transformer 架构（pre-LN 版，TensorFlow 实现）", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig2_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 正弦位置编码 ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="编码值")
    ax.set_xlabel("编码维度 i")
    ax.set_ylabel("位置 pos")
    ax.set_title("正弦位置编码（论文 3.5 节公式）：每个位置一根独特的\"指纹\"")
    fig.savefig(os.path.join(figdir, "fig3_positional_encoding.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 训练曲线 ----
    steps = np.arange(1, len(losses) + 1)
    losses_arr = np.array(losses)
    smooth = np.convolve(losses_arr, np.ones(51) / 51, mode="valid")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(steps, losses_arr, color="#2e8b57", alpha=0.25, lw=0.8)
    ax.plot(steps[25:-25], smooth, color="#2e8b57", lw=2, label="滑动平均(51步)")
    ax.axhline(np.log(V), color="gray", ls="--", lw=1.4)
    ax.text(3300, np.log(V) + 0.06, "瞎猜下限 ln(V) = %.3f" % np.log(V), fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss（对数轴）")
    ax.set_title("Transformer 训练曲线（TensorFlow，与 03 的 LSTM 同任务对照）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig4_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


if __name__ == "__main__":
    losses = train()
    print("\ntarget      :", SENT.strip())
    print("from 'a'    :", sample().strip())
    print("from prefix :", sample(seed="attention is ").strip())
    save_figures(losses)
