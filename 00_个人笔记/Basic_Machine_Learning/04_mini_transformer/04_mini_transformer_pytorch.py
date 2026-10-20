# -*- coding: utf-8 -*-
r"""
04 迷你 Transformer · PyTorch 版 —— 《Attention Is All You Need》最小可训练版
=============================================================================
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
    手搓版                            PyTorch
    ------------------------------    --------------------------------------
    手写 layernorm + 反向              nn.LayerNorm(elementwise_affine=False)
                                      ← "无仿射"正好对应手搓版没写 γ/β
    手写的多头注意力（拆分/拼回）      还是手写这几步（教学版），但全部换 torch 张量：
    手推 softmax 反向                  softmax 的反向由 autograd 处理
    手推整棵计算图的反向（40 行）      loss.backward()
    Adam 手写 / 裁剪 np.clip          torch.optim.Adam / grad.clamp_

    （工程里注意力会直接用 F.scaled_dot_product_attention(is_causal=True)——
      融合算子更快；本教学版把矩阵乘一步步写开，图和公式才对得上。）

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 04_mini_transformer\04_mini_transformer_pytorch.py
预期:  梯度检查 < 1e-4(含 ReLU, 阈值放宽);  loss 从 2.56 降到 0.1 以下;  逐字默写
产物:  本文件夹 figs_pytorch/ 下 4 张参考图（与 TF 版同名，可并排对照）:
       fig1 注意力热图(2层x4头) | fig2 架构图+因果mask | fig3 位置编码 | fig4 训练曲线
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)

# ---------- 数据(与 03 完全一致, 保证对照公平；也与 TF 版逐位相同) ----------
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


class Block(nn.Module):
    """pre-LN 的一个 block：LN→注意力→残差；LN→FFN→残差。全部无偏置、LN 无仿射。"""

    def __init__(self, l):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-5).double()
        self.ln2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-5).double()
        self.q = nn.Linear(D, D, bias=False).double()
        self.k = nn.Linear(D, D, bias=False).double()
        self.v = nn.Linear(D, D, bias=False).double()
        self.o = nn.Linear(D, D, bias=False).double()
        self.f1 = nn.Linear(D, 4 * D, bias=False).double()
        self.f2 = nn.Linear(4 * D, D, bias=False).double()
        with torch.no_grad():
            # nn.Linear 存 (出,入)，手搓 W 是 (入,出) → 全部转置后灌入
            self.q.weight.copy_(torch.tensor(NW['Wq%d' % l].T))
            self.k.weight.copy_(torch.tensor(NW['Wk%d' % l].T))
            self.v.weight.copy_(torch.tensor(NW['Wv%d' % l].T))
            self.o.weight.copy_(torch.tensor(NW['Wo%d' % l].T))
            self.f1.weight.copy_(torch.tensor(NW['W1%d' % l].T))
            self.f2.weight.copy_(torch.tensor(NW['W2%d' % l].T))


class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D).double()
        with torch.no_grad():
            self.emb.weight.copy_(torch.tensor(NW['Emb']))
        self.register_buffer("pe", torch.tensor(PE, dtype=torch.float64))
        self.blocks = nn.ModuleList([Block(l) for l in range(NL)])
        self.out = nn.Linear(D, V, bias=False).double()
        with torch.no_grad():
            self.out.weight.copy_(torch.tensor(NW['Wout'].T))

    def forward(self, idx, need_attn=False):
        """idx: (L,) 字符索引。返回 logits (L,V)；need_attn 时附带每层每头的注意力权重。"""
        L = idx.shape[0]
        x = self.emb(idx) + self.pe[:L]                  # 词义 + 位置, 直接相加(论文 3.5)
        masks = (torch.triu(torch.ones(L, L, dtype=torch.bool), 1))   # 因果mask: 未来=禁看
        attns = []
        for blk in self.blocks:
            xn = blk.ln1(x)                              # pre-LN: 先归一化再进子层
            L_ = L
            qh = blk.q(xn).reshape(L_, NH, DH).transpose(0, 1)        # (NH, L, DH)
            kh = blk.k(xn).reshape(L_, NH, DH).transpose(0, 1)
            vh = blk.v(xn).reshape(L_, NH, DH).transpose(0, 1)
            scores = qh @ kh.transpose(1, 2) / (DH ** 0.5)            # 缩放点积, 论文式(1)
            scores = scores + torch.where(masks, torch.tensor(-1e9, dtype=torch.float64),
                                          torch.tensor(0.0, dtype=torch.float64))
            A = F.softmax(scores, dim=-1)                             # 每行和为1的注意力权重
            ah = A @ vh                                               # 按权重加权 V
            concat = ah.transpose(0, 1).reshape(L_, D)                # 拼回所有头
            x = x + blk.o(concat)                                     # 残差连接(论文 3.1)
            x = x + blk.f2(F.relu(blk.f1(blk.ln2(x))))                # FFN + 残差(论文 3.3)
            if need_attn:
                attns.append(A)
        logits = self.out(x)
        return (logits, attns) if need_attn else logits


model = MiniTransformer()


def loss_on(xs, ys):
    logits = model(torch.tensor(xs))
    return F.cross_entropy(logits, torch.tensor(ys))   # 种子 p-onehot 藏在这里的反向里


# ---------- 数值梯度检查: 质检员上岗（autograd vs 差分） ----------
GC_S = 100                                            # 固定一段语料，两版用同一段
chunk = encode(data[GC_S:GC_S + T + 1])
gc_xs, gc_ys = chunk[:-1], chunk[1:]
loss = loss_on(gc_xs, gc_ys)
loss.backward()

eps = 1e-5
worst = 0.0
named = [("Emb", model.emb.weight)]
for l in range(NL):
    blk = model.blocks[l]
    for nm, prm in [("Wq", blk.q.weight), ("Wk", blk.k.weight), ("Wv", blk.v.weight),
                    ("Wo", blk.o.weight), ("W1", blk.f1.weight), ("W2", blk.f2.weight)]:
        named.append(("%s%d" % (nm, l), prm))
named.append(("Wout", model.out.weight))
for name, prm in named:
    flat = prm.detach().reshape(-1)
    g = prm.grad.reshape(-1)
    n = flat.numel()
    for j in [0, n // 2, n - 1]:                      # 每个参数张量抽 3 个元素查
        old = flat[j].item()
        with torch.no_grad():
            flat[j] = old + eps; lp = loss_on(gc_xs, gc_ys).item()
            flat[j] = old - eps; lm = loss_on(gc_xs, gc_ys).item()
            flat[j] = old
        num = (lp - lm) / (2 * eps)
        ana = g[j].item()
        scaled = abs(ana - num) / max(1.0, abs(ana), abs(num))    # 大梯度看相对, 小梯度看绝对
        worst = max(worst, scaled)
print("grad check  max scaled error = %.2e" % worst)
assert worst < 1e-4, "autograd 反向推错了?!"       # 含 ReLU，阈值按惯例放宽
print("grad check passed\n")


# ---------- 训练(和 03 同一套流程) ----------
def train(iters=4000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    print("V=%d, 瞎猜下限 ln(V) = %.3f" % (V, np.log(V)))
    losses = []
    for it in range(1, iters + 1):
        s = rng.integers(0, len(data) - T - 1)        # 与 TF 版同一随机流抽起点
        chunk = encode(data[s:s + T + 1])
        xs, ys = chunk[:-1], chunk[1:]
        opt.zero_grad()
        loss = loss_on(xs, ys)
        loss.backward()
        with torch.no_grad():
            for prm in model.parameters():
                prm.grad.clamp_(-5.0, 5.0)            # 与手搓版同款逐元素裁剪
        opt.step()
        losses.append(loss.item())
        if it % 250 == 0 or it == 1:
            print("step %5d  loss %.4f  ppl %.2f" % (it, loss.item(), np.exp(loss.item())))
    return losses


# ---------- 生成: 逐字符自回归, 每次只取最后一个位置的预测 ----------
def sample(seed="a", n=41):
    model.eval()
    ctx = encode(seed)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = model(torch.tensor(ctx[-T:]))
            nxt = int(logits[-1].argmax())
            ctx.append(nxt)
            out.append(itos[nxt])
    model.train()
    return seed + "".join(out)


# ---------- 注意力权重提取（做 fig1 热图用） ----------
def attention_maps(sentence):
    model.eval()
    with torch.no_grad():
        _, attns = model(torch.tensor(encode(sentence)), need_attn=True)
    model.train()
    return [A.numpy() for A in attns]     # 每层 (NH, L, L)


# ================= 参考图 =================
def save_figures(losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_pytorch")
    os.makedirs(figdir, exist_ok=True)
    labels = list(SENT)                  # 27 个字符

    # ---- 图1: 注意力热图（2 层 x 4 头, 论文 Figure 3-5 的自制品） ----
    maps = attention_maps(SENT)
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
    fig.suptitle("训练后的注意力权重（PyTorch）：行=当前字符(Q)，列=它看的字符(K)；"
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
    ax.set_title("迷你 Transformer 架构（pre-LN 版，PyTorch 实现）", fontsize=12)
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
    ax.plot(steps, losses_arr, color="#c44e52", alpha=0.25, lw=0.8)
    ax.plot(steps[25:-25], smooth, color="#c44e52", lw=2, label="滑动平均(51步)")
    ax.axhline(np.log(V), color="gray", ls="--", lw=1.4)
    ax.text(3300, np.log(V) + 0.06, "瞎猜下限 ln(V) = %.3f" % np.log(V), fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss（对数轴）")
    ax.set_title("Transformer 训练曲线（PyTorch，与 03 的 LSTM 同任务对照）")
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
