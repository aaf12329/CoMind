# -*- coding: utf-8 -*-
r"""
04 迷你 Transformer —— 《Attention Is All You Need》最小可训练版
================================================================
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

对照 03_lstm.py 最重要的一个观察:
    LSTM 里有个 for t in range(...) 的时间循环, 一步算一帧;
    这里没有时间循环 —— 注意力把整段序列的交互一次矩阵乘算完。
    这就是论文 Table 1 里 Self-Attention 那一行 O(1) 串行操作的代价。

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 04_mini_transformer\04_mini_transformer.py
预期:  梯度检查 < 1e-4(含 ReLU, 阈值放宽);  loss 从 2.56 降到 0.1 以下;
       逐字默写 "attention is all you need."
产物:  本文件夹 figs/ 下生成 4 张参考图:
       fig1 注意力热图(2层x4头) | fig2 架构图+因果mask | fig3 位置编码 | fig4 训练曲线
"""
import os
import numpy as np

rng = np.random.default_rng(0)

# ---------- 数据(与 03 完全一致, 保证对照公平) ----------
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


def init_params():
    s = 0.02
    p = {'Emb': rng.normal(0, s, (V, D))}
    for l in range(NL):
        for nm, sh in [('Wq', (D, D)), ('Wk', (D, D)), ('Wv', (D, D)),
                       ('Wo', (D, D)), ('W1', (D, 4 * D)), ('W2', (4 * D, D))]:
            p['%s%d' % (nm, l)] = rng.normal(0, s, sh)
    p['Wout'] = rng.normal(0, s, (D, V))
    return p


# ---------- 基础件 ----------
def layernorm(x):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    xn = (x - mu) * rstd
    return xn, (xn, rstd)


def ln_backward(dy, cache):
    """无仿射 LayerNorm 的反向, 标准公式。"""
    xn, rstd = cache
    return rstd * (dy - dy.mean(-1, keepdims=True)
                   - xn * (dy * xn).mean(-1, keepdims=True))


def softmax_rows(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ---------- 前向 ----------
def forward(p, xs, ys=None):
    L = len(xs)
    idx = np.array(xs)
    x = p['Emb'][idx] + PE[:L]           # 词义 + 位置, 直接相加(论文 3.5)
    caches = []
    for l in range(NL):
        bc = {}
        xn1, bc['ln1'] = layernorm(x)    # pre-LN: 先归一化再进子层
        # ---- 多头注意力(论文 3.2.1 + 3.2.2) ----
        q = xn1 @ p['Wq%d' % l]
        k = xn1 @ p['Wk%d' % l]
        v = xn1 @ p['Wv%d' % l]
        qh = q.reshape(L, NH, DH).transpose(1, 0, 2)     # 拆成 NH 个头 (NH,L,DH)
        kh = k.reshape(L, NH, DH).transpose(1, 0, 2)
        vh = v.reshape(L, NH, DH).transpose(1, 0, 2)
        scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(DH)   # 缩放点积, 论文式(1)
        mask = np.triu(np.ones((L, L)), 1).astype(bool)
        scores = scores + np.where(mask, -1e9, 0.0)         # 因果mask: 未来=负无穷
        A = softmax_rows(scores)                            # 每行和为1的注意力权重
        ah = A @ vh                                         # 按权重加权 V
        concat = ah.transpose(1, 0, 2).reshape(L, D)        # 拼回所有头
        attn = concat @ p['Wo%d' % l]
        xm = x + attn                                       # 残差连接(论文 3.1)
        # ---- 逐位置前馈网络(论文 3.3 节) ----
        xn2, bc['ln2'] = layernorm(xm)
        h = np.maximum(xn2 @ p['W1%d' % l], 0.0)            # ReLU
        ffn = h @ p['W2%d' % l]
        x = xm + ffn                                        # 残差连接
        bc.update(qh=qh, kh=kh, vh=vh, A=A, concat=concat, h=h)
        caches.append(bc)
    logits = x @ p['Wout']
    out = {'logits': logits, 'idx': idx, 'x_last': x}
    if ys is not None:
        ls = logits - logits.max(-1, keepdims=True)
        e = np.exp(ls)
        P = e / e.sum(-1, keepdims=True)
        yarr = np.array(ys)
        out['P'] = P
        out['loss'] = -np.mean(np.log(P[np.arange(L), yarr] + 1e-12))
        out['y'] = yarr
    return out, caches


# ---------- 反向传播: 从损失出发, 层间倒序(时间上没有循环!) ----------
def backward(p, out, caches):
    L = out['logits'].shape[0]
    P, yarr, x_last = out['P'], out['y'], out['x_last']
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    # ---- 起点: 损失块, 种子梯度 p - onehot(y) ----
    dlogits = P.copy()
    dlogits[np.arange(L), yarr] -= 1.0
    dlogits /= L
    grads['Wout'] += x_last.T @ dlogits
    dx = dlogits @ p['Wout'].T                       # (L,D)
    for l in reversed(range(NL)):
        bc = caches[l]
        # ---- 残差2: x = xm + FFN(LN(xm)) ----
        dh = dx @ p['W2%d' % l].T
        dh = dh * (bc['h'] > 0)                      # ReLU 导数
        grads['W2%d' % l] += bc['h'].T @ dx
        dxn2 = dh @ p['W1%d' % l].T
        grads['W1%d' % l] += bc['ln2'][0].T @ dh
        dxm = dx + ln_backward(dxn2, bc['ln2'])
        # ---- 残差1: xm = x0 + Concat(heads)@Wo ----
        dconcat = dxm @ p['Wo%d' % l].T
        grads['Wo%d' % l] += bc['concat'].T @ dxm
        ah = dconcat.reshape(L, NH, DH).transpose(1, 0, 2)
        dA = ah @ bc['vh'].transpose(0, 2, 1)        # 注意力权重 A 的梯度
        dvh = bc['A'].transpose(0, 2, 1) @ ah
        # softmax 反向(逐行): dS = A⊙(dA - rowsum(dA⊙A))
        dscores = bc['A'] * (dA - (dA * bc['A']).sum(-1, keepdims=True))
        dqh = dscores @ bc['kh'] / np.sqrt(DH)       # 对 q,k,v 的梯度
        dkh = dscores.transpose(0, 2, 1) @ bc['qh'] / np.sqrt(DH)
        dq = dqh.transpose(1, 0, 2).reshape(L, D)
        dk = dkh.transpose(1, 0, 2).reshape(L, D)
        dv = dvh.transpose(1, 0, 2).reshape(L, D)
        grads['Wq%d' % l] += bc['ln1'][0].T @ dq
        grads['Wk%d' % l] += bc['ln1'][0].T @ dk
        grads['Wv%d' % l] += bc['ln1'][0].T @ dv
        dxn1 = dq @ p['Wq%d' % l].T + dk @ p['Wk%d' % l].T + dv @ p['Wv%d' % l].T
        dx = dxm + ln_backward(dxn1, bc['ln1'])      # 两路残差梯度汇合, 传给下一层
    np.add.at(grads['Emb'], out['idx'], dx)          # 同一字符多次出现要累加
    return grads


# ---------- 数值梯度检查 ----------
def grad_check(p, n=20, eps=1e-5):
    xs = list(rng.integers(0, V, T))
    ys = list(rng.integers(0, V, T))
    out, caches = forward(p, xs, ys)
    ana = backward(p, out, caches)
    worst, worst_info = 0.0, None
    for k in p:
        flat = p[k]
        idxs = rng.choice(flat.size, size=min(n, flat.size), replace=False)
        for j in idxs:
            idx = np.unravel_index(j, flat.shape)
            old = flat[idx]
            flat[idx] = old + eps; lp, _ = forward(p, xs, ys)
            flat[idx] = old - eps; lm, _ = forward(p, xs, ys)
            flat[idx] = old
            num = (lp['loss'] - lm['loss']) / (2 * eps)
            a = ana[k][idx]
            scaled = abs(a - num) / max(1.0, abs(a), abs(num))
            if scaled > worst:
                worst, worst_info = scaled, (k, tuple(int(i) for i in idx), a, num)
    print("grad check  worst: %s%s  manual=%.3e numeric=%.3e" % worst_info)
    return worst


# ---------- Adam ----------
class Adam:
    def __init__(self, params, lr=3e-3):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, 0.9, 0.999, 1e-8, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m = self.m[k] / (1 - self.b1 ** self.t)
            v = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m / (np.sqrt(v) + self.eps)


def clip_grads(grads, limit=5.0):
    for g in grads.values():
        np.clip(g, -limit, limit, out=g)


# ---------- 训练(和 03 同一套流程) ----------
def train(iters=4000):
    p = init_params()
    opt = Adam(p)
    print("V=%d, 瞎猜下限 ln(V) = %.3f" % (V, np.log(V)))
    losses = []
    for it in range(1, iters + 1):
        s = rng.integers(0, len(data) - T - 1)
        chunk = encode(data[s:s + T + 1])
        xs, ys = chunk[:-1], chunk[1:]
        out, caches = forward(p, xs, ys)
        grads = backward(p, out, caches)
        clip_grads(grads)
        opt.step(p, grads)
        losses.append(out['loss'])
        if it % 250 == 0 or it == 1:
            print("step %5d  loss %.4f  ppl %.2f" % (it, out['loss'], np.exp(out['loss'])))
    return p, losses


# ---------- 生成: 逐字符自回归, 每次只取最后一个位置的预测 ----------
def sample(p, seed="a", n=41):
    ctx = encode(seed)
    out = []
    for _ in range(n):
        fwd, _ = forward(p, ctx[-T:])
        nxt = int(np.argmax(fwd['logits'][-1]))
        ctx.append(nxt)
        out.append(itos[nxt])
    return seed + "".join(out)


# ================= 参考图 =================
def save_figures(p, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figdir, exist_ok=True)
    labels = list(SENT)                  # 27 个字符

    # ---- 图1: 注意力热图（2 层 x 4 头, 论文 Figure 3-5 的自制品） ----
    out, caches = forward(p, encode(SENT))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8))
    for l in range(NL):
        A = caches[l]['A']               # (NH, L, L)
        for hh in range(NH):
            axx = axes[l][hh]
            im = axx.imshow(A[hh], cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
            axx.set_xticks(range(len(labels)))
            axx.set_yticks(range(len(labels)))
            axx.set_xticklabels(labels, rotation=90, fontsize=5)
            axx.set_yticklabels(labels, fontsize=5)
            axx.set_title("layer %d  head %d" % (l, hh), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.62, label="注意力权重 0~1（每行和为1）")
    fig.suptitle("训练后的注意力权重：行=当前字符(Q)，列=它看的字符(K)；"
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
    # 因果mask小图
    axm = fig.add_axes([0.68, 0.62, 0.2, 0.24])
    mask = np.triu(np.ones((len(labels), len(labels))), 1)
    axm.imshow(1 - mask, cmap="gray", interpolation="nearest")
    axm.set_title("因果mask\n白=可见 黑=禁看未来", fontsize=8)
    axm.set_xticks([]); axm.set_yticks([])
    ax.set_title("迷你 Transformer 架构（pre-LN 版）", fontsize=12)
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
    losses = np.array(losses)
    smooth = np.convolve(losses, np.ones(51) / 51, mode="valid")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(steps, losses, color="#c44e52", alpha=0.25, lw=0.8)
    ax.plot(steps[25:-25], smooth, color="#c44e52", lw=2, label="滑动平均(51步)")
    ax.axhline(np.log(V), color="gray", ls="--", lw=1.4)
    ax.text(3300, np.log(V) + 0.06, "瞎猜下限 ln(V) = %.3f" % np.log(V), fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss（对数轴）")
    ax.set_title("Transformer 训练曲线（与 03 的 LSTM 同任务对照）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig4_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


if __name__ == "__main__":
    p = init_params()
    worst = grad_check(p)
    print("grad check  max scaled error = %.2e" % worst)
    assert worst < 1e-4, "反向传播推错了!"
    print("grad check passed\n")
    p, losses = train()
    print("\ntarget      :", SENT.strip())
    print("from 'a'    :", sample(p).strip())
    print("from prefix :", sample(p, seed="attention is ").strip())
    save_figures(p, losses)
