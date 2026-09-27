# -*- coding: utf-8 -*-
r"""
03 LSTM —— 手推 BPTT 的完整版
==============================
任务: 字符级语言模型。语料就是一句话循环出现:
      "attention is all you need. "
      看前文, 预测下一个字符。训好后逐字默写出这句话。

LSTM 前向方程(每一步 t, ⊙ 是逐元素乘):
    a = W·x + U·h_prev + b         一次算出 4 个门的预激活, 按行切成 4 份
    f = sigmoid(a_f)               遗忘门 0~1: 旧记忆保留几成
    i = sigmoid(a_i)               输入门 0~1: 新信息写入几成
    o = sigmoid(a_o)               输出门 0~1: 记忆向外露出几成
    g = tanh(a_g)                  候选内容 -1~1: 这一步想写入什么
    c = f⊙c_prev + i⊙g            细胞状态 = 传送带, LSTM 的核心
    h = o⊙tanh(c)                 隐状态: 传送带经输出门后的对外读数
    logits = Why·h + by -> softmax -> 交叉熵

反向传播(BPTT): 时间上从最后一个字符倒着回到第一个。
两条梯度流, 每步在门处汇合:
    流1 dh:  来自本步输出头 + 未来所有步经 U 传回
    流2 dc:  本步 dh 的一份 + 未来经遗忘门传回(乘 f, 传送带跨步)
    dlogits = p - onehot(y)                  种子
    dh  = Why^T@dlogits + dh_next
    do  = dh⊙tanh(c);        da_o = do⊙o(1-o)
    dc  = dh⊙o⊙(1-tanh(c)^2) + dc_next
    df  = dc⊙c_prev -> da_f = df⊙f(1-f)
    di  = dc⊙g       -> da_i = di⊙i(1-i)
    dg  = dc⊙i       -> da_g = dg⊙(1-g^2)
    dW += da⊗x;  dU += da⊗h_prev;  db += da
    dh_next = U^T@da;   dc_next = dc⊙f

对比 RNN: c 这条传送带上每步只做"乘 f(接近1) + 加新内容",
    梯度沿 c 回传几乎不衰减 —— 这就是 LSTM 治梯度消失的机理。
跑法:  D:\Anaconda\envs\imooc_ai\python.exe 03_lstm\03_lstm.py
预期:  梯度检查 < 1e-6;  loss 从 2.56 降到 0.05 以下;  逐字默写全句
产物:  本文件夹 figs/ 下生成 4 张参考图:
       fig1 LSTM单元图 | fig2 BPTT展开图 | fig3 训练曲线 | fig4 门活动热图
"""
import os
import numpy as np

rng = np.random.default_rng(0)

# ---------- 数据 ----------
SENT = "attention is all you need. "
data = SENT * 400
chars = sorted(set(data))
V = len(chars)
stoi = {ch: k for k, ch in enumerate(chars)}
itos = {k: ch for ch, k in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]

H = 96          # 隐状态维度


def init_params():
    p = {
        'W':   rng.normal(0, 0.1, (4 * H, V)),   # 输入 -> 4个门
        'U':   rng.normal(0, 0.1, (4 * H, H)),   # h_prev -> 4个门
        'b':   np.zeros(4 * H),
        'Why': rng.normal(0, 0.1, (V, H)),       # h -> 词表打分
        'by':  np.zeros(V),
    }
    p['b'][:H] = 1.0        # 遗忘门偏置置1: 初期倾向于"别忘", 经典技巧
    return p


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- 前向 ----------
def lstm_step(x, h_prev, c_prev, p):
    a = p['W'] @ x + p['U'] @ h_prev + p['b']
    f = sigmoid(a[0:H])
    i = sigmoid(a[H:2 * H])
    o = sigmoid(a[2 * H:3 * H])
    g = np.tanh(a[3 * H:])
    c = f * c_prev + i * g
    tanh_c = np.tanh(c)
    h = o * tanh_c
    cache = dict(x=x, h_prev=h_prev, c_prev=c_prev,
                 f=f, i=i, o=o, g=g, c=c, tanh_c=tanh_c, h=h)
    return h, c, cache


def forward(p, xs, ys):
    """xs/ys: 索引序列。返回平均交叉熵 + 每步缓存(反向要用)。"""
    h = np.zeros(p['Why'].shape[1])
    c = np.zeros_like(h)
    caches, loss = [], 0.0
    for t in range(len(xs)):
        x = np.zeros(V); x[xs[t]] = 1.0
        h, c, cache = lstm_step(x, h, c, p)
        logits = p['Why'] @ h + p['by']
        logits = logits - logits.max()          # 数值稳定的 softmax
        pr = np.exp(logits)
        pr = pr / pr.sum()
        loss += -np.log(pr[ys[t]] + 1e-12)
        cache['probs'] = pr
        caches.append(cache)
    return loss / len(xs), caches


# ---------- 反向传播: 从损失出发, 时间倒序 ----------
def backward(p, caches, ys):
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    T = len(caches)
    dh_next = np.zeros(H)
    dc_next = np.zeros(H)
    for t in reversed(range(T)):
        ca = caches[t]
        # ---- 反向传播的起点: 损失块 ----
        dlogits = ca['probs'].copy()            # softmax+CE 的种子: p - onehot(y)
        dlogits[ys[t]] -= 1.0
        grads['Why'] += np.outer(dlogits, ca['h'])
        grads['by'] += dlogits
        dh = p['Why'].T @ dlogits + dh_next     # 流1: 本步输出头 + 未来传回
        # ---- 输出门: h = o ⊙ tanh(c) ----
        do = dh * ca['tanh_c']
        da_o = do * ca['o'] * (1 - ca['o'])
        # ---- 流2: 细胞状态, 两路汇合 ----
        dc = dh * ca['o'] * (1 - ca['tanh_c'] ** 2) + dc_next
        # ---- 三个门对 c 的贡献 ----
        df = dc * ca['c_prev']
        da_f = df * ca['f'] * (1 - ca['f'])
        di = dc * ca['g']
        da_i = di * ca['i'] * (1 - ca['i'])
        dg = dc * ca['i']
        da_g = dg * (1 - ca['g'] ** 2)
        da = np.concatenate([da_f, da_i, da_o, da_g])
        grads['W'] += np.outer(da, ca['x'])
        grads['U'] += np.outer(da, ca['h_prev'])
        grads['b'] += da
        # ---- 传给 t-1 的两条流 ----
        dh_next = p['U'].T @ da
        dc_next = dc * ca['f']                  # 走遗忘门这条传送带
    for k in grads:
        grads[k] /= T                           # 前向 loss 取了平均, 两边一致
    return grads


# ---------- 数值梯度检查 ----------
def grad_check(p, n=30, eps=1e-5):
    xs = list(rng.integers(0, V, 7))
    ys = list(rng.integers(0, V, 7))
    _, caches = forward(p, xs, ys)
    ana = backward(p, caches, ys)
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
            num = (lp - lm) / (2 * eps)
            a = ana[k][idx]
            # 大梯度看相对误差, 小梯度看绝对误差(分母趋零会假报)
            scaled = abs(a - num) / max(1.0, abs(a), abs(num))
            if scaled > worst:
                worst, worst_info = scaled, (k, idx, a, num)
    k, idx, a, num = worst_info
    print("grad check  worst: %s%s  manual=%.3e numeric=%.3e" % (k, tuple(int(i) for i in idx), a, num))
    return worst


# ---------- Adam ----------
class Adam:
    def __init__(self, params, lr=0.02):
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
    """梯度裁剪: 防 BPTT 连乘爆炸, RNN 时代祖传操作。"""
    for g in grads.values():
        np.clip(g, -limit, limit, out=g)


# ---------- 训练 ----------
def train(iters=3000, T=32):
    p = init_params()
    opt = Adam(p)
    print("V=%d, 瞎猜下限 ln(V) = %.3f" % (V, np.log(V)))
    losses = []
    for it in range(1, iters + 1):
        s = rng.integers(0, len(data) - T - 1)
        chunk = encode(data[s:s + T + 1])
        xs, ys = chunk[:-1], chunk[1:]
        loss, caches = forward(p, xs, ys)
        grads = backward(p, caches, ys)
        clip_grads(grads)
        opt.step(p, grads)
        losses.append(loss)
        if it % 200 == 0 or it == 1:
            print("step %5d  loss %.4f  ppl %.2f" % (it, loss, np.exp(loss)))
    return p, losses


# ---------- 生成 ----------
def sample(p, seed="a", n=40):
    h = np.zeros(p['Why'].shape[1])
    c = np.zeros_like(h)
    x = np.zeros(V); x[stoi[seed]] = 1.0
    out = []
    for _ in range(n):
        h, c, _ = lstm_step(x, h, c, p)
        idx = int(np.argmax(p['Why'] @ h + p['by']))    # 贪心解码
        out.append(itos[idx])
        x = np.zeros(V); x[idx] = 1.0
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

    # ---- 图1: LSTM 单元图（一条传送带 + 三个开关） ----
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    def rbox(cx, cy, bw, bh, text, fc, fs=9.5):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.06", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def circ(cx, cy, text):
        ax.text(cx, cy, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="circle,pad=0.28", fc="white", ec="#444444", lw=1.2))

    def arr(p1, p2, color="#666666", lw=1.5, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    # 传送带（细胞状态）
    ax.annotate("", xy=(9.2, 6.3), xytext=(0.9, 6.3),
                arrowprops=dict(arrowstyle="-|>", color="#2e8b57", lw=2.8))
    ax.text(0.55, 6.55, "c(t-1)", fontsize=11, color="#2e8b57")
    ax.text(9.15, 6.55, "c(t)", fontsize=11, color="#2e8b57")
    circ(2.2, 6.3, "×f")
    circ(4.9, 6.3, "+")
    # 门（学出来的开关）
    rbox(2.0, 3.4, 1.9, 0.9, "遗忘门 f=σ(·)\n留多少旧记忆", "#dbe9ff")
    rbox(4.15, 3.4, 1.9, 0.9, "输入门 i=σ(·)\n写多少新内容", "#ffe9c9")
    rbox(6.3, 3.4, 1.9, 0.9, "候选 g=tanh(·)\n要写入的内容", "#fff9c4")
    rbox(8.5, 3.4, 1.9, 0.9, "输出门 o=σ(·)\n向外露出多少", "#e4d9f7")
    # 输入
    rbox(2.6, 1.0, 2.4, 0.75, "x(t)   h(t-1)", "#ececec")
    for gx in [2.0, 4.15, 6.3, 8.5]:
        arr((2.9, 1.42), (gx, 2.9), "#888888")
    # 门 -> 传送带
    arr((2.05, 3.9), (2.18, 5.95), "#2e8b57")
    arr((5.0, 3.9), (4.92, 5.95), "#2e8b57")
    ax.text(5.3, 5.15, "i⊙g", fontsize=9, color="#2e8b57")
    # 读数路径: c -> tanh -> ⊙o -> h（走传送带和门排之间的走廊）
    arr((7.35, 6.3), (7.35, 5.55), "#3b6fd4", lw=1.8)
    circ(7.35, 5.2, "tanh")
    circ(8.6, 5.2, "⊙o")
    arr((7.62, 5.2), (8.32, 5.2), "#3b6fd4")
    arr((8.6, 3.9), (8.6, 4.9), "#3b6fd4")
    arr((8.88, 5.2), (9.35, 5.2), "#3b6fd4")
    ax.text(9.6, 5.2, "h(t)", fontsize=11, color="#3b6fd4")
    ax.text(0.3, 0.12,
            "传送带: c(t) = f⊙c(t-1) + i⊙g\n"
            "读数:   h(t) = o⊙tanh(c(t))",
            fontsize=10, linespacing=1.6, va="bottom")
    ax.set_title("LSTM 单元：一条传送带（绿）+ 三个学习出来的开关", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig1_lstm_cell.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: BPTT 时间展开（两条反向梯度流） ----
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    xs4 = [1.6, 4.4, 7.2, 10.0]
    for i, xc in enumerate(xs4):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((xc - 0.85, 2.65), 1.7, 1.1,
                                    boxstyle="round,pad=0.06", fc="#ececec", ec="#444444", lw=1.2))
        ax.text(xc, 3.2, "LSTM cell", ha="center", va="center", fontsize=10)
        ax.text(xc, 2.15, "t=%d" % (i + 1), ha="center", fontsize=9, color="#666666")
        circ = ax.text(xc, 1.3, "L", ha="center", va="center", fontsize=10,
                       bbox=dict(boxstyle="circle,pad=0.25", fc="#ffd9d2", ec="#d4402e", lw=1.1))
        arr((xc, 2.6), (xc, 1.62), "#3b6fd4", lw=1.3)
        if i < 3:
            nxt = xs4[i + 1]
            arr((xc + 0.9, 3.2), (nxt - 0.9, 3.2), "#3b6fd4", lw=1.8)          # h 前向
            arr((xc + 0.9, 4.75), (nxt - 0.9, 4.75), "#2e8b57", lw=2.2)        # c 前向
            arr((nxt - 0.9, 3.75), (xc + 0.9, 3.75), "#d4402e", lw=1.6, ls="--")   # dh 反向
            arr((nxt - 0.9, 4.25), (xc + 0.9, 4.25), "#e07b00", lw=1.6, ls="--")   # dc 反向
    ax.text(2.95, 3.42, "h →", fontsize=9, color="#3b6fd4")
    ax.text(2.95, 4.92, "c →", fontsize=9, color="#2e8b57")
    ax.text(5.9, 3.9, "dh 反向", fontsize=9, color="#d4402e")
    ax.text(5.9, 4.42, "dc 反向（乘 f 传回）", fontsize=9, color="#e07b00")
    arr((11.35, 1.3), (1.0, 1.3), "#d4402e", lw=1.6, ls="--")
    ax.text(6.2, 0.85, "每一步的 L 都有自己的种子: dlogits = p − onehot(y)",
            fontsize=9.5, color="#d4402e", ha="center")
    ax.text(11.6, 1.3, "反向传播\n从这里开始", fontsize=9, color="#d4402e", va="center")
    ax.set_title("BPTT：前向（实线）从左到右，反向（虚线）从最后一个损失倒着走", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig2_bptt_unroll.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 训练曲线 ----
    steps = np.arange(1, len(losses) + 1)
    losses = np.array(losses)
    smooth = np.convolve(losses, np.ones(51) / 51, mode="valid")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(steps, losses, color="#c44e52", alpha=0.25, lw=0.8)
    ax.plot(steps[25:-25], smooth, color="#c44e52", lw=2, label="滑动平均(51步)")
    ax.axhline(np.log(V), color="gray", ls="--", lw=1.4)
    ax.text(2400, np.log(V) + 0.06, "瞎猜下限 ln(V) = %.3f" % np.log(V), fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss（对数轴）")
    ax.set_title("LSTM 训练曲线：从瞎猜到逐字默写")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig3_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 门活动热图（同一句话上的 f / i / o） ----
    idxs = encode(SENT)
    h = np.zeros(H); c = np.zeros(H)
    Fs, Is, Os = [], [], []
    for k in idxs:
        xv = np.zeros(V); xv[k] = 1.0
        h, c, ca = lstm_step(xv, h, c, p)
        Fs.append(ca['f']); Is.append(ca['i']); Os.append(ca['o'])
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 5.8), sharex=True)
    for axx, M, name in zip(axes, [np.array(Fs).T, np.array(Is).T, np.array(Os).T],
                            ["遗忘门 f（留多少旧记忆）", "输入门 i（写多少新内容）", "输出门 o（露出多少）"]):
        im = axx.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axx.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
    axes[2].set_xticks(range(len(SENT)))
    axes[2].set_xticklabels(list(SENT), fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.75, label="门开合程度 0~1")
    fig.suptitle("门的活动：96 个维度 × 27 个字符 —— 开关是训练学出来的", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig4_gates_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


if __name__ == "__main__":
    p = init_params()
    worst = grad_check(p)
    print("grad check  max scaled error = %.2e" % worst)
    assert worst < 1e-6, "BPTT 推错了!"
    print("grad check passed\n")
    p, losses = train()
    print("\ntarget :", SENT.strip())
    print("lstm   :", sample(p).strip())
    save_figures(p, losses)
