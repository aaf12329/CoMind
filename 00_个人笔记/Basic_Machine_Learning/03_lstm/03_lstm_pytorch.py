# -*- coding: utf-8 -*-
r"""
03 LSTM · PyTorch 版 —— 手推 BPTT 换成框架的循环层
====================================================
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

框架替你做了什么（与手搓版逐条对应）：
    手搓版                            PyTorch
    ------------------------------    --------------------------------------
    手写 lstm_step（8 行前向）        nn.LSTM(H) 一个层，时间维自动展开
    手推 BPTT（36 行反向，两条流）    loss.backward() 沿时间自动反传
    逐字符循环喂 x                   一次喂整段序列 (1, T, V)，框架内部沿时间跑
    梯度裁剪 np.clip(*, -5, 5)        p.grad.clamp_(-5, 5)（同款逐元素裁剪）
    Adam 手写 17 行                   torch.optim.Adam

⚠️ 一个真实的坑（也是本版最值得看的教学点）：
    PyTorch 的 LSTM 把四个门按 **[输入 i, 遗忘 f, 候选 g, 输出 o]** 排列；
    本仓手搓版用 **[遗忘 f, 输入 i, 输出 o, 候选 g]**。灌权重时必须按行重排
    （见 GATE_PERM），否则四个门张冠李戴——模型看起来能跑，但语义全错。

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 03_lstm\03_lstm_pytorch.py
预期:  梯度检查 < 1e-6;  loss 从 2.56 降到 0.05 以下;  逐字默写全句
产物:  本文件夹 figs_pytorch/ 下 4 张参考图（与 TF 版同名，可并排对照）:
       fig1 LSTM单元图 | fig2 BPTT展开图 | fig3 训练曲线 | fig4 门活动热图
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)

# ---------- 数据（与 TF 版逐位相同：同一 numpy 种子） ----------
SENT = "attention is all you need. "
data = SENT * 400
chars = sorted(set(data))
V = len(chars)
stoi = {ch: k for k, ch in enumerate(chars)}
itos = {k: ch for ch, k in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]

H = 96          # 隐状态维度


# ---------- 初始权重：按手搓版的随机流顺序抽（两版同起点） ----------
def init_weights_numpy():
    W = rng.normal(0, 0.1, (4 * H, V))       # 输入 -> 4个门（手搓版门序 f,i,o,g）
    U = rng.normal(0, 0.1, (4 * H, H))       # h_prev -> 4个门
    b = np.zeros(4 * H)
    b[:H] = 1.0                              # 遗忘门偏置置1: 初期倾向于"别忘"
    Why = rng.normal(0, 0.1, (V, H))         # h -> 词表打分
    by = np.zeros(V)
    return W, U, b, Why, by


W, U, b, Why, by = init_weights_numpy()

# 手搓版门序 [f,i,o,g] → PyTorch 门序 [i,f,g,o] 的行重排（每个门 H 行）
GATE_PERM = np.concatenate([np.arange(H, 2 * H), np.arange(0, H),
                            np.arange(3 * H, 4 * H), np.arange(2 * H, 3 * H)])


class LSTMLM(nn.Module):
    """LSTM + 输出头。名字里的 LM = language model。"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=V, hidden_size=H, batch_first=True).double()
        self.head = nn.Linear(H, V).double()
        with torch.no_grad():
            # 灌手搓版抽的权重：四门按 PyTorch 顺序重排（这是本版最关键的 4 行）
            self.lstm.weight_ih_l0.copy_(torch.tensor(W[GATE_PERM]))
            self.lstm.weight_hh_l0.copy_(torch.tensor(U[GATE_PERM]))
            self.lstm.bias_ih_l0.copy_(torch.tensor(b[GATE_PERM]))
            self.lstm.bias_hh_l0.zero_()                # PyTorch 把偏置拆成两份，这里全放 bias_ih
            self.head.weight.copy_(torch.tensor(Why))
            self.head.bias.copy_(torch.tensor(by))


model = LSTMLM()


def batch(xs, ys):
    """索引序列 → one-hot 张量（与手搓版的 x 逐位一致）。"""
    X = torch.zeros(len(xs), V, dtype=torch.float64)
    X[torch.arange(len(xs)), torch.tensor(xs)] = 1.0
    return X.unsqueeze(0), torch.tensor(ys)          # (1, T, V), (T,)


def loss_on(xs, ys):
    X_t, Y_t = batch(xs, ys)
    out, _ = model.lstm(X_t)                          # 一次喂整段，框架内部沿时间展开
    logits = model.head(out).squeeze(0)               # (T, V)
    return F.cross_entropy(logits, Y_t)               # 种子 p-onehot 藏在这里的反向里


# ---------- 数值梯度检查: 质检员上岗（autograd vs 差分） ----------
GC_S, GC_T = 123, 8                                   # 固定一段语料，两版用同一段
chunk = encode(data[GC_S:GC_S + GC_T + 1])
gc_xs, gc_ys = chunk[:-1], chunk[1:]
loss = loss_on(gc_xs, gc_ys)
loss.backward()

eps = 1e-5
worst = 0.0
named_params = [("W(ih)", model.lstm.weight_ih_l0, W[GATE_PERM]),
                ("U(hh)", model.lstm.weight_hh_l0, U[GATE_PERM]),
                ("b", model.lstm.bias_ih_l0, b[GATE_PERM]),
                ("Why", model.head.weight, Why),
                ("by", model.head.bias, by)]
for name, prm, _ in named_params:
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
        scaled = abs(ana - num) / max(1.0, abs(ana), abs(num))   # 大梯度看相对, 小梯度看绝对
        worst = max(worst, scaled)
print("grad check  max scaled error = %.2e" % worst)
assert worst < 1e-6, "autograd 的 BPTT 推错了?!"
print("grad check passed\n")


# ---------- 训练（Adam + 梯度裁剪） ----------
def train(iters=3000, T=32):
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    print("V=%d, 瞎猜下限 ln(V) = %.3f" % (V, np.log(V)))
    losses = []
    for it in range(1, iters + 1):
        s = rng.integers(0, len(data) - T - 1)        # 与 TF 版同一随机流抽起点
        chunk = encode(data[s:s + T + 1])
        xs, ys = chunk[:-1], chunk[1:]
        opt.zero_grad()
        loss = loss_on(xs, ys)
        loss.backward()
        with torch.no_grad():                         # 梯度裁剪: 防 BPTT 连乘爆炸
            for prm in model.parameters():
                prm.grad.clamp_(-5.0, 5.0)            # 与手搓版同款逐元素裁剪
        opt.step()
        losses.append(loss.item())
        if it % 200 == 0 or it == 1:
            print("step %5d  loss %.4f  ppl %.2f" % (it, loss.item(), np.exp(loss.item())))
    return losses


# ---------- 生成 ----------
def sample(seed="a", n=40):
    model.eval()
    hx = torch.zeros(1, 1, H, dtype=torch.float64)
    cx = torch.zeros(1, 1, H, dtype=torch.float64)
    x = torch.zeros(1, 1, V, dtype=torch.float64)
    x[0, 0, stoi[seed]] = 1.0
    out = []
    with torch.no_grad():
        for _ in range(n):
            o, (hx, cx) = model.lstm(x, (hx, cx))     # 框架内部就是"传送带带过去"
            idx = int(model.head(o.squeeze(0)).argmax(-1))    # 贪心解码
            out.append(itos[idx])
            x = torch.zeros(1, 1, V, dtype=torch.float64)
            x[0, 0, idx] = 1.0
    model.train()
    return seed + "".join(out)


# ---------- 门活动提取（框架不直接暴露门值，按框架门序重算一遍） ----------
def gates_on(sentence):
    """返回 (f, i, o) 三个 (H, len) 矩阵——做 fig4 热图用。

    做法：用模型里已训练好的权重，按 PyTorch 自己的门顺序 [i,f,g,o] 手算一遍
    每一步的四个门。这段"手工重算"恰好也是理解 LSTM 前向方程的最好练习。
    """
    Hsh = H
    Wih = model.lstm.weight_ih_l0.detach().numpy()      # (4H, V) 门序 [i,f,g,o]
    Whh = model.lstm.weight_hh_l0.detach().numpy()
    bih = model.lstm.bias_ih_l0.detach().numpy()
    h = np.zeros(Hsh); c = np.zeros(Hsh)
    Fs, Is, Os = [], [], []
    for k in encode(sentence):
        x = np.zeros(V); x[k] = 1.0
        a = Wih @ x + Whh @ h + bih                     # 4H 维预激活
        gi = 1 / (1 + np.exp(-a[0:Hsh]))                # i
        gf = 1 / (1 + np.exp(-a[Hsh:2 * Hsh]))          # f
        gg = np.tanh(a[2 * Hsh:3 * Hsh])                # g
        go = 1 / (1 + np.exp(-a[3 * Hsh:4 * Hsh]))      # o
        c = gf * c + gi * gg
        h = go * np.tanh(c)
        Fs.append(gf); Is.append(gi); Os.append(go)
    return np.array(Fs).T, np.array(Is).T, np.array(Os).T


# ================= 参考图 =================
def save_figures(losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_pytorch")
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

    ax.annotate("", xy=(9.2, 6.3), xytext=(0.9, 6.3),
                arrowprops=dict(arrowstyle="-|>", color="#2e8b57", lw=2.8))
    ax.text(0.55, 6.55, "c(t-1)", fontsize=11, color="#2e8b57")
    ax.text(9.15, 6.55, "c(t)", fontsize=11, color="#2e8b57")
    circ(2.2, 6.3, "×f")
    circ(4.9, 6.3, "+")
    rbox(2.0, 3.4, 1.9, 0.9, "遗忘门 f=σ(·)\n留多少旧记忆", "#dbe9ff")
    rbox(4.15, 3.4, 1.9, 0.9, "输入门 i=σ(·)\n写多少新内容", "#ffe9c9")
    rbox(6.3, 3.4, 1.9, 0.9, "候选 g=tanh(·)\n要写入的内容", "#fff9c4")
    rbox(8.5, 3.4, 1.9, 0.9, "输出门 o=σ(·)\n向外露出多少", "#e4d9f7")
    rbox(2.6, 1.0, 2.4, 0.75, "x(t)   h(t-1)", "#ececec")
    for gx in [2.0, 4.15, 6.3, 8.5]:
        arr((2.9, 1.42), (gx, 2.9), "#888888")
    arr((2.05, 3.9), (2.18, 5.95), "#2e8b57")
    arr((5.0, 3.9), (4.92, 5.95), "#2e8b57")
    ax.text(5.3, 5.15, "i⊙g", fontsize=9, color="#2e8b57")
    arr((7.35, 6.3), (7.35, 5.55), "#3b6fd4", lw=1.8)
    circ(7.35, 5.2, "tanh")
    circ(8.6, 5.2, "⊙o")
    arr((7.62, 5.2), (8.32, 5.2), "#3b6fd4")
    arr((8.6, 3.9), (8.6, 4.9), "#3b6fd4")
    arr((8.88, 5.2), (9.35, 5.2), "#3b6fd4")
    ax.text(9.6, 5.2, "h(t)", fontsize=11, color="#3b6fd4")
    ax.text(0.3, 0.12,
            "传送带: c(t) = f⊙c(t-1) + i⊙g\n"
            "读数:   h(t) = o⊙tanh(c(t))\n"
            "在 nn.LSTM 里：这就是它内部一步做的事（门序 [i,f,g,o] 由框架定）",
            fontsize=10, linespacing=1.6, va="bottom")
    ax.set_title("LSTM 单元：一条传送带（绿）+ 三个学习出来的开关（PyTorch 版）", fontsize=12)
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
        ax.text(xc, 1.3, "L", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="circle,pad=0.25", fc="#ffd9d2", ec="#d4402e", lw=1.1))
        arr((xc, 2.6), (xc, 1.62), "#3b6fd4", lw=1.3)
        if i < 3:
            nxt = xs4[i + 1]
            arr((xc + 0.9, 3.2), (nxt - 0.9, 3.2), "#3b6fd4", lw=1.8)
            arr((xc + 0.9, 4.75), (nxt - 0.9, 4.75), "#2e8b57", lw=2.2)
            arr((nxt - 0.9, 3.75), (xc + 0.9, 3.75), "#d4402e", lw=1.6, ls="--")
            arr((nxt - 0.9, 4.25), (xc + 0.9, 4.25), "#e07b00", lw=1.6, ls="--")
    ax.text(2.95, 3.42, "h →", fontsize=9, color="#3b6fd4")
    ax.text(2.95, 4.92, "c →", fontsize=9, color="#2e8b57")
    ax.text(5.9, 3.9, "dh 反向", fontsize=9, color="#d4402e")
    ax.text(5.9, 4.42, "dc 反向（乘 f 传回）", fontsize=9, color="#e07b00")
    arr((11.35, 1.3), (1.0, 1.3), "#d4402e", lw=1.6, ls="--")
    ax.text(6.2, 0.85, "每一步的 L 都有自己的种子: dlogits = p − onehot(y)",
            fontsize=9.5, color="#d4402e", ha="center")
    ax.text(11.6, 1.3, "反向传播\n从这里开始", fontsize=9, color="#d4402e", va="center")
    ax.set_title("BPTT：前向（实线）从左到右，反向（虚线）由 backward() 自动倒着走", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig2_bptt_unroll.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 训练曲线 ----
    steps = np.arange(1, len(losses) + 1)
    losses_arr = np.array(losses)
    smooth = np.convolve(losses_arr, np.ones(51) / 51, mode="valid")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(steps, losses_arr, color="#c44e52", alpha=0.25, lw=0.8)
    ax.plot(steps[25:-25], smooth, color="#c44e52", lw=2, label="滑动平均(51步)")
    ax.axhline(np.log(V), color="gray", ls="--", lw=1.4)
    ax.text(2400, np.log(V) + 0.06, "瞎猜下限 ln(V) = %.3f" % np.log(V), fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数"); ax.set_ylabel("loss（对数轴）")
    ax.set_title("LSTM 训练曲线（PyTorch）：从瞎猜到逐字默写")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig3_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 门活动热图（权重仍来自训练好的模型） ----
    Fs, Is, Os = gates_on(SENT)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 5.8), sharex=True)
    for axx, M, name in zip(axes, [Fs, Is, Os],
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
    losses = train()
    print("\ntarget :", SENT.strip())
    print("lstm   :", sample().strip())
    save_figures(losses)
