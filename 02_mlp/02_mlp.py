# -*- coding: utf-8 -*-
r"""
02 多层感知机 MLP —— 给反向传播加一层"深度"
==============================================
任务: 两月牙二分类 —— 线性回归画一条直线分不开的两个月牙,
      中间夹一个 tanh 隐层就能分开。这就是"神经网络"三个字的全部含义:
      线性变换 -> 非线性 -> 再线性变换, 深度产生了非线性边界。

模型:  h = tanh(X @ W1 + b1)          X:(N,2)  W1:(2,32)  h:(N,32)
       logits = h @ W2 + b2           W2:(32,2)
       P = softmax(logits),  L = mean(-log P[正确类])

手推梯度(链式法则开始跨层了, 从损失倒着走):
    dlogits = (P - onehot(y)) / N        <- 种子, 和 01 里的 (y_hat-y) 同一角色
    dW2 = h.T @ dlogits;   db2 = dlogits.sum(0)
    dh  = dlogits @ W2.T
    dz1 = dh * (1 - h*h)                 <- tanh 的导数
    dW1 = X.T @ dz1;       db1 = dz1.sum(0)

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 02_mlp\02_mlp.py
预期:  梯度检查 < 1e-6;  训练后准确率 > 95%
产物:  本文件夹 figs/ 下生成 4 张参考图:
       fig1 月牙数据 | fig2 决策边界(MLP弯 vs 线性直) | fig3 网络结构+反向链 | fig4 loss/acc曲线
"""
import os
import numpy as np

rng = np.random.default_rng(0)

# ---------- 造数据: 两个交错的半圆 ----------
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

p = {
    'W1': rng.normal(0, 0.5, (DX, H)), 'b1': np.zeros(H),
    'W2': rng.normal(0, 0.5, (H, 2)),  'b2': np.zeros(2),
}
Y1h = np.zeros((N, 2)); Y1h[np.arange(N), Y] = 1.0


def forward(p):
    z1 = X @ p['W1'] + p['b1']
    h = np.tanh(z1)
    logits = h @ p['W2'] + p['b2']
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    P = e / e.sum(1, keepdims=True)
    L = -np.mean(np.log(P[np.arange(N), Y] + 1e-12))
    return L, (z1, h, P)


def backward(p, cache):
    z1, h, P = cache
    grads = {}
    dlogits = (P - Y1h) / N                     # 反向传播的种子
    grads['W2'] = h.T @ dlogits
    grads['b2'] = dlogits.sum(0)
    dh = dlogits @ p['W2'].T
    dz1 = dh * (1 - h * h)                      # tanh 的导数
    grads['W1'] = X.T @ dz1
    grads['b1'] = dz1.sum(0)
    return grads


def accuracy(p):
    _, (_, _, P) = forward(p)
    return np.mean(P.argmax(1) == Y)


# ---------- 数值梯度检查 ----------
def num_grad(p, k, idx, eps=1e-5):
    old = p[k][idx]
    p[k][idx] = old + eps; lp, _ = forward(p)
    p[k][idx] = old - eps; lm, _ = forward(p)
    p[k][idx] = old
    return (lp - lm) / (2 * eps)

_, cache = forward(p)
ana = backward(p, cache)
worst = 0.0
for k in p:
    n = p[k].size
    for j in [0, n // 2, n - 1]:          # 每个参数张量抽 3 个元素查
        idx = np.unravel_index(j, p[k].shape)
        num = num_grad(p, k, idx)
        rel = abs(ana[k][idx] - num) / max(abs(ana[k][idx]), abs(num), 1e-8)
        worst = max(worst, rel)
print("grad check  max relative error = %.2e" % worst)
assert worst < 1e-6, "手推的梯度错了!"
print("grad check passed\n")


# ---------- Adam(和论文 5.3 节同款, 手写就这几行) ----------
class Adam:
    def __init__(self, params, lr=0.05):
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


opt = Adam(p)
hist = []                               # (loss, accuracy), 画图用
for step in range(1, 1501):
    L, cache = forward(p)
    grads = backward(p, cache)
    opt.step(p, grads)
    hist.append((L, accuracy(p)))
    if step % 250 == 0 or step == 1:
        print("step %4d  loss %.4f  accuracy %.1f%%" % (step, L, 100 * accuracy(p)))

print("\nfinal accuracy: %.1f%%   (线性模型的极限约 85%%, 隐层把边界掰弯了)" % (100 * accuracy(p)))


# ================= 参考图 =================
def save_figures(p, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
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
    # 线性基线: 同样的 softmax 交叉熵, 但没有隐层 -> 只能画出直边界
    pl = {'W': rng.normal(0, 0.5, (2, 2)), 'b': np.zeros(2)}
    optl = Adam(pl)
    for _ in range(400):
        logits = X @ pl['W'] + pl['b']
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits); P = e / e.sum(1, keepdims=True)
        dlogits = (P - Y1h) / N
        optl.step(pl, {'W': X.T @ dlogits, 'b': dlogits.sum(0)})

    gx = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 300)
    gy = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 300)
    XX, YY = np.meshgrid(gx, gy)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)

    def prob_mlp(Gg):                    # 属于类1的概率
        hh = np.tanh(Gg @ p['W1'] + p['b1'])
        lg = hh @ p['W2'] + p['b2']
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        return (e / e.sum(1, keepdims=True))[:, 1]

    lg = G @ pl['W'] + pl['b']
    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    Plin = (e / e.sum(1, keepdims=True))[:, 1]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.contourf(XX, YY, prob_mlp(G).reshape(XX.shape), levels=[0, 0.5, 1],
                colors=["#aec7e8", "#ff9896"], alpha=0.35)
    ax.contour(XX, YY, prob_mlp(G).reshape(XX.shape), levels=[0.5],
               colors="crimson", linewidths=2.2)
    ax.contour(XX, YY, Plin.reshape(XX.shape), levels=[0.5],
               colors="gray", linestyles="--", linewidths=2.0)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=12, color="#4878b0", label="类 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=12, color="#d65f5f", label="类 1")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="crimson", lw=2.2, label="MLP 边界（弯的）"),
               Line2D([], [], color="gray", ls="--", lw=2.0, label="线性基线边界（直的）")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], loc="upper left", fontsize=9)
    ax.set_title("决策边界：隐层把直线掰弯了")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(figdir, "fig2_decision_boundary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 网络结构 + 反向传播链 ----
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def node(cx, cy, r=0.13, fc="#ffffff"):
        ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec="#444444", lw=1.1))

    def arr(p1, p2, color="#3b6fd4", lw=1.1, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    # 三列: 输入2 -> 隐藏32(画6个示意) -> 输出2
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
    ax.text(8.6, 4.6, "反向传播（从损失倒着走）:", fontsize=10, color="#d4402e", va="top")
    ax.text(8.6, 4.0,
            "种子  dlogits = (P − y)/N\n"
            "→ dW2 = h^T · dlogits\n"
            "→ dh  = dlogits · W2^T\n"
            "→ dz1 = dh⊙(1 − h²)   ← tanh′\n"
            "→ dW1 = X^T · dz1",
            fontsize=9, color="#d4402e", va="top", linespacing=1.5)
    arr((7.2, 0.85), (1.2, 0.85), "#d4402e", lw=2, ls="--")
    ax.set_title("MLP：前向（蓝）一层层算，反向（红）一层层退", fontsize=12)
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
    ax.set_title("训练曲线：loss 降、accuracy 升")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(figdir, "fig4_loss_acc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


save_figures(p, hist)
