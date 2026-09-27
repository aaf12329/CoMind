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

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 02_mlp.py
预期:  梯度检查 < 1e-6;  训练后准确率 > 95%
"""
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
for step in range(1, 1501):
    L, cache = forward(p)
    grads = backward(p, cache)
    opt.step(p, grads)
    if step % 250 == 0 or step == 1:
        print("step %4d  loss %.4f  accuracy %.1f%%" % (step, L, 100 * accuracy(p)))

print("\nfinal accuracy: %.1f%%   (线性模型的极限约 85%%, 隐层把边界掰弯了)" % (100 * accuracy(p)))
