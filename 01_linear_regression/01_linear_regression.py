# -*- coding: utf-8 -*-
r"""
01 线性回归 —— 一切从这里开始
==============================
模型:  y_hat = w*x + b
损失:  L = mean( 1/2*(y_hat - y)^2 )          乘 1/2 是为了让导数干净
手推梯度(反向传播的全过程, 就这么多):
    dL/d_yhat = (y_hat - y)        <- 反向传播的"种子", 整个系列里它反复出现
    dL/d_w    = mean( (y_hat - y) * x )
    dL/d_b    = mean( (y_hat - y) )
数值梯度检查:
    用 (L(w+e)-L(w-e)) / 2e 直接对损失做差分, 和手推结果对照。
    手推错了当场现形 —— 这套质检流程后面每个模型都会用。

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 01_linear_regression.py
预期:  梯度检查误差 < 1e-6;  w, b 收敛到 3.0 和 1.0
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- 造数据: y = 3x + 1 + 噪声 ----------
N = 200
x = rng.uniform(-2, 2, N)
y = 3.0 * x + 1.0 + rng.normal(0, 0.3, N)

w, b = 0.0, 0.0
lr = 0.1


def loss(w, b):
    return 0.5 * np.mean((w * x + b - y) ** 2)


def grads(w, b):
    """手推的解析梯度。误差项 e = y_hat - y 就是反向传播的种子。"""
    e = w * x + b - y
    return np.mean(e * x), np.mean(e)


# ---------- 数值梯度检查: 质检员上岗 ----------
eps = 1e-5
num_dw = (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps)
num_db = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
gw, gb = grads(w, b)
print("grad check  dw: manual %.8f | numeric %.8f" % (gw, num_dw))
print("grad check  db: manual %.8f | numeric %.8f" % (gb, num_db))
assert abs(gw - num_dw) < 1e-6, "手推的梯度错了!"
print("grad check passed\n")

# ---------- 梯度下降 ----------
for step in range(1, 301):
    gw, gb = grads(w, b)
    w -= lr * gw
    b -= lr * gb
    if step % 50 == 0 or step == 1:
        print("step %3d  loss %.4f  w %.4f  b %.4f" % (step, loss(w, b), w, b))

print("\nfinal:  w = %.4f (true 3.0)   b = %.4f (true 1.0)" % (w, b))
print("注意: loss 不会降到 0 —— 数据自带 0.3 的噪声, 模型只能学到趋势, 吃不掉噪声。")
