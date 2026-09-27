# dl-from-scratch：从线性回归到 Transformer，纯 numpy 手搓

> 配套阅读：论文《Attention Is All You Need》(arXiv:1706.03762)
> 环境：只需要 numpy，无任何深度学习框架。每个 .py 文件自包含，可单独运行。

## 运行方式

```bash
D:\Anaconda\envs\imooc_ai\python.exe 01_linear_regression.py
```

## 路线图（一个模型一个 commit）

| 文件 | 模型 | 任务 | 学到什么 |
|---|---|---|---|
| 01_linear_regression.py | 线性回归 | 拟合 y=3x+1+噪声 | 反向传播的种子：dL/dŷ = ŷ−y；数值梯度检查 |
| 02_mlp.py | 多层感知机 | 两月牙二分类 | 隐层+非线性；链式法则跨层传递；Adam |
| 03_lstm.py | LSTM（手推 BPTT） | 字符级语言模型 | 门控记忆；沿时间反传；梯度裁剪 |
| 04_mini_transformer.py | 迷你 Transformer | 同 03，对照实验 | Q/K/V；因果 mask；注意力没有时间循环 |

## 贯穿全系列的三条军规

1. **反向传播永远从损失开始**：前向的最后一个节点（损失）就是反向的第一个节点，
   种子梯度对回归是 `ŷ−y`，对 softmax+交叉熵是 `p−onehot(y)`，一路乘回去。
2. **手推的梯度必须过数值梯度检查**：用 `(L(θ+ε)−L(θ−ε))/2ε` 当质检员，
   手推错误会当场现形。这套流程每个文件都在用。
3. **损失必须从 ln(V) 出发**：语言模型训练前 loss ≈ ln(词表大小) = 瞎猜水平，
   偏离这个起点说明前向有 bug。

## 每个文件的统一验收标准

- 梯度检查：手推梯度 vs 数值梯度，最大相对误差 < 1e-5（平滑激活）或 < 1e-4（含 ReLU）
- 训练曲线：loss 从 ln(V) ≈ 2.56 单调下降
- 终极验收：模型逐字默写出 "attention is all you need."（03、04）
