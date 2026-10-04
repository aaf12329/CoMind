# 机器学习基础：从线性回归到 Transformer，双框架对照版（PyTorch × TensorFlow）

> 每个模型**两份代码**：PyTorch 版 + TensorFlow 版。同数据、同随机种子、同超参、同起点——
> 数字对得上（收敛值一字不差），图能并排看（同名图放各自的 figs 目录）。
> 两份代码的差别，恰好就是"两个框架的差别"。
>
> 配套阅读：论文《Attention Is All You Need》(arXiv:1706.03762)
> 环境：conda 环境 `yolo`（已装 torch 2.11 + tensorflow 2.21；这些小题都跑 CPU）。
> 每个模型一个文件夹；**进每个文件夹先读它的 README.md**：是什么、两份代码怎么对照、
> 图怎么读、自测题都在里面。

## 运行方式

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_tensorflow.py
```

每次运行会完整重训一遍模型，并**重新生成对应 `figs_pytorch/` 或 `figs_tensorflow/`
下的全部参考图**——图不是贴上去的素材，全部由代码从真实训练中画出来，与代码永远同步。

> **重建说明**：本仓 1.0 是纯 numpy 手搓版（推导过程在 git 历史里，`git log` 可查）；
> 2.0 起改为双框架对照版——numpy 版退役，但"手推公式 → 框架自动求导"的对照关系
> 保留在每个脚本的头注释和 README 的对照表里。

## 路线图（一个模型一个 commit）

| 文件夹 | 模型 | 任务 | 两份代码教什么（框架差异） | 状态 |
|---|---|---|---|---|
| 01_linear_regression | 线性回归 | 拟合 y=3x+1+噪声 | autograd / GradientTape：反向传播交给框架的第一课；`.grad` 累加 vs 记录带新建 | ✅ |
| 02_mlp | 多层感知机 | 两月牙二分类 | nn.Module / Keras 层封装；优化器接管更新 | 🔨 重建中 |
| 03_lstm | LSTM | 字符级语言模型 | 框架自带的 RNN 层与手写 BPTT 的关系 | 🔨 重建中 |
| 04_mini_transformer | 迷你 Transformer | 同 03，对照实验 | 注意力层的框架实现；因果 mask 怎么写 | 🔨 重建中 |
| 05_audio_models | 音频模型架构 | 可运行骨架（不训练） | 把 12 个经典架构在两种框架里搭出来 | 🔨 重建中 |
| 06_tree_models | 树模型家族 | 分类/回归 | 框架不是只能训神经网络：soft tree（可微版）与张量化经典算法 | 🔨 重建中 |

## 贯穿全系列的三条军规

1. **反向传播永远从损失开始**：前向的最后一个节点（损失）就是反向的第一个节点。
   种子梯度对回归是 `ŷ−y`，对 softmax+交叉熵是 `p−onehot(y)`——
   手搓时代要亲手推导，框架时代藏在 `backward()` / `tape.gradient()` 里，
   **不用写它，但必须知道它还在那儿**。
2. **梯度永远可以被差分质疑**：用 `(L(θ+ε)−L(θ−ε))/2ε` 当质检员——
   手搓时代查手推公式，双框架时代查 autograd / tape 的结果，每个章节都保留这个环节
   （判据沿用"大梯度看相对、小梯度看绝对"，纯相对判据会在梯度≈0 的条目上假报）。
3. **损失必须从 ln(V) 出发**：语言模型训练前 loss ≈ ln(词表大小) = 瞎猜水平，
   偏离这个起点说明前向有 bug。

## 图怎么读

每个模型有**两份同名参考图**（`figs_pytorch/` 与 `figs_tensorflow/`），并排看：

- 能对上的是**数学**（loss 曲线几乎重合、收敛值一致）；
- 对不上的地方是**工具的性格**（随机初始化、执行顺序、数值精度……）。

各文件夹 README.md 里有逐张图的读法说明。

## 已知现象（不是 bug）

- 04 从单字符 "a" 生成会得到相位偏移的句子（语料循环出现、训练块起点随机，
  模型无法区分这是句首的 a 还是 all 里的 a）；给全前缀 "attention is " 则确定性续写。
- 训练曲线中途偶有回弹：每步是从语料里随机截的一段，难段 loss 天然高。
- Windows 原生 TensorFlow 只有 CPU 版（GPU 仅 WSL2）；这些题很小，CPU 足够。
