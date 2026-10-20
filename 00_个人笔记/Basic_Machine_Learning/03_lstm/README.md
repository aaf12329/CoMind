# 03 LSTM —— 沿时间反传的完整版（PyTorch × TensorFlow 双版本）

## 这是什么

同一道题、两份代码：**PyTorch 版** 与 **TensorFlow 版**。
字符级语言模型：看前文，预测下一个字符。语料是一句循环出现的话
"attention is all you need. "，训好后它能**逐字默写**出来——这是最硬的验收标准。

LSTM 是 RNN 治"健忘症"的方案：一条**细胞状态传送带 c** 贯穿时间，
三个**学习出来的开关门**（遗忘门 f、输入门 i、输出门 o）决定记忆留多少、写多少、露多少。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。

## 它教你的三件事（在框架里分别长什么样）

1. **BPTT（沿时间反传）**：种子 `dlogits = p − onehot(y)` 藏在交叉熵的反向里，
   时间倒序走回第一个字符，每一步两条梯度流（`dh` 经隐状态、`dc` 经传送带乘 f 跨步）。
   手搓版要写 36 行反向；框架里 `nn.LSTM` / `keras.layers.LSTM` 一个层，
   `loss.backward()` / `tape.gradient` 一趟把 3000 步的时间链全走完。
2. **梯度裁剪**：BPTT 的连乘会爆炸，逐元素裁剪 `±5` 是 RNN 时代祖传保命操作。
   手搓 `np.clip` → PyTorch `p.grad.clamp_` → TF `tf.clip_by_value`，语义一字不差。
3. **梯度检查的坑 + 门序的坑**：
   - 判据沿用"大梯度看相对、小梯度看绝对"：`|a−n|/max(1,|a|,|n|)`（纯相对判据会在梯度≈0 处假报）；
   - **门序**：手搓版门序 `[f,i,o,g]`，PyTorch 是 `[i,f,g,o]`、Keras 是 `[i,f,c,o]`——
     灌权重必须按行重排（`GATE_PERM`）。忘了重排模型**照样能训**，只是语义张冠李戴、
     两版对不上——这种"看起来对"的错误，只能靠数字对照查出来。

## 两份代码逐环节对照

| 环节 | PyTorch 版 | TensorFlow 版 |
|---|---|---|
| 循环层 | `nn.LSTM(V, H, batch_first=True)` | `keras.layers.LSTM(H, return_sequences=True)` |
| 喂法 | 一次喂整段 `(1, T, V)` one-hot，框架内部沿时间展开（两版相同） | 同 |
| 门顺序 | `[i, f, g, o]` | `[i, f, c, o]` |
| **灌权重** | `W[门序重排]` **直接用**（torch 存 (出,入)，与手搓 W 同形） | `W[门序重排].T`（Keras kernel 存 (入,出)） |
| 偏置 | 拆成 `bias_ih + bias_hh` 两份（全放前者） | 单个 `bias` |
| BPTT | `loss.backward()` | `with tf.GradientTape() as t` → `t.gradient` |
| 裁剪 | `prm.grad.clamp_(-5, 5)` | `tf.clip_by_value(g, -5, 5)` |
| Adam | `torch.optim.Adam(lr=0.02)` | `keras.optimizers.Adam(lr=0.02, epsilon=1e-8)`（eps 对齐） |
| **踩坑记录** | 权重可 `.copy_()` 直灌 | Keras 3 的层权重在 `.weights` 列表里（没暴露 `.kernel` 属性）→ 按索引访问 + 形状断言钉死顺序 |

## 怎么跑

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 03_lstm\03_lstm_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 03_lstm\03_lstm_tensorflow.py
```

两版**同数据、同随机种子、同初始权重**（按手搓版的随机流顺序抽取）：

- 梯度检查：PyTorch **2.4e-11**、TensorFlow **3.6e-11**（都是机器精度级）；
- 训练 3000 步（Adam，lr=0.02，逐元素裁剪 ±5）：loss 从 **2.56** 降到 **0.0001** 附近
  （末步 ~0.012 是"随机截到难段"的天然回弹）；
- 结束时两版都逐字默写："attention is all you need. "
- **一个如实的细节**：两版日志在前几百步几乎逐行一致，长训练后有**浮点级微漂移**
  （如 step 3000：0.0116 vs 0.0120）——框架的矩阵乘/归约**实现顺序不同**，浮点加法
  不满足结合律。这不是 bug，是"数学相同、浮点实现不同"的物理事实。

## figs 参考图怎么读

两份代码各自生成同名 4 张图：`figs_pytorch/` 与 `figs_tensorflow/`——**并排看**。

| 图 | 看什么 |
|---|---|
| fig1_lstm_cell | 单元解剖：绿色传送带 c + 三个门开关 + 读数路径 h = o⊙tanh(c) |
| fig2_bptt_unroll | 时间展开：实线前向 →，虚线反向 ←（dh 红流、dc 橙流两条） |
| fig3_loss_curve | 训练曲线 + 瞎猜下限 ln(V)=2.565 基准线——语言模型的通用质检 |
| fig4_gates_heatmap | **最值得看的图**：训好的三个门在整句话上的开合热图——门是学出来的，不是设计出来的 |

## 自测题

1. 为什么 c 这条传送带能治梯度消失？（答：c 每步只做"乘 f＋加新内容"，
   f 接近 1 时梯度几乎无衰减地跨步回流，不像 RNN 每步都乘一遍同一个 W）
2. 传送带断了会怎样？手工版可以删掉 `dc_next = dc ⊙ f` 那一行看梯度检查当场爆炸
   （在 git 历史的手搓版里做这个实验）；框架版对应的是"梯度流是否完整"由 autograd 保证。
3. 遗忘门偏置为什么初始化成 1？（答：让模型初期倾向"别忘"，先学会记住再学会取舍）
4. 如果灌权重时忘了 `GATE_PERM` 重排，模型还能训出低 loss 吗？为什么难发现？
   （答：能训——张冠李戴后仍是"某种合法的 LSTM"，只是与手搓版/另一框架对不上，
   且遗忘门偏置 1 的技巧用错了门；所以要用"两版日志/曲线能不能对上"当质检）
5. 两版曲线为什么长训练后有微小漂移？（答：矩阵乘的累加顺序不同，浮点加法不满足结合律；
   3000 步把 1e-16 的差放大了——不是 bug）
