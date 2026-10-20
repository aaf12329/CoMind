# 01 线性回归 —— 两个框架的第一块砖

## 这是什么

同一道题、两份代码：**PyTorch 版** 与 **TensorFlow 版**。
模型简单到不配叫"模型"（`ŷ = w·x + b`，两个参数），但**反向传播的全部思想已经
装在里面了**——后面所有章节，本质上都是这一块的放大版。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。两份代码的差别，恰好就是"两个框架的差别"。

## 它教你的三件事（在框架里分别长什么样）

1. **损失函数**：`L = mean(½(ŷ−y)²)`，衡量"预测得有多离谱"。乘 ½ 纯粹是为了
   求导时把平方的 2 抵掉。PyTorch 的 `MSELoss` **不带**这个 ½——两份代码里都
   乘了个 0.5 把它补回来，梯度才与公式一格不差。
2. **反向传播的种子**：`dL/dŷ = ŷ − y`。手搓时它要你亲手推导；到了框架里，
   这粒种子藏在 `backward()` / `tape.gradient()` 的自动实现里——**代码不用写它，
   但你必须知道它还在那儿**。
3. **数值梯度检查**：用 `(L(w+ε)−L(w−ε))/2ε` 做差分当质检员。
   两版都保留了这个传统：**autograd/tape 算的梯度 vs 差分**，误差 <1e-6 断言，
   框架也会算错的时候（用错了 API），差分不会骗人。

## 两份代码逐环节对照

| 环节 | PyTorch 版 | TensorFlow 版 |
|---|---|---|
| 参数 | `nn.Linear(1,1)`（内部就是 w、b） | `tf.Variable(0.0)` × 2 |
| 前向 | `model(x)` | `w * x + b` |
| 损失 | `0.5 * nn.MSELoss()` | `0.5 * tf.reduce_mean((ŷ−y)**2)` |
| 求梯度 | `loss.backward()` → 梯度进 `.grad` | `with tf.GradientTape() as t` → `t.gradient(loss,[w,b])` |
| 更新参数 | `optimizer.step()` | `optimizer.apply_gradients([(gw,w),(gb,b)])` |
| **机制差异** | `.grad` **会累加**，每轮必须先 `zero_grad()` | tape 每轮 **新建一张记录带**，天然不累加，没有清零这一步 |

## 怎么跑

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 01_linear_regression\01_linear_regression_tensorflow.py
```

两版**同数据、同随机种子、同起点 (0,0)**：都训练 300 步、lr=0.1，
都收敛到 **w = 2.981（真值 3.0）、b = 0.976（真值 1.0）**——数值一模一样，
这就是"数学一样、工具不同"的字面证据。

> Windows 原生 TensorFlow 只有 CPU 版（GPU 仅 WSL2）；这些题很小，CPU 足够，
> PyTorch 版同样跑 CPU，两边可比。

## figs 参考图怎么读

两份代码各自生成同名 4 张图：`figs_pytorch/` 与 `figs_tensorflow/`——**并排看**。

| 图 | 看什么 |
|---|---|
| fig1_data_vs_fit | 模型学到的直线 vs 真实规律 vs 带噪声的数据——"模型只能学趋势，吃不掉噪声" |
| fig2_loss_curve | loss 对数轴下降；**两个框架的曲线几乎重合**（同种子同数据） |
| fig3_loss_surface_gd | **本文件夹最值得看的图**：损失面等高线上那条橙色 GD 轨迹——“下山”第一次有了具象 |
| fig4_backprop_diagram | 前向（蓝）手写、反向（红）**由框架自动完成**——红段就是框架替你做的事 |

## 自测题

1. 为什么 loss 永远降不到 0？（答：数据自带 0.3 的噪声，那是模型吃不掉的部分）
2. 为什么损失函数要乘 ½？PyTorch 的 `MSELoss` 少了它，两份代码里是怎么补的？
3. PyTorch 版里 `zero_grad()` 能删掉吗？TensorFlow 版为什么没有对应的一步？（动手试：删掉看 loss 怎么炸）
4. 把学习率调到 10 会发生什么？两版都试，看谁先发散。
5. `fig4` 红色那段箭头，在 PyTorch 和 TF 里分别对应哪一行代码？
