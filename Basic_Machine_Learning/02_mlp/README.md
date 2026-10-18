# 02 多层感知机 MLP —— "深度"二字的全部含义（PyTorch × TensorFlow 双版本）

## 这是什么

同一道题、两份代码：**PyTorch 版** 与 **TensorFlow 版**。
`线性变换 → tanh 非线性 → 再线性变换`，夹一层 32 个神经元的隐层。
任务：两月牙二分类——**一条直线永远分不开、一层隐层轻松分开**的教科书数据集。

神经网络 = 线性变换的叠加本来还是线性的，**中间不加非线性就等于什么都没加**。
tanh 就是那个把直线"掰弯"的东西，也是"深度"有价值的原因。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。

## 它教你的三件事（在框架里分别长什么样）

1. **链式法则跨层传递**：种子 `dlogits = (P − onehot(y))/N` 出发，先输出层（dW2）、
   穿过 tanh（乘 `1 − h²`）、再过输入层（dW1）——01 里一层链，这里两层接力。
   手搓时要写 6 行；框架里 `loss.backward()` / `tape.gradient` 一趟走完，
   **公式还是那套，只是不用再手写**。
2. **tanh 的导数**：`1 − h²`。每个激活函数都有自己的"过路费"。
   框架里写 `nn.Tanh()` / `activation="tanh"` 一个词，导数在反向里自动生效——
   换个激活（ReLU）不用改任何求导代码，这是封装的红利。
3. **Adam 优化器**：梯度下降的自动挡（动量 + 自适应步长）。手写要 17 行（m/v 两本账），
   `torch.optim.Adam` / `keras.optimizers.Adam` 一行。**注意一个暗坑**：
   Keras 的 Adam 默认 `epsilon=1e-7`，与论文/手写版的 `1e-8` 不同——
   两份代码里都显式写死 1e-8，保证与手搓版逐位对齐。

## 两份代码逐环节对照

| 环节 | PyTorch 版 | TensorFlow 版 |
|---|---|---|
| 参数 | `nn.Sequential(Linear(2,32), Tanh(), Linear(32,2))` | `keras.Sequential([Dense(32,'tanh'), Dense(2)])` |
| **权重形状** | `nn.Linear` 内部存 **(出,入)** → 灌权重要 `W1.T` | Keras kernel 就是 **(入,出)** → 直接灌 `W1`，不转置 |
| 前向 | `model(x)` | `model(x)` |
| 损失 | `F.cross_entropy(logits, y)`（内含 softmax+稳定化） | `tf.nn.sparse_softmax_cross_entropy_with_logits`（等价） |
| 求梯度 | `loss.backward()` | `with tf.GradientTape() as t` → `t.gradient(loss, trainable_variables)` |
| 更新 | `optimizer.step()`（每轮先 `zero_grad()`） | `optimizer.apply_gradients(zip(grads, vars))`（tape 天然不累加） |
| 初始权重 | 用同一 numpy 随机流抽 `W1,W2`，再 `copy_` 进层 | 同一随机流，再 `set_weights` |
| **踩坑记录** | —— | Keras 损失类默认按 **float32** 计算（dtype 策略），微小梯度在数值差分里会被量化成 0，梯度检查挂 → 改用底层算子吃 float64 |

## 怎么跑

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 02_mlp\02_mlp_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 02_mlp\02_mlp_tensorflow.py
```

两版**同数据、同随机种子、同初始权重**：训练 1500 步（Adam，lr=0.05），
**日志逐行一致**——step 1 都是 loss 1.1042 / acc 72.0%，step 1500 都是 loss 0.0219 / acc 99.0%。
这就是"数学一样、工具不同"的字面证据。

## figs 参考图怎么读

两份代码各自生成同名 4 张图：`figs_pytorch/` 与 `figs_tensorflow/`——**并排看**。

| 图 | 看什么 |
|---|---|
| fig1_moons_data | 两月牙数据本身——先看清楚为什么直线无能为力 |
| fig2_decision_boundary | **最值得看的图**：MLP 的弯边界（红）和线性基线的直边界（灰虚线）同屏对比 |
| fig3_network_backprop | 网络结构 + 右侧反向传播链公式——红段就是框架替你做的事 |
| fig4_loss_acc_curve | loss 降、accuracy 升的双轴训练曲线（两条曲线几乎重合） |

## 自测题

1. 去掉 tanh（或者把它换成另一个线性层）会怎样？
   （答：两层线性叠加还是线性，边界永远是直线，月牙分不开）
2. 为什么梯度检查对 b1/b2 这类一维向量也成立？（答：链式法则不关心形状）
3. 隐层加到 1024 个会更好吗？（答：训练集精度会上去，但可能把噪声也记住——深度不是免费午餐）
4. PyTorch 灌权重时要 `W1.T`、Keras 不用转置，谁对？（答：都对——只是两家"存权重的规矩"不同，
   转置只是把数据摆成对方要求的朝向；`fig3` 里的公式 `z1 = X·W1` 两边都没变）
5. 为什么 TF 版没用 `keras.losses.SparseCategoricalCrossentropy`？（答：它默认 float32 计算，
   差分质检过不了；底层算子按 float64 原样算——**封装越厚，精度黑箱越多，质检就越重要**）
6. 两版训练日志为什么能逐行一致？（答：同数据 + 同初始权重 + 同一套数学，包括 Adam 的 epsilon 都对齐了）
