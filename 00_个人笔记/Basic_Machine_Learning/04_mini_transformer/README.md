# 04 迷你 Transformer —— 《Attention Is All You Need》最小可训练版（PyTorch × TensorFlow 双版本）

## 这是什么

同一道题、两份代码：**PyTorch 版** 与 **TensorFlow 版**。
论文《Attention Is All You Need》(arXiv:1706.03762) 的最小可训练复刻：
正弦位置编码 ＋ 2 个 block（多头自注意力＋因果 mask、FFN，各带残差）＋ Linear→softmax。
规模缩小到 D=64、4 头、2 层，干的任务和 03 一模一样——**同题不同架构，公平对照**。

与论文的三处有意差异（都是现代惯例）：pre-LN（不用学习率预热也稳，如今 LLM 的标配）、
去掉所有偏置和 LN 仿射参数、模型缩小。docstring 里有完整说明。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。

## 它教你的三件事（在框架里分别长什么样）

1. **缩放点积注意力**：`softmax(QKᵀ/√d_k)V`——每个词按相关性从全句"抄"信息。
   手搓版拆头/拼回/softmax 反向全手写；框架版把这几步换成框架张量（仍是显式矩阵乘，
   图和公式才对得上），**softmax 的反向交给 autograd / GradientTape**。
   工程里会用融合算子（PyTorch `F.scaled_dot_product_attention(is_causal=True)`、
   Keras `MultiHeadAttention`）——教学版故意写开。
2. **因果 mask**：解码器生成第 i 个词时不能偷看未来——未来位置的分数加 −1e9，
   softmax 后权重归零。这是所有 GPT 系 LLM 的第一性规则。
3. **没有时间循环**：对照 03 的 `for t in range(...)`，这里整段序列一次矩阵乘算完。
   这就是论文 Table 1 里"串行操作 O(1)"的含义——训练能并行的根源。

## 两份代码逐环节对照

| 环节 | PyTorch 版 | TensorFlow 版 |
|---|---|---|
| 词嵌入 | `nn.Embedding(V, D)` | `keras.layers.Embedding(V, D)` |
| 归一化 | `nn.LayerNorm(elementwise_affine=False, eps=1e-5)` | `LayerNormalization(center=False, scale=False, epsilon=1e-5)`（关掉仿射=对应手搓"无 γ/β"） |
| 投影/FFN | `nn.Linear(..., bias=False)` | `keras.layers.Dense(..., use_bias=False)` |
| **权重摆法** | **转置**：torch 存 (出,入)，灌 `W.T` | **不转置**：手搓 Wq 本来就是 (入,出)（前向 `xn@Wq`），Keras kernel 同形直接灌 |
| 注意力 | 手写拆头/矩阵乘/softmax（教学版） | 同左（tf 张量版） |
| 反向 | `loss.backward()` | `with tf.GradientTape() as t: …` → `t.gradient` |
| 训练 | `torch.optim.Adam(lr=3e-3)`，`grad.clamp_(-5,5)` | `keras.optimizers.Adam(lr=3e-3, eps=1e-8)`，`tf.clip_by_value` |
| **踩坑记录** | —— | ① Dense kernel 与手搓 W 同形（**这章不转置**——与 03 的 LSTM 正相反，规则是"谁的存法和手搓一致谁不转置"）；② Keras 3 层权重在 `.weights`/属性名要用对 |

## 怎么跑

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 04_mini_transformer\04_mini_transformer_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 04_mini_transformer\04_mini_transformer_tensorflow.py
```

两版**同数据、同随机种子、同初始权重**（按手搓版的随机流顺序抽取）：

- 梯度检查（含 ReLU，阈值放宽到 1e-4）：PyTorch **5.8e-11**、TensorFlow **2.6e-11**；
- 训练 4000 步（Adam，lr=3e-3，逐元素裁剪 ±5）：loss 从 **2.56** 降到 **0.003** 附近；
- 结束时两版都从 **"attention is " 前缀确定性续写全句**：
  `attention is all you need. attention is all you need.`
- 长训练两版曲线形态一致、逐点有浮点级差异（同 03 的理由：算子实现顺序不同）。

## figs 参考图怎么读

两份代码各自生成同名 4 张图：`figs_pytorch/` 与 `figs_tensorflow/`——**并排看**。

| 图 | 看什么 |
|---|---|
| fig1_attention_heatmap | **本仓库最值得看的图**：训练后 2 层×4 头的注意力权重——layer 0 各头分工明确（有的头死盯某位置、有的头盯前一个字符），右上角黑三角就是因果 mask。论文 Figure 3-5 的自制品 |
| fig2_architecture | 架构流程图 + 因果 mask 小图 + 与 03 的对照注解 |
| fig3_positional_encoding | 正弦位置编码条纹（论文 3.5 节公式原样）——每个位置一根独特的"指纹" |
| fig4_loss_curve | 训练曲线，与 03 同基准对照 |

## 两个已知现象（不是 bug）

- 从单字符 "a" 生成会得到**相位偏移**的句子：语料循环出现、训练块起点随机，
  模型无法区分这是句首的 a 还是 all 里的 a。两个框架停在了**不同的合法相位**上
  （PyTorch 版 `atttention…`，TF 版 `all you need…`）——两个都是"对的"，前缀续写才是确定性验收。
- 训练曲线偶有回弹：每步是从语料随机截的一段，难段 loss 天然偏高。

## 自测题

1. 为什么编码器读原文不需要 mask，解码器必须带？（答：编码器看的原文是完整已知信息，
   随便看；解码器在训练时要模拟"逐字生成"，看到未来就是作弊）
2. 多头为什么"多视角不加钱"？（答：8 头各 64 维拼回 512，总计算量与单头 512 相当）
3. 位置编码为什么直接加到词向量上而不拼接？（答：加法保持维度且让两者在同一空间
   里叠加，实验证明学习式和正弦式效果几乎一样——论文 Table 3 (E)）
4. 这章 Keras 版为什么不转置、而 03 的 LSTM 章要转置？
   （答：看**手搓版当时怎么存 W**——本章手搓 `q = xn@Wq` 存 (入,出)，与 Keras 同形；
   03 手搓 `a = W@x` 存 (出,入)，与 PyTorch 同形。规则：**谁和手搓存法一致谁不转置**）
5. ReLU 在场时梯度检查阈值为什么放宽到 1e-4？（答：ReLU 在 0 点不可导，差分可能恰好跨过
   折点——放宽是承认数学现实，不是放水；实测仍能到 1e-11 量级）
