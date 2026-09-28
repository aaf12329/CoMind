# dl-from-scratch：从线性回归到 Transformer，纯 numpy 手搓

> 配套阅读：论文《Attention Is All You Need》(arXiv:1706.03762)
> 环境：只需要 numpy + matplotlib，无任何深度学习框架。
> 每个模型一个文件夹（脚本 + figs/ 参考图），每个文件自包含、可单独运行。

## 运行方式

```bash
D:\Anaconda\envs\imooc_ai\python.exe 01_linear_regression\01_linear_regression.py
```

每次运行会完整重训一遍模型，并**重新生成该文件夹 figs/ 下的全部参考图**——
图不是贴上去的素材，全部由代码从真实训练中画出来，与代码永远同步。

## 路线图（一个模型一个 commit）

| 文件夹 | 模型 | 任务 | 学到什么 |
|---|---|---|---|
| 01_linear_regression | 线性回归 | 拟合 y=3x+1+噪声 | 反向传播的种子：dL/dŷ = ŷ−y；数值梯度检查 |
| 02_mlp | 多层感知机 | 两月牙二分类 | 隐层+非线性；链式法则跨层传递；Adam |
| 03_lstm | LSTM（手推 BPTT） | 字符级语言模型 | 门控记忆；沿时间反传；梯度裁剪 |
| 04_mini_transformer | 迷你 Transformer | 同 03，对照实验 | Q/K/V；因果 mask；注意力没有时间循环 |
| 05_audio_models | 音频模型架构图（纯示意图，不训练） | —— | 声音表示 + 12 个模型/范式：WaveNet、RNN逐点、Tacotron2、HiFi-GAN、神经codec、Whisper、wav2vec2、VITS、声音克隆、MusicGen |
| 06_tree_models | 树模型家族（手搓 MiniCART，不用sklearn） | 两月牙分类 + 1D 回归 | 决策树/深度=复杂度；随机森林=Bagging降方差；GBDT=残差接力降偏差；XGBoost=二阶+正则+工程 |

## 各模型参考图（figs/ 目录）

**01_linear_regression/figs/**
- fig1_data_vs_fit.png 数据散点 + 真实线 + 拟合线
- fig2_loss_curve.png loss 对数轴下降曲线
- fig3_loss_surface_gd.png 损失面等高线 + 梯度下降轨迹（"下山"的具象化）
- fig4_backprop_diagram.png 反向传播计算图：前向蓝、反向红，起点是损失

**02_mlp/figs/**
- fig1_moons_data.png 两月牙数据
- fig2_decision_boundary.png MLP 弯边界 vs 线性直边界（同屏对照）
- fig3_network_backprop.png 网络结构 + 反向传播链公式
- fig4_loss_acc_curve.png loss/accuracy 双轴训练曲线

**03_lstm/figs/**
- fig1_lstm_cell.png LSTM 单元：传送带 c + 三个门开关
- fig2_bptt_unroll.png 时间展开：前向实线、反向虚线（dh/dc 两条梯度流）
- fig3_loss_curve.png 训练曲线 + 瞎猜下限 ln(V) 基准线
- fig4_gates_heatmap.png 训练后三个门在整句话上的开合热图（96 维 × 27 字符）

**04_mini_transformer/figs/**
- fig1_attention_heatmap.png 训练后注意力权重（2 层 × 4 头，因果 mask 可见）——论文 Figure 3-5 的自制品
- fig2_architecture.png 架构流程图 + 因果 mask 小图
- fig3_positional_encoding.png 正弦位置编码矩阵（论文 3.5 公式的可视化）
- fig4_loss_curve.png 训练曲线（与 03 同任务对照）

**05_audio_models/figs/**（音频模型全家福，12 张，编号即阅读顺序）
- fig1_signal_forms.png 声音的四种形态：波形/频谱图/MFCC/token（频谱图是真算的）
- fig2_wavenet.png WaveNet：膨胀因果卷积逐点拟合波形 + 门控 + skip 求和
- fig3_rnn_pointwise.png SampleRNN/WaveRNN：RNN 系逐点生成
- fig4_tacotron2.png Tacotron 2：LSTM 编码器-解码器 + 对齐带 + 声码器
- fig5_hifi_gan.png HiFi-GAN：生成器 vs 双判别器的对抗训练
- fig6_neural_codec_llm.png 现代路线：EnCodec 把声音变 token，Transformer 接手
- fig7_whisper.png Whisper：多任务语音识别（转写/翻译/语种/时间戳，特殊token调度）
- fig8_wav2vec2.png wav2vec 2.0：自监督预训练，零标注先学会"听"
- fig9_vits.png VITS：端到端语音合成（Piper 血统），训练件与推理件分离
- fig10_voice_clone.png 少样本声音克隆三段式（GPT-SoVITS/CosyVoice/VALL-E 通用范式）
- fig11_musicgen.png MusicGen：文本→音乐（4层RVQ交错 + 单个LM）
- fig12_task_map.png 任务全景图：任务×输入形态×代表模型收口

**06_tree_models/figs/**（树模型家族，6 张；分类/回归树与集成全部纯 numpy 手搓，不依赖 sklearn）
- fig1_decision_tree.png 决策树结构：以材料吸附为例的"连连提问"
- fig2_depth_overfit.png 深度=复杂度旋钮：depth 1/3/12 的实测边界（欠拟合→刚好→过拟合）
- fig3_random_forest.png 随机森林=Bagging：示意 + 单棵深树 vs 51棵森林的实测边界
- fig4_gbdt_residual.png GBDT 残差接力：F0→+1棵→+20棵→+200棵 的实测拟合演化
- fig5_xgboost.png XGBoost 给 GBDT 加了什么：二阶信息/显式正则/行列采样/工程优化 + 分裂增益公式
- fig6_family_map.png 家族演化全景：CART→Bagging(随机森林/ExtraTrees) 与 Boosting(AdaBoost→GBDT→XGBoost/LightGBM/CatBoost)

## 贯穿全系列的三条军规

1. **反向传播永远从损失开始**：前向的最后一个节点（损失）就是反向的第一个节点，
   种子梯度对回归是 `ŷ−y`，对 softmax+交叉熵是 `p−onehot(y)`，一路乘回去。
2. **手推的梯度必须过数值梯度检查**：用 `(L(θ+ε)−L(θ−ε))/2ε` 当质检员，
   判据用"大梯度看相对、小梯度看绝对"：`|a−n|/max(1,|a|,|n|)`，
   纯相对判据会在梯度≈0 的条目上假报（03 踩过的坑）。
3. **损失必须从 ln(V) 出发**：语言模型训练前 loss ≈ ln(词表大小) = 瞎猜水平，
   偏离这个起点说明前向有 bug。

## 已知现象（不是 bug）

- 04 从单字符 "a" 生成会得到相位偏移的句子（语料循环出现、训练块起点随机，
  模型无法区分这是句首的 a 还是 all 里的 a）；给全前缀 "attention is " 则确定性续写。
- 训练曲线中途偶有回弹：每步是从语料里随机截的一段，难段 loss 天然高。
