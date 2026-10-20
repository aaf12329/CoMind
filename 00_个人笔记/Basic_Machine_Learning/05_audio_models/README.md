# 05 音频模型架构图鉴 —— 12 张图 + 可运行骨架（PyTorch × TensorFlow 双版本）

## 这是什么

**这个文件夹不训练任何模型**。两份代码（PyTorch 版 / TensorFlow 版）各自做两件事：

1. **可运行骨架**：把图鉴里的代表架构在本框架里真实搭出来，跑一次前向验形状
   （只验连通性，不训练）；
2. **12 张架构参考图**：编号即阅读顺序，每张图底部加一行"本框架对应实现"。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。

## 阅读路线（编号即顺序，正好是一部技术进化史）

| 图 | 模型/主题 | 一句话 | 骨架里对应哪段 |
|---|---|---|---|
| fig1 | 声音的四种形态 | 波形→频谱图→MFCC→token，所有语音模型的入口（频谱图是真算的） | 张量排法 |
| fig2 | WaveNet | 膨胀因果卷积逐采样点拟合波形；门控思想与 LSTM 同款 | `WaveNetBlock` |
| fig3 | SampleRNN/WaveRNN | RNN 系逐点生成，03 的血缘亲戚 | `GRUPointwise` |
| fig4 | Tacotron 2 | LSTM 版语音合成两件套：编码-解码写频谱，声码器还原波形 | `Tacotron2Lite` |
| fig5 | HiFi-GAN | 生成器画波形、双判别器挑刺的对抗训练，现在的主流声码器 | `HiFiGANLite` + `Discriminator` |
| fig6 | 神经codec + LLM | EnCodec 把声音压成 token，Transformer 像写文章一样"写声音" | `CodecLM` |
| fig7 | Whisper | 多任务语音识别：转写/翻译/语种/时间戳全靠特殊 token 调度 | `WhisperLite` |
| fig8 | wav2vec 2.0 | 自监督预训练：零标注先学会"听"，10 分钟标注也能微调出 ASR | `Wav2Vec2Lite` |
| fig9 | VITS | 端到端 TTS：训练件与推理件分离，Piper 的血统，手机能跑 | 编码器 + `HiFiGANLite` |
| fig10 | 声音克隆三段式 | GPT-SoVITS/CosyVoice/VALL-E 通用范式：注册声纹→LLM 生成→声码器还原 | 编码器 + `CodecLM` |
| fig11 | MusicGen | 文本→音乐：4 层 RVQ 四个"声部"，delay pattern 让单个 LM 全管完 | `CodecLM` |
| fig12 | 任务全景图 | 收口：任务 × 输入形态 × 代表模型，一张表看完 | —— |

## 两份代码怎么对照（骨架实测形状）

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 05_audio_models\05_audio_models_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 05_audio_models\05_audio_models_tensorflow.py
```

跑完骨架部分会打印各架构的输入输出形状（两版数值形状一致，只是通道维度摆法不同）：

| 骨架 | PyTorch（通道在前） | TensorFlow/Keras（通道在后） | 亮点对照 |
|---|---|---|---|
| WaveNetBlock | `(1, 16, 50)` → 同形状 + skip | `(1, 50, 16)` → 同形状 + skip | **Keras 自带 `padding="causal"`；PyTorch 要 `padding=d*(k-1)` 后手动裁右端** |
| GRUPointwise | 波形 `(1,1,200)` → `(1,200,256)` | 波形 `(1,200,1)` → `(1,200,256)` | `nn.GRU` ↔ `keras.layers.GRU` |
| Tacotron2Lite | text `(1,7)` → 梅尔 `(1,12,80)` | 同左 | `nn.LSTMCell` ↔ `keras.layers.LSTMCell`（记忆带着走） |
| HiFi-GAN G→D | mel →波形 →真/假打分 | 同左 | `ConvTranspose1d` ↔ `Conv1DTranspose` |
| Wav2Vec2Lite | `(1,1,1600)` → `(1,161,64)` | `(1,1600,1)` → `(1,160,64)` | 卷积下采样 + Transformer |
| WhisperLite | mem+token → `(1,6,64)` | 同左 | 因果自注意力 + 交叉注意力 |
| CodecLM | tokens `(1,24)` → `(1,24,1024)` | 同左 | GPT 式因果解码 |

## 读完应该带走的三句话

1. 声音不是特殊东西，就是一条采样率很高的序列——你已经会处理序列了。
2. 所有路线共用三件套：**先把声音变成序列（频谱图或 token）→ Transformer/卷积加工 →
   最后都要一个声码器/解码器回到波形**。
3. 现代 LLM 语音模式的核心一跃：声音被 token 化成"一门外语"，之后一切都回到
   你在 04 里已经学过的自回归游戏。

> 诚实声明：频谱图、注意力对齐带是真数据算出来的；MFCC 和 token 网格是示意图，图上已注明。

## 自测题

1. 为什么 WaveNet 用膨胀卷积而不是普通卷积？（答：感受野指数级扩大——层数不多也能看到很远的
   过去，且没有 RNN 的串行瓶颈）
2. 因果（causal）在音频生成里为什么是硬约束？（答：自回归逐点生成时，第 t 个点的预测绝不能
   使用 t+1 及以后的信息，否则训练时"偷看答案"，推理时露馅）
3. Keras 的 causal 填充和 PyTorch 的手动左填充在数学上等价吗？
   （答：等价——都是只看左边 d*(k-1) 个输入；差别只是谁替你写这段代码）
4. 为什么 HiFi-GAN 要两个判别器（MPD+MSD）？（答：一个盯周期性（音高相关），一个盯多分辨率
   细节（音质相关）——"挑刺角度"互补）
5. 声音克隆三段式里，哪一段决定了"像不像本人"，哪一段决定"说得对不对"？
   （答：参考 token 决定音色（像不像），LLM 生成的语音 token 决定内容（对不对），声码器只负责还原）
