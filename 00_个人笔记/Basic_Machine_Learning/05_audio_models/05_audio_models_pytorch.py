# -*- coding: utf-8 -*-
r"""
05 音频模型架构 · PyTorch 版 —— 可运行骨架 + 参考图
=====================================================
本版两块内容：

  A. 【可运行骨架】把 12 张图里的代表架构，用 PyTorch 真实搭出来并跑一次前向
     （只验形状与连通性，不训练——这是本章的定位：认识架构，不是训练模型）：
       WaveNet 膨胀因果卷积块 / SampleRNN-WaveRNN 的 GRU 逐点生成 /
       Tacotron2 式编码器-注意力解码器 / HiFi-GAN 生成器+判别器 /
       wav2vec2 式卷积特征+Transformer 编码器 / Whisper 式加解码器 /
       神经 codec 的 token 网格 + GPT 式语言模型
  B. 【参考图】与原版相同的 12 张架构图（fig1~fig12），每张图底部加一行
     "PyTorch 对应实现"——看图就知道这段架构落到哪个层。

跑法:  C:\Users\Guards\.conda\envs\yolo\python.exe 05_audio_models\05_audio_models_pytorch.py
产物:  本文件夹 figs_pytorch/ 下 12 张 PNG（与 TF 版同名，可并排对照）
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(7)


# =========================================================================
# A. 可运行骨架（PyTorch）
# =========================================================================

class WaveNetBlock(nn.Module):
    """膨胀因果卷积块：因果卷积（不许看未来）+ 门控 tanh⊙σ + skip 求和（fig2）。"""

    def __init__(self, ch, dilations=(1, 2, 4), k=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = d * (k - 1)                     # 因果 = 左填充，右不填
            self.convs.append(nn.Conv1d(ch, 2 * ch, k, dilation=d, padding=pad))
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x):                          # x: (B, ch, T)
        skip = 0
        for conv in self.convs:
            y = conv(x)
            y = y[:, :, : x.shape[-1]]             # 右端裁掉"多看的未来"
            a, b = y.chunk(2, dim=1)
            x = x + self.proj(torch.tanh(a) * torch.sigmoid(b))   # 门控（LSTM 同款）
            skip = skip + x
        return x, skip


class GRUPointwise(nn.Module):
    """SampleRNN / WaveRNN 家族：GRU 记记忆，逐采样点预测下一档（fig3）。"""

    def __init__(self, ch=32, bins=256):
        super().__init__()
        self.gru = nn.GRU(1, ch, batch_first=True)
        self.head = nn.Linear(ch, bins)

    def forward(self, x):                          # x: (B, 1, T) 原始波形采样点
        h, _ = self.gru(x.transpose(1, 2))         # (B, T, ch)
        return self.head(h)                        # (B, T, 256) 每点一个分布


class Tacotron2Lite(nn.Module):
    """Tacotron2 式：文字 → LSTM 编码器 → 注意力 → LSTM 解码器逐帧写频谱（fig4）。"""

    def __init__(self, vocab=64, emb=32, hid=64, mel=80):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.enc = nn.LSTM(emb, hid, batch_first=True)
        self.dec = nn.LSTMCell(mel, hid)
        self.q = nn.Linear(hid, hid)
        self.k = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, hid)
        self.mel_out = nn.Linear(hid, mel)

    def forward(self, text_ids, n_steps=12):       # 演示：贪心走 12 帧
        enc, _ = self.enc(self.emb(text_ids))      # (B, L, hid)
        K, V = self.k(enc), self.v(enc)
        h, c = torch.zeros(enc.shape[0], 64), torch.zeros(enc.shape[0], 64)
        frame = torch.zeros(enc.shape[0], 80)
        outs = []
        for _ in range(n_steps):
            h, c = self.dec(frame, (h, c))
            att = torch.softmax(self.q(h).unsqueeze(1) @ K.transpose(1, 2) / 8.0, dim=-1)
            frame = self.mel_out((att @ V).squeeze(1))     # 上一步的帧接回解码器
            outs.append(frame)
        return torch.stack(outs, 1)                # (B, n_steps, 80) 梅尔帧


class HiFiGANLite(nn.Module):
    """HiFi-GAN：生成器转置卷积上采样画波形；判别器多层卷积挑刺（fig5）。"""

    def __init__(self, mel=80, up=8):
        super().__init__()
        self.up_in = nn.ConvTranspose1d(mel, 128, up * 2, stride=up, padding=up // 2)
        self.res = WaveNetBlock(128, dilations=(1, 3, 5))     # 残差块：膨胀卷积 ×{1,3,5}
        self.up_out = nn.ConvTranspose1d(128, 1, 4, stride=2, padding=1)

    def forward(self, mel):                        # (B, 80, T) → (B, 1, ~T*16) 波形
        x, _ = self.res(self.up_in(mel))
        return self.up_out(x)


class Discriminator(nn.Module):
    """判别器（MSD 的简化骨架）：多分辨率卷积下采样 → 真/假打分。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 15, stride=4, padding=7), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 15, padding=7))

    def forward(self, wav):                        # (B,1,T) → (B,1,T/16) 逐段打分
        return self.net(wav)


class Wav2Vec2Lite(nn.Module):
    """wav2vec2 式：卷积特征编码器（直接吃波形）+ Transformer 上下文网络（fig8）。"""

    def __init__(self, d=64):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 32, 10, stride=5, padding=5), nn.GELU(),
            nn.Conv1d(32, d, 3, stride=2, padding=1), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True)
        self.ctx = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, wav):                        # (B,1,T) → (B, T', d) 上下文特征
        return self.ctx(self.feat(wav).transpose(1, 2))


class WhisperLite(nn.Module):
    """Whisper 式：编码器吃整段频谱，解码器因果 mask + 交叉注意力出 token（fig7）。"""

    def __init__(self, d=64, vocab=64):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True), num_layers=2)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, 4, 4 * d, batch_first=True), num_layers=2)
        self.out = nn.Linear(d, vocab)

    def forward(self, mel_feats, token_ids):       # (B,Tm,d) + (B,Tt,) → (B,Tt,vocab)
        mem = self.enc(mel_feats)
        L = token_ids.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.dec(torch.randn(token_ids.shape[0], L, mem.shape[-1]),
                     mem, tgt_mask=causal)
        return self.out(h)


class CodecLM(nn.Module):
    """神经 codec + LLM（fig6/fig11）：token 网格进 GPT 式解码器，预测下一个 token。"""

    def __init__(self, codebook=1024, d=64, ctx=64):
        super().__init__()
        self.emb = nn.Embedding(codebook, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, codebook)
        self.ctx = ctx

    def forward(self, token_ids):                  # (B, T) int → (B, T, codebook)
        L = token_ids.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.body(self.emb(token_ids), mask=causal)
        return self.head(h)


def demo_skeletons():
    """跑一遍全部骨架，只验形状与连通性（不训练）。"""
    x = torch.randn(1, 16, 50)
    y, s = WaveNetBlock(16)(x)
    print("WaveNetBlock  :", tuple(x.shape), "->", tuple(y.shape), "| skip", tuple(s.shape))
    print("GRUPointwise  :", "波形(1,1,200) ->", tuple(GRUPointwise()(torch.randn(1, 1, 200)).shape))
    print("Tacotron2Lite :", "text(1,7) ->", tuple(Tacotron2Lite()(torch.randint(0, 40, (1, 7))).shape))
    mel = torch.randn(1, 80, 40)
    wav = HiFiGANLite()(mel)
    print("HiFiGAN  G→D  :", "mel", tuple(mel.shape), "-> wav", tuple(wav.shape),
          "-> score", tuple(Discriminator()(wav).shape))
    print("Wav2Vec2Lite  :", "wav(1,1,1600) ->", tuple(Wav2Vec2Lite()(torch.randn(1, 1, 1600)).shape))
    mem = torch.randn(1, 50, 64)
    print("WhisperLite   :", "mem(1,50,64)+tok(1,6) ->",
          tuple(WhisperLite()(mem, torch.randint(0, 40, (1, 6))).shape))
    print("CodecLM       :", "tokens(1,24) ->", tuple(CodecLM()(torch.randint(0, 1000, (1, 24))).shape))


# =========================================================================
# B. 参考图（与原版相同的 12 张；底部加一行本框架的对应实现）
# =========================================================================

def stft_mag(sig, nfft=256, hop=64):
    """手写 STFT 幅度谱，画频谱图用。"""
    win = np.hanning(nfft)
    frames = [np.abs(np.fft.rfft(win * sig[i:i + nfft]))
              for i in range(0, len(sig) - nfft, hop)]
    return np.array(frames).T            # (频率, 时间)


def fake_mel(F=64, Tm=48):
    """长得像梅尔频谱的假数据：低频强、有谐波纹、随时间摆动。"""
    mel = 2.2 * np.exp(-np.arange(F)[:, None] / 22.0) * np.ones((1, Tm))
    mel += 0.5 * np.sin(np.arange(F)[:, None] / 3.0 + np.arange(Tm)[None, :] / 2.0) \
        * np.exp(-np.arange(F)[:, None] / 30.0)
    mel += rng.normal(0, 0.06, mel.shape)
    return mel


def fake_alignment(L=10, F=36):
    """长得像注意力对齐带的假数据：一条从左下到右上的斜带。"""
    att = np.zeros((L, F))
    for i in range(L):
        center = i * (F - 1) / (L - 1)
        att[i] = np.exp(-((np.arange(F) - center) ** 2) / 5.0)
    return att


def save_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_pytorch")
    os.makedirs(figdir, exist_ok=True)

    def rbox(ax, cx, cy, bw, bh, text, fc, fs=10):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.07", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arr(ax, p1, p2, color="#3b6fd4", lw=1.7, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    def fw(fig, note):
        """底部一行：本图对应 PyTorch 实现（TF 版同一位置是对应 keras 层）。"""
        fig.text(0.5, -0.015, "PyTorch 对应实现：" + note, ha="center", fontsize=8.5, color="#555555")

    BLUE, GREEN, RED, ORANGE = "#3b6fd4", "#2e8b57", "#d4402e", "#e07b00"

    # ================= fig1 声音的四种形态 =================
    sr = 4000
    tt = np.arange(sr) / sr
    sig = 0.5 * np.sin(2 * np.pi * np.cumsum(200 + 700 * tt) / sr) \
        + 0.2 * np.sin(2 * np.pi * 900 * tt)
    fig = plt.figure(figsize=(11.5, 5.0))
    ax = fig.add_axes([0.02, 0.44, 0.96, 0.52])
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.6); ax.axis("off")
    boxes = [(1.5, "原始波形\n16000个采样点/秒", "#dbe9ff"),
             (4.5, "频谱图\n时间 × 频率能量", "#d6f0f7"),
             (7.5, "MFCC\n13维/帧（拟人耳）", "#fff3d6"),
             (10.5, "神经codec token\n离散的\"声音字\"", "#e4d9f7")]
    for cx, txt, fc in boxes:
        rbox(ax, cx, 1.5, 2.6, 1.0, txt, fc, fs=9.5)
    for x1, x2 in [(2.8, 3.2), (5.8, 6.2), (8.8, 9.2)]:
        arr(ax, (x1, 1.5), (x2, 1.5), "#666666")
    ax.text(6.0, 0.22, "同一段声音的四种形态 —— 所有语音模型都从其中一种入口吃进去",
            ha="center", fontsize=10.5)
    ax1 = fig.add_axes([0.055, 0.08, 0.17, 0.26])
    ax1.plot(tt[:800] / sr * 4000, sig[:800], lw=0.7, color=BLUE)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("波形", fontsize=9)
    S = stft_mag(sig)
    ax2 = fig.add_axes([0.305, 0.08, 0.17, 0.26])
    ax2.imshow(20 * np.log10(S + 1e-6), origin="lower", aspect="auto", cmap="magma")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("频谱图（真算的）", fontsize=9)
    mf = np.outer(np.linspace(1, 0.25, 13), np.sin(np.linspace(0, 8, 24))) \
        + rng.normal(0, 0.06, (13, 24))
    ax3 = fig.add_axes([0.555, 0.08, 0.17, 0.26])
    ax3.imshow(mf, aspect="auto", cmap="cividis", interpolation="nearest")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("MFCC（示意）", fontsize=9)
    toks = rng.integers(0, 20, (4, 16))
    ax4 = fig.add_axes([0.805, 0.08, 0.17, 0.26])
    ax4.imshow(toks, aspect="auto", cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title("token 网格（示意）", fontsize=9)
    fw(fig, "波形=裸张量；频谱=torch.stft；四种形态只是张量的不同排法")
    fig.savefig(os.path.join(figdir, "fig1_signal_forms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig2 WaveNet =================
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(6.0, 0.32, "输入：原始波形采样点  …  x(t-3)  x(t-2)  x(t-1)  x(t)",
            ha="center", fontsize=10)
    dil = [(1.3, "膨胀因果卷积  dilation=1（看前1个点）"),
           (2.35, "膨胀因果卷积  dilation=2（隔1个看）"),
           (3.4, "膨胀因果卷积  dilation=4（隔3个看）"),
           (4.45, "膨胀因果卷积  dilation=8（隔7个看）")]
    for i, (y, txt) in enumerate(dil):
        rbox(ax, 3.6, y, 3.6, 0.72, txt, "#dbe9ff" if i % 2 == 0 else "#c7dcff", fs=9.5)
        if i == 0:
            arr(ax, (3.6, 0.62), (3.6, 0.92), "#666666")
        else:
            arr(ax, (3.6, dil[i - 1][0] + 0.36), (3.6, y - 0.38), "#666666")
        arr(ax, (5.45, y), (8.25, 4.75), BLUE, lw=1.0)   # skip 连接
    ax.text(0.4, 4.0, "膨胀率翻倍，\n感受野也翻倍：\n层数不多也能\n看到很远的过去",
            fontsize=9, color="#444444", va="center", linespacing=1.5)
    ax.text(6.8, 2.45, "块内门控（LSTM同款）：\nz = tanh(a) ⊙ σ(b)",
            fontsize=9, color="#444444", linespacing=1.6, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.9))
    rbox(ax, 9.0, 4.75, 2.2, 0.8, "Σ 全部 skip 相加", "#fff3d6", fs=9.5)
    rbox(ax, 9.0, 3.5, 2.6, 0.85, "softmax over 256档\n→ 下一个采样点的分布", "#ffd9d2", fs=9)
    rbox(ax, 9.0, 2.3, 2.0, 0.7, "采样出 x(t+1)", "#d9f2d9", fs=9.5)
    arr(ax, (9.0, 4.35), (9.0, 3.95), "#666666")
    arr(ax, (9.0, 3.05), (9.0, 2.68), "#666666")
    arr(ax, (9.0, 1.92), (9.0, 0.75), RED, lw=1.6, ls="--")
    arr(ax, (8.7, 0.75), (5.5, 0.75), RED, lw=1.6, ls="--")
    ax.text(6.15, 1.02, "自回归：新点接回输入，逐点生成", fontsize=9, color=RED)
    ax.set_title("WaveNet：膨胀因果卷积逐点拟合波形（2016，谷歌助手早期的嗓子）", fontsize=12)
    fw(fig, "nn.Conv1d(dilation=d, padding=d*(k-1)) 后裁右端；门控=tanh⊙sigmoid（见骨架 WaveNetBlock）")
    fig.savefig(os.path.join(figdir, "fig2_wavenet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig3 RNN 逐点生成 =================
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    xs4 = [1.4, 3.8, 6.2, 8.6]
    for i, xc in enumerate(xs4):
        ax.text(xc, 0.55, "x(t-%d)" % (4 - i), ha="center", fontsize=10)
        rbox(ax, xc, 2.3, 1.5, 0.85, "GRU\n(记忆h)", "#fff3d6", fs=9.5)
        arr(ax, (xc, 0.82), (xc, 1.85), "#666666")
        arr(ax, (xc, 2.75), (xc, 3.7), BLUE)
        ax.text(xc, 3.95, "预测下一个点", ha="center", fontsize=8.5, color=BLUE)
        if i < 3:
            arr(ax, (xc + 0.78, 2.3), (xs4[i + 1] - 0.78, 2.3), GREEN, lw=2)
    ax.text(5.0, 3.02, "h（记忆）→", fontsize=9, color=GREEN)
    arr(ax, (9.35, 3.85), (11.3, 3.85), RED, lw=1.5, ls="--")
    arr(ax, (11.3, 3.85), (11.3, 0.55), RED, lw=1.5, ls="--")
    arr(ax, (11.3, 0.55), (9.35, 0.55), RED, lw=1.5, ls="--")
    ax.text(10.6, 2.1, "自回归：\n生成的点\n接回输入", fontsize=9, color=RED,
            ha="center", linespacing=1.4)
    ax.text(0.3, 5.0, "SampleRNN：粗/细两三层分辨率——粗层定节奏，细层补毛刺\n"
                      "WaveRNN：单个 GRU + 8bit 双分支，小到手机能跑",
            fontsize=9.5, linespacing=1.6)
    ax.set_title("RNN 系逐点生成：SampleRNN / WaveRNN —— 用记忆一步步\"念\"出波形", fontsize=12)
    fw(fig, "nn.GRU（骨架 GRUPointwise）+ Linear(256) 输出每点 256 档分布")
    fig.savefig(os.path.join(figdir, "fig3_rnn_pointwise.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig4 Tacotron 2 =================
    fig = plt.figure(figsize=(11.5, 6.2))
    ax = fig.add_axes([0.02, 0.46, 0.96, 0.5])
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
    rbox(ax, 1.4, 1.9, 2.0, 0.9, "文字\n\"你好世界\"", "#ececec", fs=9.5)
    rbox(ax, 4.4, 1.9, 3.0, 0.9, "Embedding + LSTM 编码器\n（读懂每个字）", "#dbe9ff", fs=9)
    rbox(ax, 8.2, 1.9, 3.2, 0.9, "注意力 + LSTM 解码器\n（逐帧写频谱，自回归）", "#fff3d6", fs=9)
    rbox(ax, 11.8, 1.9, 1.9, 0.9, "80维\n梅尔频谱", "#d6f0f7", fs=9.5)
    arr(ax, (2.45, 1.9), (2.85, 1.9), "#666666")
    arr(ax, (5.95, 1.9), (6.55, 1.9), "#666666")
    arr(ax, (9.85, 1.9), (10.8, 1.9), "#666666")
    arr(ax, (8.2, 1.4), (8.2, 0.75), BLUE, lw=1.4, ls="--")
    arr(ax, (8.9, 0.75), (8.9, 1.4), BLUE, lw=1.4, ls="--")
    ax.text(9.5, 0.85, "上一步写的帧\n接回解码器", fontsize=8.5, color=BLUE, linespacing=1.3)
    axa = fig.add_axes([0.16, 0.07, 0.16, 0.30])
    axa.imshow(fake_alignment(), aspect="auto", cmap="viridis", interpolation="nearest")
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_title("注意力对齐带\n写到哪，看哪个字", fontsize=8.5)
    axb = fig.add_axes([0.44, 0.07, 0.16, 0.30])
    axb.imshow(fake_mel(), origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("梅尔频谱", fontsize=8.5)
    rbox(ax, 11.7, 0.62, 2.4, 0.7, "声码器：WaveNet / HiFi-GAN\n频谱 → 波形", "#ffd9d2", fs=8)
    arr(ax, (11.8, 1.42), (11.75, 1.0), "#666666")
    axc = fig.add_axes([0.76, 0.07, 0.16, 0.30])
    axc.plot(tt[:1500] / sr * 4000, sig[:1500], lw=0.6, color=BLUE)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title("最终波形", fontsize=8.5)
    ax.set_title("Tacotron 2：LSTM 版语音合成的经典双件套（编码器-解码器 + 声码器）", fontsize=12)
    fw(fig, "nn.LSTM 编码器 + nn.LSTMCell 逐帧解码 + 点积注意力（见骨架 Tacotron2Lite）")
    fig.savefig(os.path.join(figdir, "fig4_tacotron2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig5 HiFi-GAN =================
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.8); ax.axis("off")
    rbox(ax, 1.5, 5.0, 2.4, 0.9, "80维梅尔频谱\n（ Tacotron 2 的输出）", "#d6f0f7", fs=9)
    rbox(ax, 5.0, 5.0, 3.8, 1.3, "生成器 G\n转置卷积逐级上采样（×256）\n残差块：膨胀卷积 ×{1,3,5}", "#fff3d6", fs=9)
    arr(ax, (2.75, 5.0), (3.05, 5.0), "#666666")
    rbox(ax, 5.0, 2.7, 2.2, 0.7, "生成的波形（假）", "#ffd9d2", fs=9)
    arr(ax, (5.0, 4.32), (5.0, 3.08), "#666666")
    rbox(ax, 2.2, 2.7, 2.0, 0.7, "真实波形（真）", "#d9f2d9", fs=9)
    rbox(ax, 10.2, 5.6, 4.6, 0.9, "判别器 MPD\n把波形重排成周期图案，查周期性破绽", "#dbe9ff", fs=9)
    rbox(ax, 10.2, 4.1, 4.6, 0.9, "判别器 MSD\n三种分辨率下采样，听细节破绽", "#c7dcff", fs=9)
    arr(ax, (6.15, 2.7), (7.8, 5.35), ORANGE, lw=1.4)
    arr(ax, (6.15, 2.7), (7.8, 4.15), ORANGE, lw=1.4)
    ax.text(7.0, 3.75, "假", fontsize=9, color=ORANGE)
    arr(ax, (2.6, 3.08), (7.8, 5.45), GREEN, lw=1.4)
    arr(ax, (2.6, 3.08), (7.8, 3.95), GREEN, lw=1.4)
    ax.text(3.9, 3.35, "真", fontsize=9, color=GREEN)
    rbox(ax, 10.2, 2.6, 1.8, 0.7, "真 / 假 打分", "#ececec", fs=9.5)
    arr(ax, (10.2, 5.13), (10.2, 2.98), "#666666", lw=1.2)
    arr(ax, (12.2, 2.55), (12.2, 0.9), RED, lw=1.6, ls="--")
    arr(ax, (12.0, 0.9), (5.0, 0.9), RED, lw=1.6, ls="--")
    arr(ax, (5.0, 0.9), (5.0, 1.32), RED, lw=1.6, ls="--")
    ax.text(7.6, 1.15, "对抗损失：生成器学着骗过两个判别器（另有特征匹配损失：中间特征也要像真的）",
            fontsize=9, color=RED, ha="center")
    ax.set_title("HiFi-GAN：生成器画波形，两个判别器挑刺（对抗训练，现在的主流声码器）", fontsize=12)
    fw(fig, "生成器=nn.ConvTranspose1d 上采样 + 膨胀残差块；判别器=Conv1d 堆叠（见骨架 HiFiGANLite/Discriminator）")
    fig.savefig(os.path.join(figdir, "fig5_hifi_gan.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig6 神经 codec + LLM =================
    fig = plt.figure(figsize=(11.5, 6.2))
    ax = fig.add_axes([0.02, 0.42, 0.96, 0.54])
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.4); ax.axis("off")
    rbox(ax, 1.5, 2.2, 2.2, 0.9, "一段声音\n/音乐", "#dbe9ff", fs=9.5)
    rbox(ax, 4.6, 2.2, 3.0, 0.9, "EnCodec 编码器\n卷积逐级降采样（1/320）", "#d6f0f7", fs=9)
    rbox(ax, 9.0, 2.2, 3.4, 0.9, "Transformer 语音模型\n预测下一个音频 token（GPT同款）", "#fff3d6", fs=9)
    rbox(ax, 12.0, 2.2, 1.9, 0.9, "EnCodec\n解码器", "#e4d9f7", fs=9)
    arr(ax, (2.65, 2.2), (3.05, 2.2), "#666666")
    arr(ax, (5.9, 1.7), (6.8, 0.5), "#666666")
    arr(ax, (8.5, 0.5), (9.0, 1.7), "#666666")
    arr(ax, (10.75, 2.2), (11.0, 2.2), "#666666")
    arr(ax, (12.0, 1.7), (12.0, 0.85), "#666666")
    ax.text(6.3, 0.18, "↓ token 网格 ↓", fontsize=8, ha="center", color="#666666")
    axt = fig.add_axes([0.36, 0.09, 0.24, 0.26])
    axt.imshow(toks, aspect="auto", cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    axt.set_xticks([]); axt.set_yticks([])
    axt.set_title("离散 token：多层RVQ码本\n第1层管轮廓，后几层补细节", fontsize=8.5)
    axw = fig.add_axes([0.72, 0.09, 0.2, 0.26])
    axw.plot(tt[:1200] / sr * 4000, sig[:1200], lw=0.6, color=BLUE)
    axw.set_xticks([]); axw.set_yticks([])
    axw.set_title("生成的声音/克隆的嗓音", fontsize=8.5)
    ax.text(1.6, 0.55, "VALL-E：3秒声音克隆\nMusicGen：文字→音乐\nChatGPT 语音模式：同一条 token 路",
            fontsize=9.5, linespacing=1.7)
    ax.set_title("现代路线：神经编解码器把声音变成 token，然后让 LLM \"说话\"", fontsize=12)
    fw(fig, "codec=卷积降采样+量化（多数现成实现是库）；LM=nn.Embedding + 因果 mask 的 Transformer（见骨架 CodecLM）")
    fig.savefig(os.path.join(figdir, "fig6_neural_codec_llm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig7 Whisper =================
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
    rbox(ax, 1.5, 6.1, 2.2, 0.85, "30秒音频窗口", "#dbe9ff", fs=9.5)
    rbox(ax, 4.4, 6.1, 2.6, 0.85, "log-mel 频谱\n80 × 3000", "#d6f0f7", fs=9)
    rbox(ax, 7.5, 6.1, 2.3, 0.85, "两层卷积\n（下采样）", "#d6f0f7", fs=9)
    rbox(ax, 10.9, 6.1, 3.2, 0.9, "Transformer 编码器\n双向注意力，看整段", "#dbe9ff", fs=9)
    arr(ax, (2.65, 6.1), (3.05, 6.1), "#666666")
    arr(ax, (5.75, 6.1), (6.3, 6.1), "#666666")
    arr(ax, (8.7, 6.1), (9.25, 6.1), "#666666")
    rbox(ax, 3.4, 3.5, 3.6, 0.95, "已生成的 token 序列\n<转写><语种><时间戳> 特殊token开路", "#fff9c4", fs=8.5)
    rbox(ax, 8.3, 3.5, 3.2, 0.9, "Transformer 解码器\n因果mask ＋ 交叉注意力", "#fff3d6", fs=9)
    rbox(ax, 12.0, 3.5, 1.9, 0.9, "预测\n下一个token", "#d9f2d9", fs=9)
    arr(ax, (5.25, 3.5), (6.65, 3.5), "#666666")
    arr(ax, (9.95, 3.5), (11.0, 3.5), "#666666")
    arr(ax, (10.9, 5.6), (9.2, 4.0), BLUE, lw=1.6)
    ax.text(10.4, 4.75, "交叉注意力", fontsize=9, color=BLUE)
    ax.text(0.3, 1.7, "一个模型包揽四件事：转写、翻译、语种识别、时间戳——全靠特殊 token 调度",
            fontsize=10)
    ax.text(0.3, 0.9, "68万小时弱监督音频预训练——“把规模卷到月亮上”的典型（OpenAI, 2022）",
            fontsize=10)
    ax.set_title("Whisper：音频→文字的多任务语音识别（编码器-解码器 + 特殊token）", fontsize=12)
    fw(fig, "nn.TransformerEncoder + nn.TransformerDecoder(tgt_mask=因果)（见骨架 WhisperLite）")
    fig.savefig(os.path.join(figdir, "fig7_whisper.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig8 wav2vec 2.0 =================
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    rbox(ax, 1.4, 6.0, 2.0, 0.85, "原始波形", "#dbe9ff", fs=9.5)
    rbox(ax, 4.2, 6.0, 2.7, 0.9, "卷积特征编码器\n直接吃波形，无标签", "#d6f0f7", fs=9)
    rbox(ax, 7.4, 6.0, 2.2, 0.8, "特征序列", "#d6f0f7", fs=9.5)
    rbox(ax, 10.8, 6.0, 3.0, 0.9, "Transformer\n上下文网络 C", "#dbe9ff", fs=9)
    arr(ax, (2.45, 6.0), (2.8, 6.0), "#666666")
    arr(ax, (5.6, 6.0), (6.25, 6.0), "#666666")
    ax.text(6.35, 6.6, "随机mask一小段", fontsize=8.5, color="#666666")
    arr(ax, (8.55, 6.0), (9.25, 6.0), "#666666")
    rbox(ax, 7.4, 4.1, 2.2, 0.85, "量化网络\n→ 目标", "#fff9c4", fs=9)
    arr(ax, (7.4, 5.55), (7.4, 4.58), "#666666")
    rbox(ax, 11.0, 4.1, 2.9, 0.9, "对比损失：\n认出 mask 处的目标", "#ffd9d2", fs=9)
    arr(ax, (10.8, 5.5), (11.6, 4.6), "#666666")
    arr(ax, (8.55, 4.1), (9.5, 4.1), "#666666")
    ax.text(0.3, 3.6, "预训练零标注：\n自己mask、自己出题、\n自己对答案（对比学习）",
            fontsize=9.5, linespacing=1.6, va="top")
    rbox(ax, 3.4, 2.0, 5.4, 0.85, "下游：接一个线性层微调 → 语音识别", "#d9f2d9", fs=9.5)
    ax.text(0.3, 1.0, "标注数据少到 10 分钟级别，也能微调出可用的 ASR——低资源语言的救星",
            fontsize=10)
    ax.set_title("wav2vec 2.0：自监督预训练——先无标签学会“听”，再小数据微调", fontsize=12)
    fw(fig, "Conv1d 特征编码器 + nn.TransformerEncoder 上下文网络（见骨架 Wav2Vec2Lite）")
    fig.savefig(os.path.join(figdir, "fig8_wav2vec2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig9 VITS =================
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 7); ax.axis("off")
    rbox(ax, 1.3, 6.0, 1.7, 0.8, "文本", "#ececec", fs=9.5)
    rbox(ax, 4.0, 6.0, 2.7, 0.9, "文本编码器\n＋单调对齐搜索", "#dbe9ff", fs=9)
    rbox(ax, 6.9, 6.0, 2.4, 0.9, "时长扩展\n（随机时长预测器）", "#d6f0f7", fs=9)
    rbox(ax, 9.5, 6.0, 2.0, 0.8, "Flow 流模型", "#fff3d6", fs=9.5)
    rbox(ax, 12.2, 6.0, 2.5, 0.9, "HiFi-GAN\n解码器", "#fff3d6", fs=9.5)
    arr(ax, (2.2, 6.0), (2.6, 6.0), "#666666")
    arr(ax, (5.4, 6.0), (5.6, 6.0), "#666666")
    arr(ax, (8.15, 6.0), (8.45, 6.0), "#666666")
    arr(ax, (10.55, 6.0), (10.9, 6.0), "#666666")
    ax.text(13.55, 6.0, "波形", fontsize=10, va="center")
    rbox(ax, 1.3, 3.4, 2.0, 0.85, "真实波形", "#d9f2d9", fs=9.5)
    rbox(ax, 4.0, 3.4, 2.8, 0.9, "后验编码器\n（真 → z）", "#ececec", fs=9)
    rbox(ax, 9.5, 3.4, 2.8, 0.9, "判别器\n（真假打分）", "#ececec", fs=9)
    arr(ax, (2.35, 3.4), (2.55, 3.4), "#666666")
    arr(ax, (4.6, 3.9), (9.2, 5.5), "#888888", lw=1.3, ls="--")
    arr(ax, (9.6, 3.9), (11.7, 5.5), "#888888", lw=1.3, ls="--")
    ax.text(6.7, 4.55, "灰色虚线模块只在训练时存在——推理时整个扔掉，所以快",
            fontsize=9, color="#666666")
    ax.text(0.3, 2.1, "端到端：一次前向直接出波形（对比 Tacotron 2 的两段式）",
            fontsize=10)
    ax.text(0.3, 1.3, "变分推断 ＋ 对抗训练双管齐下；Piper 的血统，手机都能跑",
            fontsize=10)
    ax.set_title("VITS：端到端语音合成——生成与判别一体，快而稳", fontsize=12)
    fw(fig, "文本编码器≈nn.LSTM/Transformer；Flow≈可逆网络（用现成实现）；解码器=HiFi-GAN 骨架")
    fig.savefig(os.path.join(figdir, "fig9_vits.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig10 少样本声音克隆三段式 =================
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    ax.set_xlim(0, 13.8); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(0.35, 6.2, "① 注册声纹", fontsize=11, color="#8a1f1f", va="center")
    rbox(ax, 3.0, 6.2, 2.6, 0.85, "参考音频\n3～10秒", "#dbe9ff", fs=9)
    rbox(ax, 6.2, 6.2, 2.2, 0.8, "编码器", "#d6f0f7", fs=9.5)
    rbox(ax, 9.9, 6.2, 3.0, 0.85, "音色/语义 参考token", "#fff9c4", fs=9)
    arr(ax, (4.35, 6.2), (5.05, 6.2), "#666666")
    arr(ax, (7.35, 6.2), (8.35, 6.2), "#666666")
    ax.text(11.6, 6.2, "→ 声音的身份证", fontsize=9.5, va="center", color="#666666")
    ax.text(0.35, 4.0, "② 生成", fontsize=11, color="#8a1f1f", va="center")
    rbox(ax, 3.0, 4.0, 2.6, 0.85, "目标文本", "#ececec", fs=9.5)
    rbox(ax, 9.9, 4.0, 3.2, 0.9, "LLM式自回归模型\n（GPT同款解码器）", "#fff3d6", fs=9)
    arr(ax, (4.35, 4.0), (9.6, 4.0), "#666666", lw=1.4)
    ax.text(6.6, 4.25, "文本与参考token一起送入", fontsize=8.5, color="#666666")
    arr(ax, (9.9, 5.72), (9.9, 4.52), BLUE, lw=1.5)
    ax.text(10.4, 5.1, "注入音色", fontsize=9, color=BLUE)
    ax.text(0.35, 1.8, "③ 还原", fontsize=11, color="#8a1f1f", va="center")
    rbox(ax, 9.9, 1.8, 2.6, 0.85, "流匹配 / 声码器", "#fff3d6", fs=9)
    arr(ax, (9.9, 3.5), (9.9, 2.3), "#666666")
    ax.text(10.4, 2.9, "语音token下发", fontsize=9, color="#666666")
    rbox(ax, 12.35, 1.8, 1.9, 0.85, "克隆的\n波形", "#d9f2d9", fs=9)
    arr(ax, (11.3, 1.8), (11.4, 1.8), "#666666")
    ax.text(0.35, 0.75, "零训练拼装：三段全是现成模块；参考音频的质量＝克隆的上限（3秒起步，10秒更稳）",
            fontsize=10)
    ax.text(0.35, 0.2, "GPT-SoVITS / CosyVoice / VALL-E 都是这个范式",
            fontsize=10)
    ax.set_title("少样本声音克隆的三段式通用范式", fontsize=12)
    fw(fig, "三段全部可用 nn.Module 组装：编码器 + CodecLM（见骨架）+ 声码器")
    fig.savefig(os.path.join(figdir, "fig10_voice_clone.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig11 MusicGen =================
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7); ax.axis("off")
    rbox(ax, 1.6, 6.0, 2.2, 0.8, "文字描述\n\"轻快的钢琴\"", "#dbe9ff", fs=9)
    rbox(ax, 4.8, 6.0, 2.8, 0.85, "T5 / CLAP\n文本编码器", "#d6f0f7", fs=9)
    arr(ax, (2.75, 6.0), (3.35, 6.0), "#666666")
    arr(ax, (4.8, 5.52), (7.0, 4.75), "#666666")
    ax.text(5.0, 5.05, "条件", fontsize=9, color="#666666")
    rbox(ax, 1.6, 3.0, 2.4, 0.85, "可选提示旋律", "#ececec", fs=9)
    rbox(ax, 4.8, 3.0, 2.6, 0.85, "EnCodec 编码\n→ 提示token", "#d6f0f7", fs=9)
    arr(ax, (2.85, 3.0), (3.45, 3.0), "#666666")
    arr(ax, (6.15, 3.3), (7.0, 4.1), "#666666")
    rbox(ax, 8.6, 4.3, 3.4, 1.1, "单个 Transformer 语言模型\n4层码本交错建模（delay pattern）", "#fff9c4", fs=9)
    rbox(ax, 12.3, 4.3, 2.0, 0.9, "EnCodec\n解码器", "#e4d9f7", fs=9.5)
    arr(ax, (10.35, 4.6), (11.25, 4.6), "#666666")
    ax.text(12.3, 5.5, "音乐", fontsize=10, va="center", ha="center")
    ax.text(0.3, 1.6, "4层RVQ＝4个“声部”：第1层定主干，后3层补细节；delay pattern 把4层错开半拍，"
                      "一个LM按顺序全管完", fontsize=10)
    ax.text(0.3, 0.85, "30秒一段＋滑窗拼接可以无限长；文本条件决定风格、节奏与乐器",
            fontsize=10)
    ax.set_title("MusicGen：文本→音乐的生成路线（Meta）", fontsize=12)
    fw(fig, "条件编码=任意文本编码器（如 Hugging Face 现成模型）；生成器=CodecLM 骨架")
    fig.savefig(os.path.join(figdir, "fig11_musicgen.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= fig12 音频任务全景 =================
    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.text(2.4, 7.9, "任务", fontsize=12, ha="center", fontweight="bold")
    ax.text(6.2, 7.9, "输入形态", fontsize=12, ha="center", fontweight="bold")
    ax.text(10.4, 7.9, "代表模型（关键件）", fontsize=12, ha="center", fontweight="bold")
    rows = [
        ("语音识别 ASR", "频谱图", "Whisper、wav2vec2（编码器-解码器 / 自监督）", "#dbe9ff", "#e8f2ff", "#eef6ff"),
        ("语音合成 TTS", "文本", "Tacotron2、VITS、Piper（编码-解码＋声码器）", "#d6f0f7", "#e2f5fa", "#eaf8fb"),
        ("声音克隆", "文本＋参考音频", "VALL-E、GPT-SoVITS、CosyVoice（token语言模型）", "#fff9c4", "#fffbe0", "#fefae8"),
        ("声音分类", "MFCC / 频谱图", "CNN、LSTM（你手搓的03加个分类头就能上）", "#ffd9d2", "#ffe8e4", "#ffefec"),
        ("音乐生成", "文字描述", "MusicGen（codec token ＋ 语言模型）", "#e4d9f7", "#ece3fa", "#f1eafc"),
    ]
    ys = [6.7, 5.35, 4.0, 2.65, 1.3]
    for (task, inp, mod, c1, c2, c3), y in zip(rows, ys):
        rbox(ax, 2.4, y, 3.0, 0.95, task, c1, fs=10)
        rbox(ax, 6.2, y, 2.4, 0.95, inp, c2, fs=9.5)
        rbox(ax, 10.4, y, 5.6, 0.95, mod, c3, fs=9)
    ax.text(0.3, 0.3, "万变不离其宗：① 声音先变成序列（频谱图或 token）→ ② Transformer 当主力加工"
                      "→ ③ 最后都要一个声码器/解码器回到波形", fontsize=10.5)
    ax.set_title("音频模型任务全景图（12张图到这里收口）", fontsize=12)
    fw(fig, "“分类”一档：把 CRNN3 的 LSTM 骨架加 nn.Linear(c, 类别数) 就是分类器")
    fig.savefig(os.path.join(figdir, "fig12_task_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


if __name__ == "__main__":
    print("=== A. 可运行骨架（PyTorch，只验形状不训练）===")
    demo_skeletons()
    print("\n=== B. 生成 12 张参考图 ===")
    save_all()
