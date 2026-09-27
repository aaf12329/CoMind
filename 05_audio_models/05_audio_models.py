# -*- coding: utf-8 -*-
r"""
05 音频模型架构参考图（纯示意图，不训练，几秒跑完）
====================================================
背景：上一轮聊"能拟合声音的模型"，这一批图把它们的架构画出来。
配套阅读顺序 = 图的编号：

    fig1  声音的四种形态 —— 所有语音模型的入口
    fig2  WaveNet           膨胀因果卷积，逐采样点拟合波形
    fig3  SampleRNN/WaveRNN RNN 系逐点生成
    fig4  Tacotron 2        LSTM 版语音合成经典双件套
    fig5  HiFi-GAN          生成器画波形，判别器挑刺
    fig6  神经codec + LLM   声音 token 化后交给 Transformer

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 05_audio_models\05_audio_models.py
产物:  本文件夹 figs/ 下 6 张 PNG。图里的波形/频谱/对齐带是真算的，
       MFCC 和 token 网格是示意图（已注明）。
"""
import os
import numpy as np

rng = np.random.default_rng(7)


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

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figdir, exist_ok=True)

    def rbox(ax, cx, cy, bw, bh, text, fc, fs=10):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.07", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arr(ax, p1, p2, color="#3b6fd4", lw=1.7, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

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
    fig.savefig(os.path.join(figdir, "fig6_neural_codec_llm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("参考图已生成 -> %s" % figdir)


if __name__ == "__main__":
    save_all()
