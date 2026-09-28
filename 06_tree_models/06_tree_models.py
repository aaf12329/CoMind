# -*- coding: utf-8 -*-
r"""
06 树模型家族 —— 手搓最小 CART + 随机森林 + GBDT
================================================
仍然只用 numpy（连 sklearn 都不用），和 01-04 一个血统。
任务背景：树模型是表格数据之王，也是你吸附项目主模型 XGBoost 的家族。

本文件三件套（全部手搓）：
    MiniTree    最小 CART 决策树：贪心找"最值得问的问题"，
                回归用 SSE、二分类用 Gini，逐层递归切空间
    随机森林    Bootstrap 抽样本 + 随机抽特征，长一群树投票（降方差）
    GBDT        一队树接力纠错：每棵新树拟合"目前 ensemble 的残差"（降偏差）

跑法:  D:\Anaconda\envs\imooc_ai\python.exe 06_tree_models\06_tree_models.py
产物:  本文件夹 figs/ 下 6 张参考图:
       fig1 决策树结构(吸附领域例子) | fig2 深度如何决定过拟合(实测边界)
       fig3 随机森林=Bagging(示意+实测边界) | fig4 GBDT残差接力(实测拟合演化)
       fig5 XGBoost给GBDT加了什么(示意) | fig6 树模型家族演化全景(示意)
"""
import os
import numpy as np

rng = np.random.default_rng(1)


# ================= 数据（与 02_mlp 同源的两月牙） =================
def make_moons(n=400, noise=0.15):
    t = rng.uniform(0, np.pi, n // 2)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    Y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return X, Y


# ================= 手搓最小 CART =================
class MiniTree:
    """最小决策树。回归(task='reg')叶子=均值,分裂按SSE；分类(task='clf',0/1标签)
    叶子=多数类,分裂按加权Gini。max_features 用于随机森林的特征随机。"""

    class _Node:
        __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self, max_depth=3, task="reg", min_samples=4,
                 max_features=None, rng=None):
        self.max_depth = max_depth
        self.task = task
        self.min_samples = min_samples
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)

    def _best_split(self, X, y, feats):
        """排序+前缀和实现，每个特征 O(n log n)。返回 (代价, 特征, 阈值)。"""
        n = len(y)
        best = (np.inf, None, None)
        for f in feats:
            order = np.argsort(X[:, f])
            xs, ys = X[order, f], y[order]
            k = np.arange(1, n)
            if self.task == "reg":
                cs, cq = np.cumsum(ys), np.cumsum(ys * ys)
                sse_l = cq[:-1] - cs[:-1] ** 2 / k
                sse_r = (cq[-1] - cq[:-1]) - (cs[-1] - cs[:-1]) ** 2 / (n - k)
                imp = sse_l + sse_r
            else:                                   # 二分类 0/1 的加权 Gini
                tot1 = ys.sum()
                c1 = np.cumsum(ys)[:-1]
                g_l = 1 - (c1 / k) ** 2 - ((k - c1) / k) ** 2
                g_r = 1 - ((tot1 - c1) / (n - k)) ** 2 - ((n - k - tot1 + c1) / (n - k)) ** 2
                imp = k * g_l + (n - k) * g_r
            imp[:self.min_samples - 1] = np.inf
            imp[n - self.min_samples:] = np.inf
            imp[xs[1:] == xs[:-1]] = np.inf         # 同值之间不能切
            j = int(np.argmin(imp))
            if imp[j] < best[0]:
                best = (imp[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    def _grow(self, X, y, depth):
        node = self._Node()
        node.feat = node.thr = node.left = node.right = None
        if depth >= self.max_depth or len(y) < self.min_samples:
            node.value = y.mean() if self.task == "reg" else int(np.bincount(y.astype(int)).argmax())
            return node
        d = X.shape[1]
        feats = range(d)
        if self.max_features:
            feats = self.rng.choice(d, self.max_features, replace=False)
        cost, f, thr = self._best_split(X, y, feats)
        if f is None:
            node.value = y.mean() if self.task == "reg" else int(np.bincount(y.astype(int)).argmax())
            return node
        node.feat, node.thr = int(f), float(thr)
        m = X[:, f] <= thr
        node.left = self._grow(X[m], y[m], depth + 1)
        node.right = self._grow(X[~m], y[~m], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _predict_one(self, x):
        nd = self.root
        while nd.feat is not None:
            nd = nd.left if x[nd.feat] <= nd.thr else nd.right
        return nd.value

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, float)])


# ================= 随机森林（Bagging） =================
class MiniForest:
    def __init__(self, n_trees=51, max_depth=10, max_features=1, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(2)
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.trees = []
        for i in range(self.n_trees):
            idx = self.rng.integers(0, len(y), len(y))     # Bootstrap 抽样本
            t = MiniTree(max_depth=self.max_depth, task="clf",
                         max_features=self.max_features,
                         rng=np.random.default_rng(1000 + i)).fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict(self, X):
        votes = np.mean([t.predict(X) for t in self.trees], axis=0)
        return (votes >= 0.5).astype(int)


# ================= GBDT（Boosting，回归版） =================
def gbdt_fit(X, y, n_est=200, lr=0.3, max_depth=2, snap_at=(1, 20, 200)):
    """每棵新树拟合当前 ensemble 的残差。返回 (初始F0, [(stage, F快照, mse)], 最终F)。"""
    F = np.full(len(y), y.mean())
    F0 = y.mean()
    snaps = []
    want = set(snap_at)
    for k in range(1, n_est + 1):
        t = MiniTree(max_depth=max_depth, task="reg", min_samples=4).fit(X, y - F)
        F = F + lr * t.predict(X)
        if k in want:
            snaps.append((k, F.copy(), float(np.mean((y - F) ** 2))))
    return F0, snaps, F


# ================= 边界画图工具 =================
def boundary(ax, model, X, Y, title, gx, gy):
    XX, YY = np.meshgrid(gx, gy)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)
    P = model.predict(G).reshape(XX.shape)
    ax.contourf(XX, YY, P, levels=[0, 0.5, 1], colors=["#aec7e8", "#ff9896"], alpha=0.4)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=11, c="#4878b0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=11, c="#d65f5f")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# ================= 参考图 =================
def save_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figdir, exist_ok=True)

    def rbox(ax, cx, cy, bw, bh, text, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.06", fc=fc, ec="#444444", lw=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arr(ax, p1, p2, color="#666666", lw=1.5, ls="-"):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))

    BLUE, GREEN, RED, ORANGE = "#3b6fd4", "#2e8b57", "#d4402e", "#e07b00"

    # ---- fig1 决策树结构（吸附领域例子） ----
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")
    rbox(ax, 6.5, 5.6, 4.2, 0.85, "比表面积 > 500 m²/g ？", "#dbe9ff", fs=10)
    rbox(ax, 3.3, 4.0, 3.4, 0.85, "孔容 > 0.4 cm³/g ？", "#dbe9ff", fs=10)
    rbox(ax, 9.7, 4.0, 3.4, 0.85, "吸附温度 > 60 ℃ ？", "#dbe9ff", fs=10)
    leaves = [(1.8, "吸附量 ≈ 3.2"), (4.8, "吸附量 ≈ 1.1"),
              (8.2, "吸附量 ≈ 2.4"), (11.2, "吸附量 ≈ 0.5")]
    for cx, txt in leaves:
        rbox(ax, cx, 2.4, 2.6, 0.8, txt, "#d9f2d9", fs=9.5)
    arr(ax, (5.6, 5.15), (4.0, 4.5), "#666666")
    arr(ax, (7.4, 5.15), (9.0, 4.5), "#666666")
    arr(ax, (2.6, 3.55), (2.2, 2.85), "#666666")
    arr(ax, (4.0, 3.55), (4.4, 2.85), "#666666")
    arr(ax, (8.9, 3.55), (8.5, 2.85), "#666666")
    arr(ax, (10.4, 3.55), (10.9, 2.85), "#666666")
    ax.text(4.35, 4.85, "是", fontsize=10, color=GREEN)
    ax.text(8.65, 4.85, "否", fontsize=10, color=RED)
    ax.text(1.95, 3.2, "是", fontsize=9, color=GREEN)
    ax.text(4.6, 3.2, "否", fontsize=9, color=RED)
    ax.text(8.35, 3.2, "是", fontsize=9, color=GREEN)
    ax.text(11.05, 3.2, "否", fontsize=9, color=RED)
    ax.text(0.3, 1.1, "分裂准则：分类用基尼/熵，回归用方差(SSE)；每个节点贪心挑“最值得问的问题”\n"
                      "叶子 = 预测值（回归取均值，分类取多数类）；深度越深越会背答案——单棵深树极易过拟合\n"
                      "可解释性之王：任何一次预测都能还原成一串 if-else，审稿人和工程师都看得懂",
            fontsize=9.5, linespacing=1.8, va="top")
    ax.set_title("决策树：一个自动长出来的“连连提问”结构（以材料吸附为例）", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig1_decision_tree.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig2 深度如何决定过拟合（实测边界） ----
    X, Y = make_moons()
    gx = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 300)
    gy = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 300)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    for ax, depth, name in zip(axes, [1, 3, 12],
                               ["深度=1：欠拟合（只会问一个问题）",
                                "深度=3：刚刚好",
                                "深度=12：过拟合（把噪声也背下来了）"]):
        t = MiniTree(max_depth=depth, task="clf", min_samples=4).fit(X, Y)
        boundary(ax, t, X, Y, name, gx, gy)
    fig.suptitle("决策树怎么切空间：只用横平竖直的刀，深度就是复杂度旋钮（手搓CART实测）", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig2_depth_overfit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig3 随机森林 = Bagging ----
    fig = plt.figure(figsize=(12.0, 6.8))
    ax = fig.add_axes([0.02, 0.47, 0.96, 0.5])
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.1); ax.axis("off")
    rbox(ax, 1.6, 2.3, 2.2, 0.9, "原始训练集\n(400条)", "#dbe9ff", fs=9.5)
    for i, cx in enumerate([4.6, 7.0, 9.4]):
        rbox(ax, cx, 2.3, 1.7, 0.9, "Bootstrap\n抽样%d" % (i + 1), "#fff9c4", fs=8.5)
        rbox(ax, cx, 0.9, 1.7, 0.8, "树%d\n(特征也随机抽)" % (i + 1), "#d9f2d9", fs=8.5)
        arr(ax, (cx - 0.35, 1.85), (cx - 0.35, 1.35), "#666666")
        ax.annotate("", xy=(cx - 0.9, 2.3), xytext=(2.75, 2.3),
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                                    connectionstyle="arc3,rad=%.2f" % (-0.32 * i)))
    arr(ax, (10.3, 2.3), (11.0, 2.3), "#666666")
    rbox(ax, 12.0, 2.3, 1.8, 0.9, "投票/平均", "#ffd9d2", fs=9.5)
    ax.text(6.5, 0.25, "每棵树都在略有不同的数据上、用略有不同的特征长成——一群有分歧的树投票，"
                       "方差被平均掉了（这是“降方差”的集成）", ha="center", fontsize=9.5)
    ax.set_title("随机森林 = Bagging：并行长 N 棵有分歧的树，一起投票", fontsize=12)
    # 实测：单棵深树 vs 森林
    ax1 = fig.add_axes([0.08, 0.06, 0.38, 0.36])
    deep = MiniTree(max_depth=12, task="clf", min_samples=2).fit(X, Y)
    boundary(ax1, deep, X, Y, "单棵深树：锯齿过拟合", gx, gy)
    ax2 = fig.add_axes([0.55, 0.06, 0.38, 0.36])
    forest = MiniForest(n_trees=51, max_depth=10, max_features=1).fit(X, Y)
    acc = np.mean(forest.predict(X) == Y)
    boundary(ax2, forest, X, Y, "随机森林(51棵)：平滑（训练集精度 %.0f%%）" % (100 * acc), gx, gy)
    fig.savefig(os.path.join(figdir, "fig3_random_forest.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig4 GBDT：残差接力（实测演化） ----
    xr = np.linspace(0, 10, 240)
    yr = 3.0 * np.sin(1.5 * xr) * np.exp(-0.15 * xr) + 0.15 * xr \
        + rng.normal(0, 0.12, len(xr))
    F0, snaps, F = gbdt_fit(xr.reshape(-1, 1), yr, n_est=200, lr=0.3, max_depth=2)
    stages = [("F0：只有一个常数（全体的均值）", np.full_like(yr, F0)),
              ("＋第1棵树（拟合残差后）", snaps[0][1]),
              ("＋20棵树", snaps[1][1]),
              ("＋200棵树", snaps[2][1])]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for ax, (name, Fc) in zip(axes.ravel(), stages):
        ax.scatter(xr, yr, s=8, c="#9db8d2", label="数据")
        ax.plot(xr, 3.0 * np.sin(1.5 * xr) * np.exp(-0.15 * xr) + 0.15 * xr,
                "g--", lw=1.6, label="真实规律")
        ax.plot(xr, Fc, "r-", lw=1.8, label="当前 ensemble F")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("GBDT：每棵新树都在拟合“目前预测的残差”，一队树接力纠错（手搓实现，"
                 "lr=0.3, depth=2）\n最终 MSE = %.4f" % snaps[2][2], fontsize=12)
    fig.savefig(os.path.join(figdir, "fig4_gbdt_residual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig5 XGBoost 给 GBDT 加了什么 ----
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")
    rbox(ax, 6.5, 5.3, 6.6, 1.1, "GBDT 的损失：L = Σ l(y, ŷ)\n"
         "XGBoost 的目标：Obj = Σ l(y, ŷ) ＋ Σ Ω(f)，  Ω(f) = γ·T ＋ ½λ·Σw²",
         "#dbe9ff", fs=10)
    boxes = [(2.4, 3.4, "① 二阶信息\n用梯度g和Hessian h\n定分裂增益与叶重", "#fff9c4"),
             (6.5, 3.4, "② 显式正则\n惩罚叶子数和叶值\n防过拟合写进目标函数", "#ffe9c9"),
             (10.6, 3.4, "③ 行列采样\n样本抽样＋特征抽样\n(随机森林的招也借来了)", "#e4d9f7"),
             (6.5, 1.6, "④ 工程优化\n加权分位草图、缓存、并行\nLightGBM再提速(直方图+leaf-wise)", "#d9f2d9")]
    for cx, cy, txt, fc in boxes:
        rbox(ax, cx, cy, 3.6, 1.15, txt, fc, fs=9)
    arr(ax, (3.4, 4.0), (5.4, 4.9), "#666666", lw=1.2)
    arr(ax, (6.5, 4.0), (6.5, 4.72), "#666666", lw=1.2)
    arr(ax, (9.6, 4.0), (7.6, 4.9), "#666666", lw=1.2)
    arr(ax, (6.5, 2.2), (6.5, 2.82), "#666666", lw=1.2)
    ax.text(0.3, 0.4, "分裂增益公式：Gain = ½[ G_L²/(H_L+λ) ＋ G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ]  − γ"
            "      （G=梯度和，H=Hessian和；增益低于γ就别切了）", fontsize=9.5)
    ax.set_title("XGBoost = GBDT 的工程旗舰：数学上多二阶项和正则，工程上快得离谱", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig5_xgboost.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig6 家族演化全景 ----
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 7.4); ax.axis("off")
    rbox(ax, 7.0, 6.5, 4.6, 0.9, "CART 决策树 (1984)\n单棵树：可解释，但容易过拟合", "#dbe9ff", fs=10)
    rbox(ax, 3.2, 4.6, 4.2, 1.0, "并行集成 Bagging\n随机森林 (2001)：Bootstrap+特征随机\nN棵树投票 → 降方差",
         "#d9f2d9", fs=9.5)
    rbox(ax, 3.2, 3.0, 3.6, 0.8, "Extra Trees：切分点也随机\n更快更野", "#e8f7e8", fs=9)
    rbox(ax, 10.8, 4.7, 5.4, 1.15, "串行集成 Boosting\nAdaBoost (1997)：错题加权\nGBDT (2001)：改为拟合梯度/残差 → 降偏差",
         "#fff9c4", fs=9.5)
    for cx, w, txt, fc in [(8.9, 2.6, "XGBoost (2014)\n二阶＋正则＋工程", "#ffe9c9"),
                           (11.4, 2.3, "LightGBM\n直方图＋leaf-wise\n更快", "#ffe9c9"),
                           (13.3, 2.1, "CatBoost\n类别特征\n有序提升", "#ffe9c9")]:
        rbox(ax, cx, 2.35, w, 1.15, txt, fc, fs=8.5)
    arr(ax, (7.0, 6.0), (4.0, 5.15), "#666666", lw=1.4)
    arr(ax, (7.0, 6.0), (10.4, 5.3), "#666666", lw=1.4)
    arr(ax, (3.2, 4.05), (3.2, 3.45), "#666666", lw=1.2)
    arr(ax, (10.8, 4.1), (9.0, 2.95), "#666666", lw=1.2)
    arr(ax, (11.0, 4.1), (11.3, 2.95), "#666666", lw=1.2)
    arr(ax, (11.4, 4.1), (13.2, 2.95), "#666666", lw=1.2)
    ax.text(0.3, 1.15, "怎么选：表格数据 / 样本不多 / 需要解释 → 树家族（你的吸附项目主模型就是 XGBoost）",
            fontsize=10.5)
    ax.text(0.3, 0.55, "图像 / 文本 / 音频 / 序列 → 神经网络（本仓库 01-04）；两者还常做特征级联：树的输出喂给神经网络，反之亦然",
            fontsize=10.5)
    ax.set_title("树模型家族演化全景：从单棵树到两大集成门派", fontsize=12)
    fig.savefig(os.path.join(figdir, "fig6_family_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("figures -> %s" % figdir)


if __name__ == "__main__":
    save_all()
