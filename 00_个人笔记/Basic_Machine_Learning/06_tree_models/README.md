# 06 树模型家族 —— 张量化 CART、随机森林与 GBDT + 可微分软树（PyTorch × TensorFlow 双版本）

## 这是什么

同一套内容、两份代码：**PyTorch 版** 与 **TensorFlow 版**。
表格数据之王的一族，也是吸附项目主模型 **XGBoost** 的老家。

注意树模型没有反向传播——CART 不是梯度下降训出来的，而是**贪心地找一个
"最值得问的问题"逐层切数据**。这是它与 01-04 神经家族的根本区别。
但本章末尾会展示桥梁：**把 if-else 换成 sigmoid 软门，树也能梯度下降**（软树）。

> 重建说明：本仓原先每章是纯 numpy 手搓版，现已替换为双框架版
> （numpy 版在 git 历史中可查）。本版用**框架张量**重写了经典三件套的数值核心
> ——框架不只是神经网络库，更是张量计算库。

## 两份代码里有什么

| 部件 | 干什么 | PyTorch 版写法 | TensorFlow 版写法 |
|---|---|---|---|
| `MiniTree` | 最小 CART：回归叶子=均值（按 SSE 分裂），二分类叶子=多数类（按加权 Gini 分裂） | `torch.argsort` + `torch.cumsum` + `torch.where` | `tf.argsort` + `tf.cumsum` + `tf.where` |
| `MiniForest` | 随机森林：Bootstrap 抽样本＋随机抽特征，长 51 棵树投票 | 树=张量版 MiniTree（RNG 用 numpy，两版同流） | 同左 |
| `gbdt_fit` | GBDT：每棵新树拟合"当前 ensemble 的残差"，一队树接力纠错 | 同左 | 同左 |
| **`SoftTree`（新增）** | **可微分软树**：if-else → sigmoid 软门，叶子概率可导，Adam 端到端训练 | `nn.Parameter` + autograd | `tf.Variable` + GradientTape |

**树的结构仍是 Python 递归**（这是 CART 的本质）；张量化的是"找最优切分"这个重计算。

## 三句话讲清家族分工

1. **决策树**：可解释性之王，任何预测都能还原成一串 if-else；
   但单棵树要么欠拟合要么过拟合——**深度就是复杂度旋钮**。
2. **随机森林（Bagging）= 降方差**：并行长 N 棵有分歧的树投票，
   把单棵深树的锯齿过拟合平均掉。
3. **GBDT/XGBoost（Boosting）= 降偏差**：串行一队浅树接力纠错，
   每棵只修一点点（学习率缩放），叠起来威力巨大。
   XGBoost 在此之上加了二阶信息、显式正则、行列采样和一整套工程优化。

## 怎么跑

```bash
C:\Users\Guards\.conda\envs\yolo\python.exe 06_tree_models\06_tree_models_pytorch.py
C:\Users\Guards\.conda\envs\yolo\python.exe 06_tree_models\06_tree_models_tensorflow.py
```

两版**同数据、同随机种子**（连 Bootstrap 的抽样流都对齐），实测数字一字不差：

- 单棵树（深度 3）训练集精度 **94.8%**；随机森林（51 棵）**99.5%**；
- 软树：TF 版带数值梯度检查（**1.2e-11**，军规 2 照旧），两版都收敛到 **99.8%**；
- 软树初始化由各自框架的随机流定种子——两版边界形状相似但非逐位相同
  （初始门的位置不同，殊途同归）。

## figs 参考图怎么读

两份代码各自生成同名 **7 张**图：`figs_pytorch/` 与 `figs_tensorflow/`——**并排看**。

| 图 | 看什么 |
|---|---|
| fig1_decision_tree | 决策树结构：以材料吸附为例（比表面积→孔容→温度→吸附量），每个预测可还原成 if-else |
| fig2_depth_overfit | **深度=复杂度旋钮**：depth 1/3/12 在两月牙上的实测边界——欠拟合→刚好→过拟合 |
| fig3_random_forest | Bagging 示意 + 实测：单棵深树锯齿 vs 51 棵森林平滑 |
| fig4_gbdt_residual | **最值得看的图**：GBDT 四阶段实测——常数→台阶→贴趋势→咬合曲线，残差接力肉眼可见 |
| fig5_xgboost | XGBoost 的四件升级 + 分裂增益公式 |
| fig6_family_map | 家族演化全景：CART 分出 Bagging 门派（降方差）和 Boosting 门派（降偏差），怎么选写在底部 |
| **fig7_soft_tree（新增）** | **同一棵树两种训练方式**：左=CART 横平竖直的台阶边界；右=软门+梯度下降的光滑边界 |

## 自测题

1. 随机森林为什么能降方差？（答：N 棵高方差但低相关树的平均，方差 ≈ σ²/N＋协方差项，
   特征随机就是为了让它们不相关）
2. GBDT 为什么用浅树（depth 2）而随机森林用深树？（答：boosting 靠叠加补偿欠拟合，
   每棵只学一点点防止单棵太强带偏方向；bagging 靠平均，每棵本身得足够强）
3. 为什么你的吸附项目用 XGBoost 而不用神经网络？
   （答：几千行的表格数据、异构特征、需要特征重要性解释——这正是树家族的主场）
4. 张量版的 `_best_split` 和手搓版的数学完全一样，那"张量化"到底换了什么？
   （答：换的是**谁来算**——argsort/cumsum/where 由框架在连续内存上批量算，
   以后要上 GPU 只改一行 `.cuda()`；算法的每一步没变）
5. 软树的 `leaf_probs` 为什么可导？传统 CART 为什么不可导？
   （答：软树把"选左还是选右"换成 sigmoid 概率——每个样本同时"部分地在两边"，
   处处可导；CART 是 argmin 硬选择，梯度到此为止）
6. 软树和神经网络什么关系？（答：每个内部节点就是一个线性层+sigmoid——
   软树 = 结构受限的神经网络；这也是"可微决策树"研究把两个家族连起来的思路）
