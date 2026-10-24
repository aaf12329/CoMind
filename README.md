<div align="center">

# 共知 · CoMind

**个人知识、学习实践与 AI 协作参考库**

保存我理解世界的过程，也为 AI 与我协作提供共同背景。

*Personal knowledge, learning practice, and references for working with AI.*

</div>

---

## 从这里开始 / Start here

CoMind 保存四类东西：**个人笔记与学习实践**、**供 AI 使用的协作规范**、**思考与写作**、**恢复与部署清单**。
原 `Basic_Machine_Learning` 与 `Quick_deployment` 已归并到本仓库，分别保留原提交历史。

| 区域 | 内容 | 阅读方式 |
|---|---|---|
| [00_个人笔记](00_个人笔记/) | 私人笔记；其中 [Basic_Machine_Learning](00_个人笔记/Basic_Machine_Learning/README.md) 是双框架学习资料 | 笔记部分所有者自用、AI 不主动读取；**学习资料部分可按需读取** |
| [01_AI项目协作参考](01_AI项目协作参考/README.md) | AI 协作规范的**唯一出处**：子代理与效率、编程规范、提交与 git 权限 | 从该目录的 `README.md` 索引进，按"开工前 → 写代码 → 提交"三步取用 |
| [02_思考与写作](02_思考与写作/README.md) | 关于学习、智能与直觉的思考笔记、文章及 TXT 版本 | 从阅读索引选文章；需要追溯时，再查看对应笔记 |
| [Quick_deployment](Quick_deployment/README.md) | 重装恢复、新机迁移、网站入口、VS Code 与 AI 工具插件清单 | 按恢复顺序查阅对应清单 |

### 我现在要做什么？

- **学习机器学习**：先读 [学习路线与运行方式](00_个人笔记/Basic_Machine_Learning/README.md)。覆盖线性回归、MLP、LSTM、迷你 Transformer、音频架构与树模型。
- **阅读文章或继续整理想法**：从 [思考与写作](02_思考与写作/README.md) 按主题进入，区分思考笔记、文章与 TXT 版本。
- **重装系统或换电脑**：先读 [恢复流程](Quick_deployment/README.md)，再查 [相关网站](Quick_deployment/相关网站.md)、[VS Code 插件](Quick_deployment/VSCode插件清单.md) 和 [AI 插件](Quick_deployment/AI插件.md)。
- **让 AI 接手工作**：AI 先读根目录 [`AGENTS.md`](AGENTS.md)，再进 [`01_AI项目协作参考/README.md`](01_AI项目协作参考/README.md)。

## AI 导引 / References for AI

> 🔴 **AI 的唯一入口是 [`AGENTS.md`](AGENTS.md)**，那是根级锚点，含三条红线与三步流程。
> 下面这张表是给人看的速查；细则一律以 `01_AI项目协作参考/` 内的文件为准。

| 你要做什么 / 找什么 | 去哪里 | 备注 |
|---|---|---|
| 开工前决定要不要开子代理、开几个 | [00_子代理与效率.md](01_AI项目协作参考/00_子代理与效率.md) | 🔴 同时最多 **3 个**；须申请获批，申请里附所有者原话 |
| 写代码 / 按语言查规范 | [01_编程规范/代码规范.md](01_AI项目协作参考/01_编程规范/代码规范.md) | 按语言跳转；工程项目另读工程架构规范，学习资料另读学习资料规范 |
| 提交、push、看 git 权限 | [02_提交与协作规范.md](01_AI项目协作参考/02_提交与协作规范.md) | 🔴 权限在 §02，提交格式在 §03/§04 |
| 接手或新建项目 | [03_项目接手手册_模板与写法.md](01_AI项目协作参考/03_项目接手手册_模板与写法.md) | 只写项目事实，标准只给指针 |
| 了解所有者背景与协作偏好 | [用户画像_协作偏好.md](01_AI项目协作参考/用户个人信息参考/用户画像_协作偏好.md) | 带【请过滤本文件】标记；分享、批量处理或生成公开材料时跳过整个文件 |
| 学习与比较机器学习实现 | [Basic_Machine_Learning/README.md](00_个人笔记/Basic_Machine_Learning/README.md) | 子目录有章节说明、两套实现与配套参考图 |
| 恢复开发环境、编辑器或 AI 工具 | [Quick_deployment/README.md](Quick_deployment/README.md) | 清单中的版本与本机路径是历史快照，恢复时按新机器调整 |
| 配置 conda | [conda 路径注意事项](01_AI项目协作参考/01_编程规范/语言规范/Python/conda路径注意事项.md) | 仅使用 conda 时读取；环境以机器上实际状态为准 |
| 调研 GitHub / 找论文 | [04_资料检索/](01_AI项目协作参考/04_资料检索/) | GitHub 限流与降级、学术来源核实与合法全文获取 |

本次迁入的思考类笔记与文章见 [02_思考与写作](02_思考与写作/README.md)；其他个人资料与环境配置总表仍以所有者指定的位置为准，不因归并自动收入或读取。

## 目录结构 / Layout

```text
CoMind/
├── AGENTS.md                            AI 的唯一入口（红线 + 三步）
├── README.md                            本文件
├── 00_个人笔记/
│   ├── conda常用指令.md
│   ├── 前端整合笔记_TS_JS_路线图_CSS选择器.md
│   └── Basic_Machine_Learning/           双框架学习资料（AI 可读）
│       ├── README.md
│       ├── Create-Venv.ps1               自选 Python，在本目录创建 .venv
│       ├── 01_linear_regression/ … 06_tree_models/
│       │   各章节保留代码、README、figs_pytorch/、figs_tensorflow/
├── 01_AI项目协作参考/                    规范的唯一出处
│   ├── README.md                         索引（先看这里）
│   ├── 00_子代理与效率.md                 ① 开工前：开几个 / 配额默认 3 / 报备模板
│   ├── 01_编程规范/                       ② 写代码
│   │   ├── 代码规范.md                   按语言跳转的入口
│   │   ├── 工程架构规范.md
│   │   ├── 学习资料代码规范.md
│   │   ├── 语言规范/{Python,JavaScript,TypeScript,PowerShell,Bash}/
│   │   └── 专项规范/LLM_Agent项目规范.md
│   ├── 02_提交与协作规范.md               ③ 提交：格式 + git 权限（版本唯一出处）
│   ├── 03_项目接手手册_模板与写法.md
│   ├── 04_资料检索/
│   └── 用户个人信息参考/                 分享时遵守文件过滤标记
├── 02_思考与写作/                        思考笔记、文章与 TXT 版本
│   ├── README.md                         按主题阅读与版本索引
│   ├── 给变化以时间——关于学习与神经连接的猜想.md
│   ├── 奇思妙想.md、奇思妙想_文章版.md / .txt
│   └── 直觉.md、直觉_文章版.md / .txt
└── Quick_deployment/
    ├── README.md
    ├── 相关网站.md
    ├── VSCode插件清单.md
    └── AI插件.md
```

## 获取与使用 / Get the repository

```powershell
git clone https://github.com/aaf12329/CoMind.git
cd CoMind
```

机器学习示例的命令以 `00_个人笔记/Basic_Machine_Learning/` 为工作目录。使用 [Create-Venv.ps1](00_个人笔记/Basic_Machine_Learning/Create-Venv.ps1) 选择本机 Python，在 **`00_个人笔记/Basic_Machine_Learning/.venv/`** 创建环境，再自行安装相应依赖。脚本不下载 Python 或自动安装框架；具体参数见 [学习资料 README](00_个人笔记/Basic_Machine_Learning/README.md)。运行学习脚本可能重新生成章节中的参考图。

```powershell
cd 00_个人笔记/Basic_Machine_Learning
powershell -NoProfile -ExecutionPolicy Bypass -File ./Create-Venv.ps1
# 自行安装对应依赖后：
./.venv/Scripts/python.exe 01_linear_regression/01_linear_regression_pytorch.py
```

部署资料从 [Quick_deployment/README.md](Quick_deployment/README.md) 开始。账号凭据、完整私有配置、模型及数据集按各项目自己的要求另行备份和恢复。

## 归并记录 / Consolidation

2026-10-18 起，以 [CoMind](https://github.com/aaf12329/CoMind) 为统一入口：

| 原仓库 | 本库位置 | 归并时的源版本 |
|---|---|---|
| [Basic_Machine_Learning](https://github.com/aaf12329/Basic_Machine_Learning) | [00_个人笔记/Basic_Machine_Learning/](00_个人笔记/Basic_Machine_Learning/README.md) | `7cb9cb1` |
| [Quick_deployment](https://github.com/aaf12329/Quick_deployment) | [Quick_deployment/](Quick_deployment/README.md) | `6bb8778` |

保留原项目的目录结构、学习代码、配套图文和提交历史。后续统一在本库对应目录维护；原仓库链接用于追溯来源。本次归并只更新资料组织、导航及运行入口说明，不把历史实验记录视为本次复现结果。

The two repositories were consolidated into their own subdirectories with their original Git histories preserved. CoMind is the main entry point for future maintenance. This consolidation does not represent a new reproduction of the recorded experiments.

2026-10-24：将桌面“写文章的灵感”中的 7 个文件迁入 `02_思考与写作/`，保留原文件名与内容，新增主题阅读索引。

## 使用与维护约定 / Ground rules

> 协作规则的**正文只有一处**：`01_AI项目协作参考/`。本表只列边界，不复述细则；
> 逐条出处见 [01_AI项目协作参考/README.md](01_AI项目协作参考/README.md) 第三节。

1. **个人笔记按授权读取**：`00_个人笔记/` 不作为 AI 默认参考材料；
   🔴 唯一例外是 `00_个人笔记/Basic_Machine_Learning/`——它是学习资料，AI 可按需读取。
2. **遵守过滤标记**：分享本库、批量提供给 AI 或生成公开材料时，跳过带【请过滤本文件】标记的文件；AI 不代删标记。
3. **保护个人信息与凭据**：提交和公开内容中过滤真实姓名、学校、单位、地理位置等身份信息；密钥、账号凭据、环境和缓存不入库。
4. **三条红线**（细则见 [`AGENTS.md`](AGENTS.md) 与 [00_子代理与效率.md](01_AI项目协作参考/00_子代理与效率.md)、[02_提交与协作规范.md](01_AI项目协作参考/02_提交与协作规范.md) §02）：
   🔴 `push` 只由所有者本人执行，AI 权限**止于 `commit`**，授权**一次一用**，`--force` 永不在授权内，**子代理零 git 操作**；
   🔴 **子代理权限只有两个来源**：人类直接指定，或 AI 申请并获批（**不可先斩后奏**）；配额默认 3 个，用完即评留还是删；
   🔴 **不看 Token 账单，只看协作效率**——考核的是同一个活用几下干完。
5. **保留资料的适用边界**：个人化材料不直接当作通用模板；恢复清单以实际机器为准，学习示例不等于生产实现。
6. **结构变化同步导航**：增删或移动目录时，同步本 README、`AGENTS.md`、`01_AI项目协作参考/README.md` 与受影响链接。

---

*2026-10-20 结构重构：新增根级 `AGENTS.md`；`01_AI项目协作参考/` 改为"开工前 → 写代码 → 提交"三段式，
原 `代码及路径的规范/` 拆平为 `01_编程规范/` 与 `02_提交与协作规范.md`，§11 子代理抽出为 `00_子代理与效率.md`
（上限 5 → 3），删除两个兼容壳文件；`Basic_Machine_Learning/` 移入 `00_个人笔记/`；
规则改为单点出处，本 README 不再复述细则。提交与协作规范升至 v2.6。*
