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
| [00_个人笔记](00_个人笔记/) | 私人笔记，含 [前端资料](00_个人笔记/前端开发/README.md) 与 [双框架机器学习资料](00_个人笔记/Basic_Machine_Learning/README.md) | AI 仅对机器学习子目录可按需读取；其他笔记按本人指定主题读取 |
| [01_AI项目协作参考](01_AI项目协作参考/README.md) | AI 协作规范的**唯一出处**：子代理与效率、编程规范、提交与 git 权限 | 从该目录的 `README.md` 索引进，按"开工前 → 写代码 → 提交"三步取用 |
| [02_思考与写作](02_思考与写作/README.md) | 关于学习、语言、AI 工具与个人思考的 seed 原文及润色文章 | 原文在 `seed/`，成稿在 `articles/` 按同主题名配对；保留已有 HTML 排版稿 |
| [Quick_deployment](Quick_deployment/README.md) | 重装恢复、新机迁移、网站入口、VS Code 与 AI 工具插件清单 | 按恢复顺序查阅对应清单 |

### 我现在要做什么？

- **学习机器学习**：先读 [学习路线与运行方式](00_个人笔记/Basic_Machine_Learning/README.md)。覆盖线性回归、MLP、LSTM、迷你 Transformer、音频架构与树模型。
- **阅读前端笔记**：从 [前端学习资料](00_个人笔记/前端开发/README.md) 进入 JS 基础、运行工具、学习路线、CSS 与 React；配套计数器源码已归档。AI 按本人指定的主题读取。
- **阅读文章或继续整理想法**：浏览器可打开 [HTML 文章目录](02_思考与写作/index.html)；原文与成稿配对见 [思考与写作](02_思考与写作/README.md)。12 篇文章分别保留 Markdown 和按各自语气排版的 HTML。
- **重装系统或换电脑**：先读 [恢复流程](Quick_deployment/README.md)，再查 [相关网站](Quick_deployment/相关网站.md)、[VS Code 插件](Quick_deployment/VSCode插件清单.md) 和 [AI 插件](Quick_deployment/AI插件.md)。
- **让 AI 接手工作**：AI 先读根目录 [`AGENTS.md`](AGENTS.md)，再进 [`01_AI项目协作参考/README.md`](01_AI项目协作参考/README.md)。

## AI 导引 / References for AI

> 🔴 **AI 的唯一入口是 [`AGENTS.md`](AGENTS.md)**，那是根级锚点，含四条红线与三步流程。
> 下面这张表是给人看的速查；细则一律以 `01_AI项目协作参考/` 内的文件为准。

| 你要做什么 / 找什么 | 去哪里 | 备注 |
|---|---|---|
| 开工前决定要不要开子代理、开几个 | [00_子代理与效率.md](01_AI项目协作参考/00_子代理与效率.md) | 🔴 同时最多 **3 个**；须申请获批，申请里附所有者原话 |
| 写代码 / 按语言查规范 | [01_编程规范/代码规范.md](01_AI项目协作参考/01_编程规范/代码规范.md) | 按语言跳转；工程项目另读工程架构规范，学习资料另读学习资料规范 |
| 提交、push、看 git 权限 | [02_提交与协作规范.md](01_AI项目协作参考/02_提交与协作规范.md) | 🔴 权限在 §02，提交格式在 §03/§04 |
| GitHub 推送被邮箱隐私保护 GH007 拦截 | [02_提交与协作规范.md](01_AI项目协作参考/02_提交与协作规范.md) | §08.1：预防、未发布提交修复与推送验证 |
| 所有 Agent、所有话题的信息上传边界与抽象建模 | [统一规则 §07](01_AI项目协作参考/02_提交与协作规范.md#07--仓库卫生) | 在向外传输材料前执行；输入法检查按同一出处办理 |
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
│   ├── 前端开发/                         前端学习资料与独立计数器示例（按本人指定读取）
│   │   ├── README.md                     阅读索引、比对整合与验证说明
│   │   ├── JavaScript基础.md / 语言运行与工程工具.md
│   │   ├── JavaScript科技树_学习清单.md / CSS选择器.md
│   │   ├── React教程_调用关系与数据流.md
│   │   └── 示例/react-counter-demo/       源码、配置、锁文件与历史截图
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
├── 02_思考与写作/                        原文与润色文章一一对应
│   ├── README.md                         按主题阅读与配对索引
│   ├── index.html                        浏览器文章目录（12 篇 HTML 阅读版）
│   ├── seed/                             本人原文
│   └── articles/                         润色文章，按同主题名配对（Markdown / HTML）
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

同日继续整理为 `seed/` 与 `articles/` 两个目录：三组同名原文与文章，归并重复 TXT，从原始草稿及既有口述补齐 seed；旧版本通过 Git 历史追溯。

2026-10-25：将桌面四份 JS / React 资料与旧前端整合笔记比对，合并为 [前端学习资料](00_个人笔记/前端开发/README.md)。运行工具、基础语法、学习路线、CSS 与 React 按主题维护；JavaScript 临时草案只在规范目录保存一份，明确采用后适用。

## 使用与维护约定 / Ground rules

> 协作规则的**正文只有一处**：`01_AI项目协作参考/`。本表只列边界，不复述细则；
> 逐条出处见 [01_AI项目协作参考/README.md](01_AI项目协作参考/README.md) 第三节。

1. **个人笔记按授权读取**：`00_个人笔记/` 不作为 AI 默认参考材料；
   🔴 唯一例外是 `00_个人笔记/Basic_Machine_Learning/`——它是学习资料，AI 可按需读取。
2. **遵守过滤标记**：分享本库、批量提供给 AI 或生成公开材料时，跳过带【请过滤本文件】标记的文件；AI 不代删标记。
3. **保护个人信息与凭据**：提交和公开内容中过滤真实姓名、学校、单位、地理位置等身份信息；密钥、账号凭据、环境和缓存不入库。
4. **红线与权限**（细则见 [`AGENTS.md`](AGENTS.md) 与 [00_子代理与效率.md](01_AI项目协作参考/00_子代理与效率.md)、[02_提交与协作规范.md](01_AI项目协作参考/02_提交与协作规范.md) §02 / §07）：
   🔴 `push` 只由所有者本人执行，AI 权限**止于 `commit`**，授权**一次一用**，`--force` 永不在授权内，**子代理零 git 操作**；
   🔴 **子代理权限只有两个来源**：人类直接指定，或 AI 申请并获批（**不可先斩后奏**）；配额默认 3 个，用完即评留还是删；
   🔴 **不看 Token 账单，只看协作效率**——考核的是同一个活用几下干完。
   🔴 **具体政治内容的上传限制与抽象建模边界**，按 [§07](01_AI项目协作参考/02_提交与协作规范.md#07--仓库卫生)执行，覆盖所有 Agent、所有话题。
5. **保留资料的适用边界**：个人化材料不直接当作通用模板；恢复清单以实际机器为准，学习示例不等于生产实现。
6. **结构变化同步导航**：增删或移动目录时，同步本 README、`AGENTS.md`、`01_AI项目协作参考/README.md` 与受影响链接。

---

*2026-10-20 结构重构：新增根级 `AGENTS.md`；`01_AI项目协作参考/` 改为"开工前 → 写代码 → 提交"三段式，
原 `代码及路径的规范/` 拆平为 `01_编程规范/` 与 `02_提交与协作规范.md`，§11 子代理抽出为 `00_子代理与效率.md`
（上限 5 → 3），删除两个兼容壳文件；`Basic_Machine_Learning/` 移入 `00_个人笔记/`；
规则改为单点出处，本 README 不再复述细则。提交与协作规范升至 v2.6。*
