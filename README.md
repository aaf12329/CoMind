<div align="center">

# 共知 · CoMind

**个人知识、学习实践与 AI 协作参考库**

保存我理解世界的过程，也为 AI 与我协作提供共同背景。

*Personal knowledge, learning practice, and references for working with AI.*

</div>

---

## 从这里开始 / Start here

CoMind 既保存个人积累的笔记与学习实践，也保存供 AI 使用的背景、规范和工作方法。原 `Basic_Machine_Learning` 与 `Quick_deployment` 已归并到本仓库，分别保留独立目录及原提交历史。

| 区域 | 内容 | 阅读方式 |
|---|---|---|
| [00_个人笔记](00_个人笔记/) | 私人学习笔记、前端整理与常用指令 | 所有者自用；AI 不主动读取，除非所有者点名 |
| [01_AI项目协作参考](01_AI项目协作参考/) | 协作背景、代码规范、资料检索方法、项目接手手册写法 | AI 按下方导引按需读取 |
| [Basic_Machine_Learning](Basic_Machine_Learning/README.md) | PyTorch × TensorFlow 双框架学习代码、原理说明与参考图 | 从子目录 README 选择主题，再进入各章节 |
| [Quick_deployment](Quick_deployment/README.md) | 重装恢复、新机迁移、网站入口、VS Code 与 AI 工具插件清单 | 按恢复顺序查阅对应清单 |

### 我现在要做什么？

- **学习机器学习**：先读 [学习路线与运行方式](Basic_Machine_Learning/README.md)。覆盖线性回归、MLP、LSTM、迷你 Transformer、音频架构与树模型。
- **重装系统或换电脑**：先读 [恢复流程](Quick_deployment/README.md)，再查 [相关网站](Quick_deployment/相关网站.md)、[VS Code 插件](Quick_deployment/VSCode插件清单.md) 和 [AI 插件](Quick_deployment/AI插件.md)。
- **让 AI 接手工作**：先按下一节了解协作背景，再选取任务对应的规范与资料。

## AI 导引 / References for AI

只读取当前任务需要的材料，不把整个仓库一次性加载。个人笔记与供 AI 参考的资料仍保持各自的使用边界。

| 你要做什么 / 找什么 | 去哪里 | 备注 |
|---|---|---|
| 了解所有者背景与协作偏好 | [用户画像_协作偏好.md](01_AI项目协作参考/用户个人信息参考/用户画像_协作偏好.md) | 带【请过滤本文件】标记；分享、批量处理或生成公开材料时跳过整个文件。正常协作阅读时跳过标记行，不代删标记。 |
| 写代码、整理仓库或提交 | [代码规范.md](01_AI项目协作参考/代码及路径的规范/代码规范.md) | 短路由入口；按任务选择通用提交规范、工程或学习场景，以及对应语言规范。 |
| **push 或派子代理**（动仓库、开多代理前必读） | [提交与协作规范.md](01_AI项目协作参考/代码及路径的规范/通用规范/提交与协作规范.md) 的 **§02 权限铁律** 与 **§11 子代理协作** | 🔴 AI 权限止于 commit；push 需所有者授权且**一次授权只用一次**；子代理数量与职责由所有者定，AI 申请须写明**数量 + 每个的作用**。 |
| 整理算法示例或学习资料 | [学习资料代码规范.md](01_AI项目协作参考/代码及路径的规范/通用规范/学习资料代码规范.md) | 保留示例独立性、真实调用树、输入形状及代码与图文对应，不自动工程化。 |
| 学习与比较机器学习实现 | [Basic_Machine_Learning/README.md](Basic_Machine_Learning/README.md) | 子目录有章节说明、两套实现与配套参考图；音频部分为架构学习骨架。 |
| 恢复开发环境、编辑器或 AI 工具 | [Quick_deployment/README.md](Quick_deployment/README.md) | 清单中的版本与本机路径是历史快照，恢复时按新机器调整。 |
| 配置 conda | [conda 路径注意事项](01_AI项目协作参考/代码及路径的规范/语言规范/Python/conda路径注意事项.md) | 仅使用 conda 时读取；环境以机器上的实际状态为准。 |
| 接手或新建项目 | [项目接手手册：模板与写法](01_AI项目协作参考/项目接手手册参考/模板与写法.md) | 使用骨架与实例组织交接信息，具体项目状态以其仓库为准。 |
| 调研 GitHub | [GitHub 搜索与排查](01_AI项目协作参考/信息搜索路径参考/GitHub搜索_问题与解决手法.md) | Windows / Git Bash 查询、限流与降级方法。 |
| 找论文与全文 | [学术检索与获取路径](01_AI项目协作参考/信息搜索路径参考/学术检索_问题与获取路径.md) | 核实来源、机构库与合法全文获取。 |

思考类笔记与个人环境配置总表仍以所有者指定的位置为准，不因本次归并自动收入或读取。

## 目录结构 / Layout

```text
CoMind/
├── README.md
├── 00_个人笔记/                         所有者自用
├── 01_AI项目协作参考/                   AI 按任务取用
│   ├── 代码及路径的规范/
│   │   ├── 代码规范.md                  规范路由
│   │   ├── 学习资料代码规范.md          兼容入口
│   │   ├── conda路径注意事项.md         兼容入口
│   │   ├── 通用规范/                    提交、工程、学习资料
│   │   ├── 语言规范/                    Python、JS、TS、PowerShell、Bash
│   │   └── 专项规范/                    LLM / Agent
│   ├── 信息搜索路径参考/
│   ├── 项目接手手册参考/                 接手手册怎么写（模板与实例）
│   └── 用户个人信息参考/                分享时遵守文件过滤标记
├── Basic_Machine_Learning/
│   ├── README.md
│   ├── Create-Venv.ps1                  自选 Python，在本目录创建 .venv
│   ├── 01_linear_regression/
│   ├── 02_mlp/
│   ├── 03_lstm/
│   ├── 04_mini_transformer/
│   ├── 05_audio_models/
│   └── 06_tree_models/
│       各章节分别保留代码、README、figs_pytorch/ 和 figs_tensorflow/
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

机器学习示例的命令以 `Basic_Machine_Learning/` 为工作目录。使用 [Create-Venv.ps1](Basic_Machine_Learning/Create-Venv.ps1) 选择本机 Python，在 **`Basic_Machine_Learning/.venv/`** 创建环境，再自行安装相应依赖。脚本不下载 Python 或自动安装框架；具体参数见 [学习资料 README](Basic_Machine_Learning/README.md)。运行学习脚本可能重新生成章节中的参考图。

```powershell
cd Basic_Machine_Learning
powershell -NoProfile -ExecutionPolicy Bypass -File ./Create-Venv.ps1
# 自行安装对应依赖后：
./.venv/Scripts/python.exe 01_linear_regression/01_linear_regression_pytorch.py
```

部署资料从 [Quick_deployment/README.md](Quick_deployment/README.md) 开始。账号凭据、完整私有配置、模型及数据集按各项目自己的要求另行备份和恢复。

## 归并记录 / Consolidation

2026-10-18 起，以 [CoMind](https://github.com/aaf12329/CoMind) 为统一入口：

| 原仓库 | 本库位置 | 归并时的源版本 |
|---|---|---|
| [Basic_Machine_Learning](https://github.com/aaf12329/Basic_Machine_Learning) | [Basic_Machine_Learning/](Basic_Machine_Learning/README.md) | `7cb9cb1` |
| [Quick_deployment](https://github.com/aaf12329/Quick_deployment) | [Quick_deployment/](Quick_deployment/README.md) | `6bb8778` |

保留原项目的目录结构、学习代码、配套图文和提交历史。后续统一在本库对应目录维护；原仓库链接用于追溯来源。本次归并只更新资料组织、导航及运行入口说明，不把历史实验记录视为本次复现结果。

The two repositories were consolidated into their own subdirectories with their original Git histories preserved. CoMind is the main entry point for future maintenance. This consolidation does not represent a new reproduction of the recorded experiments.

## 使用与维护约定 / Ground rules

1. **个人笔记按授权读取**：`00_个人笔记/` 不作为 AI 默认参考材料。
2. **遵守过滤标记**：分享本库、批量提供给 AI 或生成公开材料时，跳过带【请过滤本文件】标记的文件；AI 不代删标记。
3. **保护个人信息与凭据**：提交和公开内容中过滤真实姓名、学校、单位、地理位置等身份信息；敏感内容按所有者已有明确授权处理。密钥、账号凭据、环境和缓存不入库。
4. **AI 权限止于 commit**：`add` / `commit` 归 AI，🔴 **`push` 只由所有者本人执行**；
   所有者可以授权，但**一次授权只能用一次**（推完即失效，不延展、不跨仓库/分支、不转授给子代理）；
   `--force` 永不在授权范围内。
5. **一切授权一次有效**：所有者给的每一条授权（push、开子代理、任何超默认操作）**只用一次**，
   干完即失效，不续用、不延展、不类推、不转授；要再干重新申请。
6. **子代理须授权**：**数量与职责由所有者定**（人类可直接用自然语言指定，AI 照办、不得增减改职责）；
   AI 申请须写清 **申请几个 / 用来干什么 / 开启之后有几个 Agent 干活**，批准后才派。
   出处：提交与协作规范 §02、§11。
7. **不看 token 账单，只看结果**：🔴 所有者明确不核算 token/调用成本——**不许为省钱缩水**
   （该查证的查、该跑的测试跑、该派的子代理派），也不许为用满预算堆无用工作；
   唯一标准是结果对不对、够不够，判断依据是真实验证证据而非"我做完了"。
   成本不受限 ≠ 权限不受限，第 4–6 条照旧。
8. **保留资料的适用边界**：个人化材料不直接当作通用模板；恢复清单以实际机器为准，学习示例不等于生产实现。
9. **结构变化同步导航**：增删或移动目录时，同步主 README、子目录说明和受影响链接。

AI 建议读取顺序：协作背景 → 代码规范短路由 → 当前任务对应的正文与项目资料。学习资料仍采用独立示例，不自动引入共享工程层或统一训练入口。

*2026-10-19 提交与协作规范升 v2.3：push 写成铁律 + 一次授权一次用（§02），新增子代理协作（§11）；本 README 规则 4/5、导引区与代码规范入口、接手手册同步。*
