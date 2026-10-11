# AI Agent 插件清单与重装恢复

记录日期：2026-10-11。

用于重装系统、重装 Agent 或换机器后恢复插件与工具配置。当前已整理 DeepSeek Harness 和 Codex，其他 Agent 的章节留空，后续直接补充。

网站与软件入口见 [相关网站.md](相关网站.md)，编辑器扩展见 [VSCode插件清单.md](VSCode插件清单.md)。

## 1. DeepSeek Harness（DSH）

来源：桌面 `我的DSH插件.md`。安装状态、MCP 工具数与握手结果沿用原清单记录；本次重新读取本机已安装包的清单，核对了 8 个原生插件的版本和仓库地址。

### 1.1 本机环境

| 项目 | 记录值 |
| --- | --- |
| DSH 版本 | `0.2.0-rc.2` |
| Profile | `desktop` |
| 配置目录 | `C:\Users\Guards\.dsh\profiles\desktop` |
| 宿主目录 | `D:\DSH` |
| CLI | `D:\DSH\resources\runtime\cli\bin\dsh.cmd` |
| DSH 自带 Node | `24.18.1` |
| DSH 自带 pnpm | `11.7.0` |

**本机插件命令使用 `--profile desktop`。** 重装后如果软件路径或 Profile 改变，先调整下文命令与配置里的路径。

### 1.2 已安装原生插件（8 个）

| 插件与仓库 | 包名 | 记录版本 | 用途与配置要点 |
| --- | --- | --- | --- |
| [DSH Better Sidebar](https://github.com/omdsh-dev/DSH-better-sidebar) | `dsh-better-sidebar` | `0.24.1` | 可编辑代码面板、文件树、Git diff、暂存/提交、后台任务查看。 |
| [DSH LSP Actions](https://github.com/PerryLink/dsh-lsp-actions) | `dsh-lsp-actions` | `0.5.12` | 语言服务器诊断、符号检索、格式化、重命名；本机已配置 TS 服务器，需连同服务器配置恢复。 |
| [DSH Checkpoint Rewind](https://github.com/PerryLink/dsh-checkpoint-rewind) | `dsh-checkpoint-rewind` | `0.6.24` | 工作区、会话、配置检查点与回退。 |
| [DSH Doublecheck](https://github.com/PerryLink/dsh-doublecheck) | `dsh-doublecheck` | `0.9.24` | 需求核对、测试证据、交付检查、审查子代理。 |
| [DSH WSL Workspace](https://github.com/dsh-wsl-workspace-maintainers/dsh-wsl-workspace) | `dsh-wsl-workspace` | `0.7.5` | 在 Windows 中操作 WSL 项目；WSL 发行版与项目工具链另行恢复。 |
| [DSH Visualize](https://github.com/Nagi-ovo/dsh-visualize) | `@nagi-ovo/dsh-visualize` | `0.1.4` | 对话内交互式 HTML 卡片，用于算法、图表等演示。 |
| [DSH Boot Animation](https://github.com/lxj5820/dsh-boot-animation) | `dsh-boot-animation` | `0.2.0` | 启动动画，原环境已有。 |
| [DeepSeek Balance Whale Widget](https://github.com/MeteorNOX/DeepSeek-Balance-Whale-Widget) | `dsh-whale-widget` | `0.3.18` | DeepSeek 余额小鲸鱼挂件，原环境已有。 |

原清单记录：以上插件均注册在 Profile 的 `dsh.profile.bundles` 中，并自带 `cordis.patch.yml`。恢复时安装包与 Profile 配置都要齐全。

### 1.3 MCP 扩展（2 个启用、1 个备用）

| 服务与仓库 | serverName | 传输 | 原记录工具数 | 状态与用途 |
| --- | --- | --- | --- | --- |
| [Context7](https://github.com/upstash/context7) | `context7` | `streamable-http` | 2 | 已启用；查询库的版本文档与 API 示例。 |
| [Playwright MCP](https://github.com/microsoft/playwright-mcp) | `playwright` | `stdio`（npx） | 25 | 已启用；操作网页、复现交互问题。 |
| [LeanToken](https://github.com/morluto/leantoken) | `leantoken` | `stdio` | 9 | 已全局安装 `0.1.28`，未默认启用；本地代码索引与检索。 |

- **Context7：**端点为 `https://mcp.context7.com/mcp`，API Key 可选。
- **Playwright MCP：**原配置通过 `npx -y @playwright/mcp@latest` 启动。原清单未记录该 MCP 包的具体版本，`@latest` 会随时间变化。
- **LeanToken：**`--root` 指向固定仓库，无法自动跟随当前会话工作区。需要时启用 `cordis.patch.yml` 中对应的注释段，并将 `--root` 改为目标项目绝对路径。

MCP 配置在 `C:\Users\Guards\.dsh\profiles\desktop\cordis.patch.yml`，每个服务一条 `@deepseek-ai/dsh-mcp-client` 配置；工具命名为 `mcp__<serverName>__<工具名>`。

配置参考：[DSH 官方 MCP 客户端中文说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/mcp/mcp-client/README.zh.md)。完整启动参数与个人凭据从个人配置备份恢复。

### 1.4 TS 语言服务器及单独的 TypeScript 环境

| 组件与地址 | 安装位置 | 记录版本 | 恢复要点 |
| --- | --- | --- | --- |
| [TypeScript Language Server](https://github.com/typescript-language-server/typescript-language-server) | npm 全局 | `6.0.2` | LSP 服务本体；`dsh-lsp-actions` 需要显式配置服务器。 |
| [TypeScript](https://github.com/microsoft/TypeScript) | `C:\Users\Guards\.dsh\tools\ts-lsp` | `5.9.3` | 单独提供 TSLS 所需的 `tsserver.js`。 |

原清单记录：本机全局 TypeScript 为 `7.0.2`，该环境未提供 TSLS 所需的 `tsserver.js`；TSLS 也未使用全局 TS，因此单独保留 TypeScript `5.9.3`，并显式指定路径。

```yaml
initializationOptions:
  tsserver:
    path: "C:\\Users\\Guards\\.dsh\\tools\\ts-lsp\\node_modules\\typescript\\lib\\tsserver.js"
```

原验证日志为 `Using Typescript version (user-setting) 5.9.3 from path ...`。重装后按新的用户目录修改路径。

### 1.5 重装恢复命令（PowerShell）

以下使用原清单中的具体版本恢复插件快照；升级 DSH 后先核对插件兼容性。

```powershell
# DSH 不在 PATH 时使用完整 CLI 路径。
$taskDshCli = 'D:\DSH\resources\runtime\cli\bin\dsh.cmd'
if (-not (Test-Path -LiteralPath $taskDshCli)) {
    throw '未找到 DSH CLI，请先安装 DSH，并修改 taskDshCli 为实际路径。'
}

# 1. 恢复 8 个原生插件。
$taskDshPackages = @(
    'dsh-better-sidebar@0.24.1'
    'dsh-lsp-actions@0.5.12'
    'dsh-checkpoint-rewind@0.6.24'
    'dsh-doublecheck@0.9.24'
    'dsh-wsl-workspace@0.7.5'
    '@nagi-ovo/dsh-visualize@0.1.4'
    'dsh-boot-animation@0.2.0'
    'dsh-whale-widget@0.3.18'
)
foreach ($taskDshPackage in $taskDshPackages) {
    & $taskDshCli plugin --profile desktop add $taskDshPackage
    if ($LASTEXITCODE -ne 0) {
        throw "插件安装失败：$taskDshPackage"
    }
}

# 2. 恢复语言服务器；以下 npm 命令需要可用的 Node.js/npm 环境。
npm install --global typescript-language-server@6.0.2

# 3. 单独恢复 TypeScript 5.9.3，供语言服务器使用。
$taskTsLspDirectory = Join-Path $env:USERPROFILE '.dsh\tools\ts-lsp'
New-Item -ItemType Directory -Force -Path $taskTsLspDirectory | Out-Null
Push-Location $taskTsLspDirectory
try {
    if (-not (Test-Path -LiteralPath 'package.json')) {
        npm init -y
    }
    npm install --save-exact typescript@5.9.3
} finally {
    Pop-Location
}

# 4. 恢复备用代码检索工具；默认仍保持未启用。
npm install --global leantoken@0.1.28

# 5. 从个人备份恢复 cordis.patch.yml 中的 LSP servers 与 MCP 条目。
# 核对 tsserver.path、MCP 命令路径、LeanToken --root 等机器相关路径。
# 重启 DSH 后检查插件、MCP 与 TS 语言服务器是否正常。
```

### 1.6 原安装记录中的恢复要点

1. **LSP 插件优先用 npm 已构建包。**原 GitHub 安装方式触发 `prepare` 脚本，在本机受到 pnpm 构建脚本策略与命令入口问题影响；当时改用 `dsh-lsp-actions@0.5.12` 的 npm 包完成安装。
2. **Windows MCP 启动配置要完整恢复。**原记录使用完整命令路径（例如 `C:\nodejs\npx.cmd`）。恢复时同时核对 `command`、`args` 及 DSH 的启动方式，避免只恢复包名后丢失可运行的配置。
3. **TSLS 的 TypeScript 路径要显式保存。**恢复 TypeScript `5.9.3` 和 `initializationOptions.tsserver.path`；如果工作区的 TypeScript 无法提供 `tsserver.js`，不要只依赖自动发现。

### 1.7 配置备份与回退

| 内容 | 原记录位置 / 文件 |
| --- | --- |
| 安装前 Profile 备份 | `C:\Users\Guards\.dsh\profiles\_backup-desktop-20261011-181257` |
| 需恢复的文件 | `package.json`、`cordis.yml`、`cordis.patch.yml`、`pnpm-workspace.yaml`、`pnpm-lock.yaml` |
| 恢复目标 | `C:\Users\Guards\.dsh\profiles\desktop` |

回退时将备份中的 5 个文件复制回目标目录，再使用原 CLI 执行 `plugin --profile desktop install` 并重启 DSH。该路径是本机备份位置，重装前还需把实际备份文件保存到重装后可访问的位置。

### 1.8 尚未安装 / 暂缓

| 项目与地址 | 原清单中的原因 |
| --- | --- |
| [DSH Context Doctor](https://github.com/Zhenyu98/dsh-context-doctor) | 当时存在 DSH `0.2.0-rc.2` + Windows 兼容问题，见 [Issue #16](https://github.com/Zhenyu98/dsh-context-doctor/issues/16)；后续安装前重新核对适配情况。 |
| [Flameox](https://github.com/morluto/flameox) | 性能分析工具，遇到性能问题再考虑；原清单记录 Python 要求 ≥3.12，Windows 初装不列优先项。 |
| 大型多 Agent / 长期记忆套装 | 先观察现有工程工作流的瓶颈，再按需补充。 |

### 1.9 配套语言工具

| 语言 | 工具与地址 | 原清单记录的本机状态 |
| --- | --- | --- |
| C | [clangd](https://clangd.llvm.org/)、[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) | 未安装。 |
| Python | [Ruff](https://github.com/astral-sh/ruff)、[pytest](https://github.com/pytest-dev/pytest) | Ruff 位于 `D:\Anaconda\Scripts\ruff.exe`；pytest 安装情况未单独记录。 |
| JS / TS | [TypeScript](https://github.com/microsoft/TypeScript)、[ESLint](https://github.com/eslint/eslint)、[Vitest](https://github.com/vitest-dev/vitest) | 全局 TypeScript `7.0.2`；项目工具 ESLint/Vitest 未安装，编辑器扩展另见 VS Code 清单。 |

原清单的验证记录：8 个原生插件已注册；3 个 MCP 服务分别完成握手并列出 2/25/9 个工具，LeanToken 随后保持备用；TS 语言服务器完成 LSP initialize 握手。

## 2. Codex

记录环境：Windows 上的 Codex 桌面会话，核对日期为 2026-10-11。以下依据本机配置、插件清单文件和本会话可用能力整理；版本是本机快照。本次核对安装记录，未逐项重新运行插件。

### 本机配置中启用的插件（11 个）

以下插件均在 `C:\Users\Guards\.codex\config.toml` 的 `[plugins."插件ID"]` 下记录了 `enabled = true`，并已核对本地 `.codex-plugin/plugin.json`。地址列提供官方使用或恢复入口；本机清单中部分仓库地址属于内部仓库，没有作为公开 GitHub 安装地址收录。

| 插件 ID | 本机版本 | 用途 | 地址与恢复说明 |
| --- | --- | --- | --- |
| `codex-app-tools@openai-bundled` | `0.1.6` | Codex 应用内文件预览、Git 工作树、PR 附件、聊天管理、自动化与 LaTeX 编译等工具。 | [插件说明](https://developers.openai.com/plugins/quickstart)；应用集成工具，恢复后以应用提供的能力为准。 |
| `browser@openai-bundled` | `26.1007.21434` | 浏览器操作相关能力，可用于页面检查、交互与本地网站验证。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复插件后核对浏览器工具是否可用。 |
| `unified-computer-use@openai-bundled` | `26.1007.21434` | 应用管理的浏览器自动化运行时；本会话提供 `cua_repl`。 | [插件说明](https://developers.openai.com/plugins/quickstart)；运行时由应用提供。 |
| `computer-use@openai-bundled` | `26.1007.21434` | Windows 桌面操作技能；具体可控范围取决于会话支持，本次未重新实测桌面操作。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复插件并检查其技能与运行时。 |
| `code-review@openai-bundled` | `26.1007.21434` | PR/CI 相关能力；当前会话可查询 CI 检查诊断。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复插件后按需要连接仓库。 |
| `visualize@openai-bundled` | `1.0.49` | 交互图表、流程图、模拟与 UI 预览，可辅助理解算法和工程结构。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 Visualize。 |
| `documents@openai-primary-runtime` | `26.1009.11318` | 创建、编辑、渲染和验证 Word 文档。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 Documents。 |
| `pdf@openai-primary-runtime` | `26.1009.11318` | PDF 读取、生成、页面渲染与版面检查。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 PDF。 |
| `spreadsheets@openai-primary-runtime` | `26.1009.11318` | 表格生成、数据分析、公式与图表；另提供 Excel Live Control 技能。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 Spreadsheets；实时 Excel 操作需相应连接会话。 |
| `presentations@openai-primary-runtime` | `26.1009.11318` | 演示文稿创建、编辑、渲染和检查。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 Presentations。 |
| `template-creator@openai-primary-runtime` | `26.1009.11318` | 将参考文档、幻灯片或表格整理为可复用的个人模板技能。 | [插件说明](https://developers.openai.com/plugins/quickstart)；恢复 Template Creator。 |

### 当前会话提供的其他插件（3 个）

下列插件的 Skills/工具出现在本会话能力列表，本地也存在对应清单文件；未在上述本机配置中发现显式启用项，因此单独记录，不计入 11 个本机启用插件。

| 插件 ID | 本地清单版本 | 用途 | 地址与恢复说明 |
| --- | --- | --- | --- |
| `plugin-management@openai-curated-remote` | `0.1.0` | 搜索插件、查看权限与依赖、管理连接。 | [插件说明](https://developers.openai.com/plugins/quickstart)；检查新会话是否提供 Plugin Management。 |
| `sites@openai-curated-remote` | `1.0.1` | 构建、修改并部署托管网站。 | [插件说明](https://developers.openai.com/plugins/quickstart)；检查 Sites 技能与连接是否可用。 |
| `work-pets@openai-curated-remote` | `0.1.6` | ChatGPT Work 动画宠物的创建、更新与管理。 | [插件说明](https://developers.openai.com/plugins/quickstart)；检查 Pets 技能与工具是否可用。 |

此外，本地缓存中有 `openai-templates@openai-curated-remote` 的 `0.1.1` 清单，未确认本机启用或当前会话加载，仅作缓存记录。

### 系统 Skills

以下 4 个技能在本会话可用，本地位于 `C:\Users\Guards\.codex\skills\.system\`，未记录独立版本。它们由系统提供；插件附带的 Skills 已按上方插件归类。

| Skill | 用途 | 地址与恢复说明 |
| --- | --- | --- |
| `openai-docs` | 查询 OpenAI/Codex 的配置、模型、API 和使用文档。 | [官方文档](https://developers.openai.com/)；检查系统技能是否提供。 |
| `skill-creator` | 创建或更新自定义 `SKILL.md` 与辅助资源。 | [Skills 文档](https://learn.chatgpt.com/docs/build-skills)；检查系统技能是否提供。 |
| `skill-installer` | 从精选列表或 GitHub 仓库安装技能。 | [Skills 文档](https://learn.chatgpt.com/docs/build-skills)；需要额外技能时由安装器处理。 |
| `imagegen` | 生成或编辑图片，可用于工程文档插图和界面素材。 | [Skills 文档](https://learn.chatgpt.com/docs/build-skills)；检查技能及图片生成工具是否可用。 |

同一系统目录还存在 `review-agent/SKILL.md`，但它未列入本会话可用技能，因此仅记录文件存在，不视为当前已加载。

### MCP 与编程基础工具

| 工具 / 服务 | 核对状态 | 用途与恢复方式 |
| --- | --- | --- |
| `node_repl` MCP | 本机配置存在 `[mcp_servers.node_repl]`，本会话有对应工具命名空间。 | 持久化 JavaScript 执行环境；换机器后需核对启动命令、运行时路径与环境配置，不能仅复制旧机器路径。[MCP 配置文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。 |
| 插件附带的 MCP | 随相应插件提供，可能不出现在独立的 `mcp_servers` 配置中。 | 例如 Codex App Tools、浏览器运行时和 Code Review；随插件恢复并检查工具可用性。[MCP 插件说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。 |
| Shell、文件编辑与本地 Git | 当前会话可用。 | 执行 PowerShell、读取和修改代码、搜索文件、运行构建与测试、提交和推送仓库；属于运行环境基础工具。 |

工程用途上，C/Python/JS/TS 的构建、测试和静态检查由项目工具链配合 Shell 执行；语言服务器、编译器和测试框架仍按项目安装。DSH 章节中的 Context7、Playwright、LeanToken 不代表已安装到 Codex。

### 环境与恢复记录

本机核对位置：

- 配置：`C:\Users\Guards\.codex\config.toml`。
- 插件缓存：`C:\Users\Guards\.codex\plugins\cache\<来源>\<插件>\<版本>\`。
- 插件清单：缓存目录中的 `.codex-plugin\plugin.json`。
- 系统技能：`C:\Users\Guards\.codex\skills\.system\`。

恢复步骤：

1. 安装并登录 Codex，按照上表插件 ID 和名称核对所需插件、技能与工具。目录插件可通过插件页面安装；应用集成能力以新版本实际提供的内容为准。[官方插件说明](https://developers.openai.com/plugins/quickstart)。
2. 保留本清单中的 ID 和版本作为核对依据。缓存文件存在不等于已启用；不将缓存目录直接当作安装包。
3. 将个人配置在本地备份，恢复时重新核对机器路径和 MCP 启动参数；账号凭据、密钥与带敏感环境变量的完整配置不随此公开仓库提交。
4. 单独备份自己编写的 Skills。当前官方文档使用用户目录 `~/.agents/skills` 和项目目录 `.agents/skills` 发现自定义技能；本机 `.codex/skills/.system` 是系统技能位置。恢复到新版本时按其文档选择目录。[Skills 加载位置](https://learn.chatgpt.com/docs/build-skills)。
5. 新开会话检查插件与技能是否出现；MCP 再核对服务能否启动、工具能否列出，并在实际项目中验证构建和测试命令。

此处记录的是当前 Codex 环境，未新增安装插件或改动本机插件配置。

## 3. Claude Code

<!-- 留空。 -->

| 插件 / 工具 | 地址 | 版本 | 用途 | 状态 / 恢复方式 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### 环境与恢复记录

<!-- 留空。 -->

## 4. Gemini CLI

<!-- 留空。 -->

| 插件 / 工具 | 地址 | 版本 | 用途 | 状态 / 恢复方式 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### 环境与恢复记录

<!-- 留空。 -->

## 5. OpenCode

<!-- 留空。 -->

| 插件 / 工具 | 地址 | 版本 | 用途 | 状态 / 恢复方式 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### 环境与恢复记录

<!-- 留空。 -->

## 6. 其他 Agent

<!-- 留空：可复制下表，按新的 Agent 名称新增章节。 -->

| 插件 / 工具 | 地址 | 版本 | 用途 | 状态 / 恢复方式 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### 环境与恢复记录

<!-- 留空。 -->
