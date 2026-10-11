# AI Agent 插件清单与重装恢复

记录日期：2026-10-11。

用于重装系统、重装 Agent 或换机器后恢复插件与工具配置。当前已整理 DeepSeek Harness，其他 Agent 的章节留空，后续直接补充。

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

<!-- 留空：后续补充此 Agent 的实际插件、MCP、Skills、版本与恢复方式。 -->

| 插件 / 工具 | 地址 | 版本 | 用途 | 状态 / 恢复方式 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### 环境与恢复记录

<!-- 留空。 -->

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
