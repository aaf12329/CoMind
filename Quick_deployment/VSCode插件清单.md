# VS Code 插件清单与重装恢复

记录日期：2026-10-09。

本次新增 **17 个前端插件**，原有 **11 个插件**保留，安装后共 **28 个**。下表版本来自安装完成后的 VS Code 实际导出，不是推荐版本号。

网站与其他软件入口见 [相关网站.md](相关网站.md)。本文件单独记录 VS Code 插件、使用要点与恢复命令。

## 1. 本次安装的前端插件

| 分类 | 插件与官方 Marketplace 页面 | 扩展 ID | 实测版本 | 用途 |
|---|---|---|---|---|
| 代码检查 | [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) | `dbaeumer.vscode-eslint` | 3.0.34 | 在编辑器中检查 JS / TS 等代码，使用项目的 ESLint 配置。 |
| 格式整理 | [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) | `esbenp.prettier-vscode` | 12.4.0 | 整理 JS、TS、HTML、CSS、JSON 等文件的格式。 |
| 编辑约定 | [EditorConfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig) | `editorconfig.editorconfig` | 0.18.2 | 读取项目的 `.editorconfig`，统一缩进、换行等规则。 |
| CSS 框架 | [Tailwind CSS IntelliSense](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss) | `bradlc.vscode-tailwindcss` | 0.16.0 | Tailwind 类名补全、提示与预览。 |
| CSS 检查 | [Stylelint](https://marketplace.visualstudio.com/items?itemName=stylelint.vscode-stylelint) | `stylelint.vscode-stylelint` | 2.2.1 | 按项目配置检查 CSS 等样式文件。 |
| 路径补全 | [Path Intellisense](https://marketplace.visualstudio.com/items?itemName=christian-kohler.path-intellisense) | `christian-kohler.path-intellisense` | 2.10.0 | 输入文件路径时提供补全。 |
| 包名补全 | [npm Intellisense](https://marketplace.visualstudio.com/items?itemName=christian-kohler.npm-intellisense) | `christian-kohler.npm-intellisense` | 1.4.5 | 导入 npm 包时提供包名提示。 |
| HTML / CSS | [HTML CSS Support](https://marketplace.visualstudio.com/items?itemName=ecmel.vscode-html-css) | `ecmel.vscode-html-css` | 2.0.14 | 在 HTML 等文件中提供 CSS 类名等提示。 |
| 错误展示 | [Error Lens](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens) | `usernamehw.errorlens` | 3.29.0 | 将诊断信息直接显示在出错行附近。 |
| 静态页预览 | [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) | `ritwickdey.liveserver` | 5.7.10 | 启动本地静态网页预览，并随文件变化刷新。 |
| Vue | [Vue (Official)](https://marketplace.visualstudio.com/items?itemName=Vue.volar) | `vue.volar` | 3.3.12 | Vue 单文件组件的语言支持。 |
| React | [ES7+ React/Redux/React-Native snippets](https://marketplace.visualstudio.com/items?itemName=dsznajder.es7-react-js-snippets) | `dsznajder.es7-react-js-snippets` | 4.4.3 | React 等代码片段；JSX / TSX 的基础支持由 VS Code 提供。 |
| Angular | [Angular Language Service](https://marketplace.visualstudio.com/items?itemName=Angular.ng-template) | `angular.ng-template` | 22.2.0 | Angular 模板补全与诊断。 |
| Svelte | [Svelte for VS Code](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode) | `svelte.svelte-vscode` | 110.3.1 | Svelte 文件的语言支持。 |
| Astro | [Astro](https://marketplace.visualstudio.com/items?itemName=astro-build.astro-vscode) | `astro-build.astro-vscode` | 2.16.20 | Astro 文件的语言支持。 |
| 浏览器测试 | [Playwright Test](https://marketplace.visualstudio.com/items?itemName=ms-playwright.playwright) | `ms-playwright.playwright` | 1.1.19 | 在编辑器中运行、调试项目的 Playwright 测试。 |
| 单元测试 | [Vitest](https://marketplace.visualstudio.com/items?itemName=vitest.explorer) | `vitest.explorer` | 1.52.2 | 在编辑器中发现、运行和调试项目的 Vitest 测试。 |

## 2. 原有插件（保留）

| 分类 | 插件与 Marketplace 页面 | 扩展 ID | 实测版本 |
|---|---|---|---|
| 中文界面 | [简体中文语言包](https://marketplace.visualstudio.com/items?itemName=MS-CEINTL.vscode-language-pack-zh-hans) | `ms-ceintl.vscode-language-pack-zh-hans` | 1.131.2026090407 |
| Python | [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | `ms-python.python` | 2026.8.0 |
| Python | [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | `ms-python.vscode-pylance` | 2026.4.1 |
| Python | [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy) | `ms-python.debugpy` | 2026.6.0 |
| Python | [Python Environments](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-python-envs) | `ms-python.vscode-python-envs` | 1.38.0 |
| C / C++ | [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) | `ms-vscode.cpptools` | 1.34.4 |
| 容器 | [Container Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers) | `ms-azuretools.vscode-containers` | 2.5.2 |
| Markdown | [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) | `yzhang.markdown-all-in-one` | 3.6.3 |
| Markdown | [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) | `davidanson.vscode-markdownlint` | 0.62.1 |
| Mermaid | [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) | `bierner.markdown-mermaid` | 1.32.1 |
| Markdown | [NextGen Markdown Previewer](https://marketplace.visualstudio.com/items?itemName=IvanVilchavskyi.nextgen-md-previewer) | `ivanvilchavskyi.nextgen-md-previewer` | 0.4.1 |

## 3. 开始使用

1. 回到 VS Code；如插件提示重载，保存文件后，在命令面板执行 **Developer: Reload Window（开发人员：重新加载窗口）**。
2. **普通 HTML 页面**：打开项目文件夹，在 HTML 文件上右键选择 **Open with Live Server**。
3. **React / Vue / Angular / Svelte / Astro 项目**：按项目 README 或 `package.json` 的脚本启动开发服务器；Live Server 用于静态页，框架项目使用自己的开发服务器。
4. **格式整理**：打开文件，执行 **Format Document With…（使用…格式化文档）**，选择 Prettier；需要时再设为该语言的默认格式化工具。
5. **项目检查**：ESLint、Stylelint、Tailwind 和测试插件需要项目自身的依赖及配置。安装 VS Code 插件不会自动给项目安装框架、测试包或浏览器。

VS Code 自带 JS / TS、HTML / CSS、JSON、Emmet 和 JavaScript 调试等基础功能。此次补齐的是编辑器扩展；项目的 Node.js、npm 依赖和构建脚本仍按各项目记录恢复。

本次没有修改用户全局的格式化、保存自动修复或其他编辑设置。格式化和检查规则优先使用项目约定，避免多个格式化工具同时处理同一文件。

## 4. 重装后批量恢复（PowerShell）

先安装 VS Code，再打开新的 PowerShell 窗口。下面默认恢复全部 **28 个插件**；只装本次前端插件时，按代码中的注释修改一行。

```powershell
# 优先使用 PATH 中的 VS Code；找不到时检查原有 D 盘安装位置。
$taskCodeCommand = Get-Command code -ErrorAction SilentlyContinue
if ($taskCodeCommand) {
    $taskCodeCli = $taskCodeCommand.Source
} elseif (Test-Path -LiteralPath 'D:\Microsoft VS Code\bin\code.cmd') {
    $taskCodeCli = 'D:\Microsoft VS Code\bin\code.cmd'
} else {
    throw '未找到 VS Code，请先安装，或将 taskCodeCli 改为实际的 code.cmd 路径。'
}

# 本次新增：17 个前端插件。
$taskFrontendExtensions = @(
    'dbaeumer.vscode-eslint'
    'esbenp.prettier-vscode'
    'editorconfig.editorconfig'
    'bradlc.vscode-tailwindcss'
    'stylelint.vscode-stylelint'
    'christian-kohler.path-intellisense'
    'christian-kohler.npm-intellisense'
    'ecmel.vscode-html-css'
    'usernamehw.errorlens'
    'ritwickdey.liveserver'
    'vue.volar'
    'dsznajder.es7-react-js-snippets'
    'angular.ng-template'
    'svelte.svelte-vscode'
    'astro-build.astro-vscode'
    'ms-playwright.playwright'
    'vitest.explorer'
)

# 原有：11 个插件。
$taskOriginalExtensions = @(
    'ms-ceintl.vscode-language-pack-zh-hans'
    'ms-python.python'
    'ms-python.vscode-pylance'
    'ms-python.debugpy'
    'ms-python.vscode-python-envs'
    'ms-vscode.cpptools'
    'ms-azuretools.vscode-containers'
    'yzhang.markdown-all-in-one'
    'davidanson.vscode-markdownlint'
    'bierner.markdown-mermaid'
    'ivanvilchavskyi.nextgen-md-previewer'
)

# 只装前端插件时，改为：$taskToInstall = $taskFrontendExtensions
$taskToInstall = $taskFrontendExtensions + $taskOriginalExtensions

# 逐项安装；网络失败时停止，解决连接后可重新运行。
foreach ($taskExtensionId in $taskToInstall) {
    & $taskCodeCli --install-extension $taskExtensionId
    if ($LASTEXITCODE -ne 0) {
        throw "插件安装失败：$taskExtensionId"
    }
}

# 核对实际安装版本。
& $taskCodeCli --list-extensions --show-versions
```

此命令安装运行时 Marketplace 提供的适配版本。要恢复某个记录版本，可使用 `扩展ID@版本`，例如：

```powershell
code --install-extension esbenp.prettier-vscode@12.4.0
```

## 5. 安装台账与验证

| 项目 | 记录 |
|---|---|
| 操作日期 | 2026-10-09 |
| VS Code 版本 | 1.141.0，Windows x64 |
| 安装范围 | 当前用户的默认 VS Code 配置 |
| 安装前 / 后 | 11 / 28 个插件 |
| 本次新增 | 17 个，全部在安装后清单中找到 |
| 原有插件 | 11 个均保留 |
| C 盘剩余空间 | 操作前 120.83 GiB；操作后 120.72 GiB |
| C 盘占用变化 | 该时段约增加 109.30 MiB，包含同期系统活动，非插件精确体积 |
| 实际验证 | 安装命令成功；再次导出插件清单，逐项核对新增与原有 ID |
| 项目运行验证 | 本次只安装与记录插件，未启动前端项目或运行测试 |
| 文档变更 | 新增本文件；`相关网站.md` 保持原样 |

Extension inventory: 17 frontend extensions added; 11 existing extensions preserved; 28 installed in total. Versions were verified through the VS Code CLI on 2026-10-09. The recovery commands above install extensions for the current user's default profile.
