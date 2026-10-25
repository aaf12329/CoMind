# JavaScript 科技树与学习清单

整理日期：2026-10-12。目标：从理解 JavaScript 的运行和调用关系，逐步做到独立开发 React 页面，再按需要进入 Next.js 或 Node.js 服务端开发。

这是一份学习路线。树中的连线表示知识依赖和可选方向，不是程序的实际调用树，也不是所有技术都必须学习。完成一个节点的标准是能解释、能动手、能验证结果；只看过教程不算完成。

[返回前端学习索引](README.md) · [基础语法](JavaScript基础.md) · [运行与工具](语言运行与工程工具.md) · [React 数据流](React教程_调用关系与数据流.md) · [CSS 选择器](CSS选择器.md)

## 1. 先看总图

~~~text
JavaScript 语言基础
├── 函数、作用域、数据与模块
│   ├── 浏览器基础：HTML / CSS / DOM / 事件
│   │   └── React：组件 / Props / 状态 / 数据流
│   │       └── Next.js：路由 / 布局 / 客户端与服务端分工
│   ├── 异步与网络：Promise / async / await / HTTP / fetch
│   │   ├── React 接口页面
│   │   └── Node.js 服务端与接口
│   └── TypeScript：类型检查 / 数据契约（可与 React 并行）
├── 工程工具：npm / Git / Vite / 调试 / 测试
└── 按需深入：原型 / 性能 / Worker / WebSocket / 发布部署
~~~

建议先走网页主线：运行机制 → JS 基础 → 函数与模块 → 浏览器基础 → 工程入门 → React。异步与网络在做接口功能前补齐；TypeScript 可以在 JS 基础之后，与 React 并行。Next.js 放在 React 基础稳定之后。

## 2. 分清这些名字分别是什么

| 名称 | 类别 | 解决什么问题 |
|---|---|---|
| JavaScript | 编程语言 | 表达数据、判断、计算、函数与业务逻辑 |
| HTML / CSS | 页面结构 / 样式 | 页面有哪些元素、如何排版和显示 |
| 浏览器 | 运行环境 | 执行 JS，并提供 DOM、事件、网络等接口 |
| Node.js | JS 运行环境 | 在浏览器之外运行 JS，提供文件、进程、服务端等能力 |
| npm | 包管理与脚本工具 | 安装依赖，执行项目声明的命令 |
| TypeScript | 带类型系统的语言与工具 | 在开发阶段检查类型，生成或由工具转换为 JS |
| JSX | 语法扩展 | 在代码中表达界面结构，需转换后执行 |
| React / ReactDOM | 界面库 / 网页渲染工具 | 组织组件和状态，把界面更新到 DOM |
| Vite | 开发与构建工具 | 开发时提供转换服务，构建时生成网页成品 |
| Next.js | React 应用框架 | 提供路由、布局、服务端能力与构建部署方式 |

浏览器的 document 不是 JS 语言本身提供的；Node.js 的文件接口也不是浏览器自带的。参考：[MDN：JavaScript 简介](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Introduction)、[Node.js 简介](https://nodejs.org/en/learn/getting-started/introduction-to-nodejs)。

## 3. 节点 A：运行机制与文件结构〔现在就学〕

- [ ] 分清源码、依赖、构建配置、构建成品各放在哪里。
- [ ] 说明 HTML 如何通过 script 加载 JS，浏览器如何执行入口代码。
- [ ] 分清 .js、.jsx、.ts、.tsx；知道浏览器执行的是转换后的 JS。
- [ ] 说明 React 项目的入口如何找到 DOM 容器，再交给 ReactDOM。
- [ ] 分清 dev、build、preview：开发运行、生成成品、预览已有成品。
- [ ] 知道打包会合并代码、压缩名字；框架代码仍可能包含在成品里。

过关任务：从现有计数器的 index.html 开始，指出加载入口、寻找 root、提交组件的代码位置，再说明 dist 中 HTML、脚本和样式如何协作。这里的 src/dist 结构是现有 Vite 示例的结构，不是所有 JS 项目的固定结构。

参考：[Vite 入门](https://vite.dev/guide/)。

## 4. 节点 B：JavaScript 基础〔主线〕

- [ ] 变量声明、赋值、const 与 let；理解 const 不代表对象内容不可修改。
- [ ] 常见类型：number、string、boolean、null、undefined、对象与数组；认识 bigint、symbol。
- [ ] 显式类型转换、严格相等、真假值；识别隐式转换造成的意外结果。
- [ ] 算术、比较、逻辑运算；说明运算输入类型和计算顺序。
- [ ] if / else、switch、循环、break / continue，以及异常处理。
- [ ] 字符串、模板字符串、数组遍历，常用 map / filter / find。

过关任务：写一个纯 JS 计算器，检查输入类型、数值范围和除零情况；把长计算拆成有意义的步骤。

工程规则与语言能力分开：你可以禁止自己编写 ++ 或三元运算符，但仍应能识别它们、读懂第三方代码。学习清单不替代项目约定；需要采用个人写法时，显式引用 [临时代码草案](../../01_AI项目协作参考/01_编程规范/语言规范/JavaScript/代码标准_临时草案.md)。

参考：[MDN：JavaScript 指南](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)。

## 5. 节点 C：函数、数据与调用关系〔进入 React 前重点学〕

- [ ] 函数定义、参数、返回值；分清声明函数和执行函数。
- [ ] 分清传递函数引用与立即调用；能追踪回调实际指向哪个函数。
- [ ] 普通函数、箭头函数、局部变量、作用域和闭包。
- [ ] 对象与数组的引用、共享修改、浅拷贝；理解解构和展开操作。
- [ ] 分清纯计算、状态修改、网络访问等副作用。
- [ ] import / export、默认导出与具名导出；区分模块依赖图与函数调用图。
- [ ] 分别画出文件依赖、直接调用、回调调度和数据传递关系。

过关任务：把计算器拆成入口、业务函数和工具函数；按你的要求，让业务父函数编排子函数，同层级业务函数不互相调用；工具可复用，但不反向触发业务流程。框架回调单独标注。

注意：导入模块可能执行模块顶层代码，但导入某个函数不等于调用了那个函数；回调函数写在另一个函数内部，也不代表外层函数直接调用了它。

参考：[MDN：语言概览](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Language_overview)、[MDN：模块](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)。

## 6. 节点 D：浏览器与页面〔网页方向主线〕

- [ ] HTML 元素、表单、输入框、按钮与基本页面结构。
- [ ] CSS 选择器、盒模型、Flex 布局和基本响应式显示。
- [ ] DOM 节点：查找元素、读取属性、修改文字、创建和移除元素。
- [ ] 事件监听、事件对象、冒泡、默认行为和事件委托。
- [ ] 浏览器调试：Console、断点、Elements、Network，区分语法与运行错误。
- [ ] 浏览器存储与 JSON；知道 localStorage 的内容不能当作可信输入。

过关任务：先不用 React，用 HTML + CSS + JS 写一个能增加、减少、重置的计数器，并指出每个事件监听器在哪里注册、由谁调用。

学习入口：[MDN：Web 开发学习区](https://developer.mozilla.org/en-US/docs/Learn_web_development)。

## 7. 节点 E：异步与网络〔接接口前必学〕

- [ ] 分清同步返回、回调执行、异步完成；认识调用栈、任务与微任务。
- [ ] Promise、async / await、try / catch；知道 await 不会阻塞整个浏览器。
- [ ] 分清串行与并发，认识 Promise.all 和部分失败的处理方式。
- [ ] HTTP 请求、响应、状态码、JSON、fetch 与响应状态检查。
- [ ] 加载中、成功、失败、空数据等界面状态。
- [ ] 请求取消、重复提交、结果乱序；初步理解同源规则和 CORS。

过关任务：获取并展示一组数据，能解释请求什么时候发出、结果由谁接收、失败如何处理，以及旧请求为什么可能覆盖新结果。

参考：[MDN：使用 Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises)。

## 8. 节点 F：工程工具〔随项目逐步补齐〕

- [ ] Node.js 与 npm 的基本使用；区分执行环境、包管理器和项目依赖。
- [ ] package.json、scripts、依赖、开发依赖和锁文件；了解 npm install 与 npm ci。
- [ ] Git 的修改、commit、push；知道 commit 不等于已保存到远程仓库。
- [ ] Vite 开发运行、构建和预览；理解生成文件应通过构建配置调整。
- [ ] 类型检查、格式化、静态检查、Source Map，各自解决什么问题。
- [ ] 为重要计算和业务行为做必要测试；能读取构建错误与运行日志。

过关任务：独立启动现有项目，修改功能，检查结果，再构建并通过本地服务器运行成品。

不需要先掌握多个构建器或包管理器；先熟悉现有项目使用的工具。参考：[Vite 入门](https://vite.dev/guide/)、[Node.js 简介](https://nodejs.org/en/learn/getting-started/introduction-to-nodejs)。

## 9. 节点 G：React〔网页主线〕

前置：JS 基础、函数与对象数组、模块、基本 HTML/CSS 和事件概念。异步知识在做数据请求功能前补齐。

- [ ] 组件、JSX、嵌套和组合；知道组件通常由 React 调用。
- [ ] Props、事件处理、回调传递；分清传入函数与执行函数。
- [ ] useState、状态快照、函数式更新；不直接修改已有状态对象。
- [ ] 条件显示、列表和 key；表单的受控输入。
- [ ] 状态提升、单向数据流、兄弟组件如何通过共同父组件协作。
- [ ] 自定义 Hook；知道复用 Hook 逻辑不等于共享同一份状态。
- [ ] Effect 用于与外部系统同步，理解清理；普通计算和点击处理不必都放进 Effect。

过关任务：独立写一个待办列表，支持新增、修改、筛选和删除，并能分别画出组件树、状态归属和事件调用链。

参考：[React 快速入门](https://react.dev/learn)、[用 React 思考](https://react.dev/learn/thinking-in-react)。

## 10. 节点 H：TypeScript〔可与 React 并行〕

- [ ] 基本类型、函数参数与返回值、对象结构和类型推断。
- [ ] type / interface、联合类型、可选属性和类型收窄。
- [ ] unknown、any、类型断言；避免用断言绕过实际问题。
- [ ] 为 Props、状态、动作和接口数据写明确契约。
- [ ] 理解静态检查与运行时输入检查不同：类型标注不会自动验证外部 JSON。

过关任务：给计数器的状态和动作加类型，让错误的动作在开发阶段被发现；再对真实外部输入做运行时验证。

TypeScript 不替代 JS 基础；先学常用类型，高级类型运算以后按需学。参考：[TypeScript：面向 JavaScript 开发者](https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes.html)。

## 11. 节点 I：Next.js〔React 后的可选路线〕

前置：React、模块、异步与 HTTP、开发与构建流程。不要求先学完所有 Node.js 服务端框架。

- [ ] App Router 的页面、布局和文件路由；理解 app/page.tsx 的约定。
- [ ] 服务端组件与客户端组件：哪些逻辑在哪里执行，哪些代码发送给浏览器。
- [ ] 页面预渲染与客户端接入交互；客户端组件也可能参与服务端预渲染。
- [ ] 数据获取、Route Handler 和 Server Action 的基本职责与边界。
- [ ] 静态导出与需要服务端运行的部署方式；服务端密钥不应发送给客户端。

过关任务：做一个列表页和详情页，明确页面在哪个文件、数据在哪里获取、点击由谁处理，以及成品需要什么运行环境。

参考：[Next.js 入门](https://nextjs.org/docs/app/getting-started)、[服务端与客户端组件](https://nextjs.org/docs/app/getting-started/server-and-client-components)、[静态导出](https://nextjs.org/docs/app/guides/static-exports)。

## 12. 节点 J：其他分支〔按目标选择〕

| 方向 | 先补什么 | 典型任务与可选工具 |
|---|---|---|
| Node.js 脚本与 CLI | JS、模块、异步、文件与路径、npm | 文件整理、自动化、命令行工具；按项目选工具库 |
| Node.js 服务端 | 模块、异步、HTTP、输入验证 | API、数据库、权限；Express / Fastify / NestJS 按需选一种 |
| 实时通信 | 异步、网络、前后端分工 | 原生 WebSocket / ws / Socket.io；聊天和协作 |
| 爬虫与自动化 | HTTP、DOM、异步、输入输出 | fetch / axios、cheerio；确需浏览器时再用 Playwright / Puppeteer / Crawlee |
| 深入语言机制 | 作用域、对象、函数 | 原型、this、迭代器、生成器、Worker 与性能测量 |
| 更复杂的前端 | React 与数据流基础 | 路由、共享状态、请求缓存；Zustand / Redux 等按问题引入；Vue / Svelte 属于可选的其他框架方向 |
| 桌面应用 | HTML/CSS/JS、界面与系统接口边界 | Electron；Tauri 还涉及其原生工具链，框架按项目选用 |
| 小程序与移动端 | 页面、组件与平台接口 | 原生小程序 / Taro / uni-app；React Native / Expo 是另一条移动端路线 |
| 图形、游戏与数据可视化 | DOM、Canvas / WebGL、数据组织 | Three.js / Phaser / Cocos；数据图表可选 ECharts / D3 |
| 全栈与边缘部署 | HTTP、运行时差异、部署约束 | React 后可选 Next.js；Nuxt 属于 Vue 生态；旧图的 Cloudflare Workers / Vercel Edge 方向先核对运行时接口限制 |
| 浏览器与服务端 AI 应用 | 页面、网络、模型输入输出 | 浏览器推理可了解 transformers.js / WebGPU；旧图还列过 Vercel AI SDK / LangChain.js 等编排工具，Python 训练服务也可与网页协作 |
| 测试、发布与维护 | 工程、HTTP、安全与部署基础 | Vitest / Jest、Playwright 等按需使用；自动检查、日志、部署与回滚 |

旧生态图中用于表示“主流”“上升期”的热度标签不作为学习前置；语言分工按工程需求决定，不把 JS、Python 或其他语言绝对划成互不重叠的领域。
这些分支不必全部解锁，也不必学习多套功能类似的框架。React 不是写 Node.js 脚本或接口的前置条件。

旧生态图中其余名词按角色放回这里，便于以后查找：

| 名词或相邻方向 | 怎样放进现有路线 |
|---|---|
| Bun / Deno | 与 Node.js 比较运行时和工具能力时再看，先核对具体版本的兼容范围 |
| Vite / webpack / esbuild | 工程工具方向的不同项目或组成部分，不要求先学完 React 才能接触构建工具 |
| pnpm / Yarn | 包管理的可选方案；跟随项目锁文件，不作为当前额外主线 |
| Pinia | 旧 Vue 分支中的状态管理例子，不搬进 React 项目当作前置 |
| Python | 模型训练、科学计算、数据处理等可与 JS 网页和服务协作，按任务选语言 |
| C / Rust | 固件、系统接口或性能敏感模块；是否需要原生层取决于项目 |
| Go / Java | 可选服务端路线；与 Node 的比较要结合计算、I/O、并发模型和部署约束 |

## 13. 你的近期学习顺序

1. 把当前示例的加载、构建和运行机制讲清楚。
2. 学变量、类型、判断、函数、对象、数组与模块。
3. 特别练习函数引用、回调、返回值、闭包与数据流。
4. 用原生 JS 做一遍计数器，直接操作 DOM。
5. 再用 React 实现同样功能，对比谁保存状态、谁调用函数、谁更新页面。
6. 学异步与接口请求，给页面增加真实数据。
7. 并行补 TypeScript 和工程工具；最后按需要学习 Next.js 或服务端开发。

当前示例入口：[react-counter-demo](示例/react-counter-demo)。相关教程：[React 教程：调用关系与数据流](React教程_调用关系与数据流.md)。

每完成一个小项目，都回答五个问题：入口在哪里？谁直接调用谁？哪些函数由框架或事件系统调用？数据从哪里来、到哪里去？浏览器最终执行的是哪些文件？

*2026-10-25：以本清单的逐项任务为主，吸收旧前端笔记中的桌面、移动端、图形、可视化、爬虫与 AI 等可选方向；保留全部原有未完成勾选项。*
