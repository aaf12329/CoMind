/**
 * 主题：HTML 容器如何接上 React。
 * 输入：index.html 中的 #root 元素；输出：挂载 App 的 React 根。
 * 结构：依赖 → 找容器 → 创建 React 根 → 提交组件描述。
 * 调用：模块入口 → document.getElementById → createRoot → root.render。
 * 框架调度：React 随后调用 App；<App /> 不是 main 直接调用 App()。
 * 数据流：HTMLElement → React 根 → <App /> → 页面 DOM。
 * 运行：npm run dev；不能直接双击 TSX 文件，详见 README。
 */
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles/global.css'

const container = document.getElementById('root')
if (!container) throw new Error('缺少 #root 容器，请检查 index.html。')
createRoot(container).render(<App />)
