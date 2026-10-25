/**
 * 主题：根组件组织页面。输入：无；输出：HomePage 的 JSX 描述。
 * 结构：App 负责根布局；HomePage 负责页面内容。
 * 框架组件树：React → App → <HomePage />，组件函数由 React 调用。
 * 数据流：组件描述 → React 渲染与提交 → DOM。
 * 运行：npm run dev，详见 README。
 */
import HomePage from './pages/HomePage'

/** 根组件只编排页面，由 React 调用。 */
export default function App() {
  return <HomePage />
}
