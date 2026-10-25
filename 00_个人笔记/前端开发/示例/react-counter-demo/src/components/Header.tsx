/**
 * 主题：无状态组件。输入：无；输出：标题栏 JSX。
 * 结构：Header 一个函数。调用树：React → Header（框架调用），无自定义子函数。
 * 数据流：固定文字 → JSX → 标题栏 DOM。运行：npm run dev，详见 README。
 */
/** 展示固定标题栏，由 React 调用。 */
export default function Header() {
  return (
    <header className="topbar">
      <div className="brand"><span className="brand-mark" aria-hidden="true">R</span><span>React 入门实验室</span></div>
      <span className="lesson-badge">LESSON 01 / 状态与界面</span>
    </header>
  )
}
