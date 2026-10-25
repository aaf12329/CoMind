/**
 * 主题：父组件保存状态并通过 Props 传给子组件。
 * 输入：无外部输入，useCounter 构造初始状态；输出：完整页面 JSX。
 * 直接调用树：HomePage → useCounter。
 * 框架组件树：React → HomePage → Header / Counter / StatePanel / HistoryPanel。
 * 子组件通过 JSX 声明，React 调用组件函数；这里不直接调用子组件函数。
 * 数据流：state → 各子组件 Props；Counter 的操作 → Hook → 新 state。
 * 运行：npm run dev，详见 README。
 */
import Header from '../components/Header'
import Counter from '../components/Counter'
import StatePanel from '../components/StatePanel'
import HistoryPanel from '../components/HistoryPanel'
import useCounter from '../hooks/useCounter'

/** 编排页面与 Props，由 React 调用；本页是 useCounter 的唯一调用者。 */
export default function HomePage() {
  const { state, dispatchCounterAction } = useCounter()

  return (
    <div className="app-shell">
      <Header />
      <main>
        <section className="intro">
          <span className="eyebrow"><span className="status-dot" /> 第一个 React 示例</span>
          <h1>点一下，<br />看状态变成界面。</h1>
          <p>一个计数器，串起组件、事件、状态和 Props。左边操作，右边观察。</p>
        </section>

        <div className="workspace">
          <Counter count={state.count} step={state.step} onAction={dispatchCounterAction} />
          <StatePanel state={state} />
        </div>

        <div className="details-grid">
          <HistoryPanel history={state.history} />
          <section className="lesson-card">
            <div className="section-heading"><span className="section-index">04</span><h2>从哪里开始读代码？</h2></div>
            <ol className="reading-list">
              <li><code>index.html</code><span>浏览器的网页入口</span></li>
              <li><code>src/main.tsx</code><span>把 App 挂到 root 容器</span></li>
              <li><code>src/pages/HomePage.tsx</code><span>取状态，组织子组件</span></li>
              <li><code>src/hooks/useCounter.ts</code><span>一次操作如何变成新状态</span></li>
            </ol>
            <p className="muted-note">修改文字或样式并保存，开发服务器会更新页面。完整图解见 README.md。</p>
          </section>
        </div>
      </main>
      <footer><span>React × TypeScript × Vite</span><span>组件是积木，状态是记忆。</span></footer>
    </div>
  )
}
