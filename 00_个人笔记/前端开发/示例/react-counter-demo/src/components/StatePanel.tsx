/**
 * 主题：多个子组件读取同一份状态，观察事件到界面的链路。
 * 输入：state:CounterState；输出：状态字段与原理步骤的 JSX。
 * 调用树：React → StatePanel（框架调用），无自定义子函数。
 * 数据流：HomePage.state → Props → 状态显示；运行：npm run dev，详见 README。
 */
import type { CounterState } from '../types/counter'

/** 读取状态快照并显示原理步骤，由 React 调用，不修改状态。 */
export default function StatePanel({ state }: { state: CounterState }) {
  return (
    <section className="state-card" aria-labelledby="state-heading">
      <div className="section-heading"><span className="section-index">02</span><h2 id="state-heading">界面背后发生了什么</h2></div>
      <div className="state-snapshot">
        <span className="snapshot-label">STATE / 同一份状态</span>
        <div className="state-values">
          <div><code>count</code><strong data-testid="state-count">{state.count}</strong></div>
          <div><code>step</code><strong data-testid="state-step">{state.step}</strong></div>
          <div><code>操作次数</code><strong data-testid="action-total">{state.totalActions}</strong></div>
        </div>
      </div>
      <ol className="flow-list">
        <li><span className="flow-number">1</span><div><strong>收到事件</strong><p>handleControl 把点击或选择转换为动作。</p></div></li>
        <li><span className="flow-number">2</span><div><strong>更新状态</strong><p>setState 把更新交给 React，计算新的 count / step。</p></div></li>
        <li><span className="flow-number">3</span><div><strong>重新描述界面</strong><p>React 调用组件，JSX 使用新的状态值。</p></div></li>
        <li><span className="flow-number">4</span><div><strong>提交到 DOM</strong><p>必要的页面元素变化由浏览器显示出来。</p></div></li>
      </ol>
      <div className="code-caption"><code>&lt;output&gt;{'{count}'}&lt;/output&gt;</code><span>现在显示 {state.count}</span></div>
    </section>
  )
}
