/**
 * 主题：读取 Props，处理事件，把动作传回父组件提供的入口。
 * 输入：count:number、step:number、onAction:动作回调；输出：计数器 JSX。
 * 结构：Counter 负责显示；handleControl 统一处理按钮与下拉框。
 * 框架调用树：React → Counter；React 事件系统 → handleControl。
 * 直接事件树：handleControl → onAction（实际为 useCounter 的 dispatchCounterAction）。
 * 同一个处理器服务全部控件，不建立互相交叉调用的辅助函数。
 * 数据流：用户事件 → CounterAction → Hook；count/step Props → 显示。
 * 运行：npm run dev，详见 README。
 */
import type { ChangeEvent, MouseEvent } from 'react'
import type { CounterAction } from '../types/counter'
import '../styles/counter.css'

interface CounterProps {
  count: number
  step: number
  onAction: (action: CounterAction) => void
}

/** 展示计数与控件，由 React 调用；不自行保存计数状态。 */
export default function Counter({ count, step, onAction }: CounterProps) {
  /** 读取控件的 data-action，构造一个动作；由 React 事件系统调用。 */
  function handleControl(event: MouseEvent<HTMLButtonElement> | ChangeEvent<HTMLSelectElement>) {
    const control = event.currentTarget
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    let action: CounterAction

    if (control instanceof HTMLSelectElement) {
      action = { type: 'set-step', value: Number(control.value), time }
    } else {
      const type = control.dataset.action
      if (type !== 'increase' && type !== 'decrease' && type !== 'reset') return
      action = { type, time }
    }
    onAction(action)
  }

  return (
    <section className="counter-card" aria-labelledby="counter-heading">
      <div className="counter-card-heading">
        <div className="section-heading"><span className="section-index">01</span><h2 id="counter-heading">动手试一试</h2></div>
        <span className="live-badge"><span className="status-dot" /> 可交互</span>
      </div>

      <div className="count-display">
        <span className="count-caption">CURRENT COUNT / 当前计数</span>
        <output className="count-value" aria-label="当前计数" aria-live="polite">{count}</output>
        <span className="count-note">这个数字来自 state.count</span>
      </div>

      <div className="counter-actions">
        <button className="action-button decrease" data-action="decrease" onClick={handleControl}>
          <span className="button-symbol" aria-hidden="true">−</span> 减少 {step}
        </button>
        <button className="action-button increase" data-action="increase" onClick={handleControl}>
          <span className="button-symbol" aria-hidden="true">＋</span> 增加 {step}
        </button>
      </div>

      <div className="counter-settings">
        <div className="step-control"><label htmlFor="step">每次变化的步长</label><select id="step" value={step} onChange={handleControl}><option value="1">1</option><option value="2">2</option><option value="5">5</option><option value="10">10</option></select></div>
        <button className="reset-button" data-action="reset" onClick={handleControl}>重置</button>
      </div>
      <p className="counter-footnote">试着改步长、连点几次，或减到负数。</p>
    </section>
  )
}
