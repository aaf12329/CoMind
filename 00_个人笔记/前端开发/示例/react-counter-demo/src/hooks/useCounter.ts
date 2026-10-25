/**
 * 主题：用 useState 保存状态，自定义 Hook 提供状态与操作入口。
 * 初始状态：{ count: 0, step: 1, totalActions: 0, history: [] }。
 * 输入：CounterAction；输出：CounterState 与 dispatchCounterAction。
 * 结构导航：① 初始数据 ② useCounter ③ 动作处理 ④ 纯状态更新。
 * 直接调用树：HomePage → useCounter → useState（第三方）。
 * 事件调用树：Counter.handleControl → dispatchCounterAction → setState（第三方）。
 * 框架调度：setState 接收 updateState；React 调用 updateState(previous)。
 * updateState 无外部副作用；时间由点击处理器提供，不在更新器中取时间。
 * 数据流：动作 + 前一状态 → updateState → 新状态 → 页面重新渲染。
 * 运行：npm run dev，详见 README；本学习单元没有其他示例依赖。
 */
import { useState } from 'react'
import type { CounterAction, CounterState, HistoryEntry } from '../types/counter'

// ① 准备：一个状态对象，字段意义与 types/counter.ts 保持一致。
const INITIAL_STATE: CounterState = {
  count: 0,
  step: 1,
  totalActions: 0,
  history: [],
}
const HISTORY_LIMIT = 5

/** 保存计数器状态，被 HomePage 调用；返回状态快照与动作入口。 */
export default function useCounter() {
  const [state, setState] = useState<CounterState>(INITIAL_STATE)

  // ② 执行：只有 Counter.handleControl 调用这个入口。
  /** 接收一个动作，把纯更新函数交给 React 调度。 */
  function dispatchCounterAction(action: CounterAction) {
    // ③ 核心：previous 由 React 提供，避免依赖旧的闭包状态。
    /** 根据前一状态计算新状态，由 React 调用；不修改 previous。 */
    function updateState(previous: CounterState): CounterState {
      let count = previous.count
      let step = previous.step
      let label: string

      switch (action.type) {
        case 'increase':
          count += step
          label = `增加 ${step}`
          break
        case 'decrease':
          count -= step
          label = `减少 ${step}`
          break
        case 'set-step':
          step = action.value
          label = `步长设为 ${step}`
          break
        case 'reset':
          count = 0
          step = 1
          label = '重置计数与步长'
          break
      }

      const entry: HistoryEntry = {
        id: previous.totalActions + 1,
        label,
        before: previous.count,
        after: count,
        time: action.time,
      }
      return {
        count,
        step,
        totalActions: entry.id,
        history: [entry, ...previous.history].slice(0, HISTORY_LIMIT),
      }
    }

    setState(updateState)
  }

  // ④ 输出：计数器组件修改状态，其他组件只读取同一份快照。
  return { state, dispatchCounterAction }
}
