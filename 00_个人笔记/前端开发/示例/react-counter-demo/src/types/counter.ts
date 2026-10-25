/**
 * 主题：计数器的数据契约。输入/输出：无运行时数据，只声明 TypeScript 类型。
 * 结构：输入动作 CounterAction → 操作记录 HistoryEntry → 页面状态 CounterState。
 * 无自定义函数、无调用树；构建时类型被移除。
 * 运行：npm run typecheck，详见 README。
 */
export type CounterAction =
  | { type: 'increase' | 'decrease' | 'reset'; time: string }
  | { type: 'set-step'; value: number; time: string }

export interface HistoryEntry {
  id: number
  label: string
  before: number
  after: number
  time: string
}

export interface CounterState {
  count: number
  step: number
  totalActions: number
  history: HistoryEntry[]
}
