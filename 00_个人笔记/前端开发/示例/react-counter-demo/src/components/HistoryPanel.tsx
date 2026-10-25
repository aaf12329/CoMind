/**
 * 主题：把状态中的数组变成 JSX 列表。
 * 输入：history:HistoryEntry[]，长度 0~5；输出：操作记录列表。
 * 结构：HistoryPanel 展示列表；renderEntry 展示一条记录。
 * 框架调用树：React → HistoryPanel。
 * 列表调用：HistoryPanel → Array.map（第三方）→ renderEntry（由 map 调用）。
 * 数据流：记录数组 → 每条记录的 JSX → <ul>；运行：npm run dev，详见 README。
 */
import type { HistoryEntry } from '../types/counter'

/** 返回一条记录的 JSX，仅由 HistoryPanel 中的 Array.map 调用。 */
function renderEntry(entry: HistoryEntry) {
  return (
    <li key={entry.id} className="history-entry">
      <span className="history-order">{String(entry.id).padStart(2, '0')}</span>
      <div><strong>{entry.label}</strong><span>{entry.before} → {entry.after}</span></div>
      <time>{entry.time}</time>
    </li>
  )
}

/** 展示最近五次操作或空状态，由 React 调用。 */
export default function HistoryPanel({ history }: { history: HistoryEntry[] }) {
  return (
    <section className="history-card">
      <div className="section-heading"><span className="section-index">03</span><h2>最近的操作</h2><span className="small-label">保留 5 条</span></div>
      {history.length === 0 ? <p className="empty-history">还没有操作。点一下“增加”，看第一条记录出现。</p> : <ul className="history-list">{history.map(renderEntry)}</ul>}
    </section>
  )
}
