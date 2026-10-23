// Small reusable UI pieces.
import { useState } from "react";

export function fmtBytes(b) {
  if (b === null || b === undefined) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (Math.abs(b) >= 1024 && i < units.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export const fmtNum = (v, digits = 2) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(digits));
export const fmtTime = (t) => new Date(t * (t < 1e12 ? 1000 : 1)).toLocaleTimeString();

export function CopyBox({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard blocked (http, old browser): select the text instead
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(`cb-${text.length}`));
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="codebox">
      <pre id={`cb-${text.length}`}>{text}</pre>
      <button onClick={copy} aria-label="Copy to clipboard">{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

export function Meter({ label, value, max, text, color }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div>
      <div className="meter-label">
        <span>{label}</span>
        <span>{text}</span>
      </div>
      <div className="meter" role="meter" aria-valuenow={value} aria-valuemin={0} aria-valuemax={max} aria-label={label}>
        <div style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

// Status pill: colour dot + text label (never colour alone).
const STATUS = {
  running: "good", training: "good", completed: "good", live: "good", online: "good",
  paused: "warning", waiting: "warning", reconnecting: "warning", connecting: "info", starting: "info", hosting: "info",
  failed: "critical", error: "critical", stopped: "off", offline: "off", idle: "off", interrupted: "critical", missing: "critical",
};
export function StatusPill({ status, label }) {
  return (
    <span className="pill">
      <span className={`dot ${STATUS[status] || "off"}`} />
      {label || status}
    </span>
  );
}

// Tooltip body shared by all charts.
export function ChartTip({ active, payload, label, labelFormatter, valueFormatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      <div className="k">{labelFormatter ? labelFormatter(label) : label}</div>
      {payload.map((p) => (
        <div key={p.dataKey}>
          <span className="dot" style={{ background: p.stroke || p.fill, marginRight: 6 }} />
          {p.name}: <b>{valueFormatter ? valueFormatter(p.value) : p.value}</b>
        </div>
      ))}
    </div>
  );
}

export const axisProps = {
  stroke: "var(--grid)",
  tick: { fill: "var(--text-muted)", fontSize: 11 },
  tickLine: false,
};
