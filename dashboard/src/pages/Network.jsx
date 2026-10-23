// Network page: how much data each node moves per step, and how fast.
import { useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTip, Stat, axisProps, fmtBytes, fmtNum } from "../components/ui.jsx";
import { nodeColor, useRoom } from "../room.jsx";
import { ChartOrEmpty } from "./Training.jsx";

// Turn [{step, node_id, value}] rows into [{step, nodeA: v, nodeB: v}] for multi-line charts.
function pivot(rows, valueFn) {
  const byStep = new Map();
  rows.forEach((r) => {
    const row = byStep.get(r.step) || { step: r.step };
    row[r.node_id] = valueFn(r);
    byStep.set(r.step, row);
  });
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

function NodeLines({ state, ids, data, unit, digits = 2 }) {
  return (
    <ChartOrEmpty data={data}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="step" {...axisProps} />
        <YAxis {...axisProps} width={56} />
        <Tooltip content={<ChartTip labelFormatter={(s) => `step ${s}`} valueFormatter={(v) => `${fmtNum(v, digits)} ${unit}`} />} />
        <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
        {ids.map((id) => (
          <Line key={id} type="monotone" dataKey={id} name={id} stroke={nodeColor(state, id)} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ChartOrEmpty>
  );
}

export default function Network() {
  const { state } = useRoom();
  const job = state.job;
  const rows = state.nodeMetrics;
  const ids = job ? job.plan.map((p) => p.node_id) : [];

  const sent = useMemo(() => pivot(rows, (r) => r.bytes_sent / (1024 * 1024)), [rows]);
  const latency = useMemo(() => pivot(rows, (r) => r.avg_send_ms), [rows]);
  const speed = useMemo(() => pivot(rows, (r) => (r.step_time_ms > 0 ? r.bytes_sent / (1024 * 1024) / (r.step_time_ms / 1000) : 0)), [rows]);

  const summary = ids.map((id) => {
    const mine = rows.filter((r) => r.node_id === id);
    const sum = (k) => mine.reduce((s, r) => s + r[k], 0);
    const time = sum("step_time_ms") / 1000;
    return {
      id,
      steps: mine.length,
      sent: sum("bytes_sent"),
      received: sum("bytes_received"),
      perStep: mine.length ? sum("bytes_sent") / mine.length : 0,
      avgSend: mine.length ? sum("avg_send_ms") / mine.length : 0,
      mbps: time > 0 ? sum("bytes_sent") / (1024 * 1024) / time : 0,
    };
  });
  const totalPerStep = summary.reduce((s, x) => s + x.perStep, 0);

  if (!job) return <div className="card empty">Start a training job to see network traffic.</div>;
  return (
    <>
      <section className="grid cols-4">
        <Stat label="Transport" value={job.transport} sub={job.transport === "relay" ? "via coordinator (works anywhere)" : "direct TCP (LAN / Tailscale)"} />
        <Stat label="Bytes per step (all nodes)" value={fmtBytes(totalPerStep)} sub="activations + labels + gradients" />
        <Stat label="Avg send time" value={`${fmtNum(summary.reduce((s, x) => s + x.avgSend, 0) / Math.max(1, summary.length), 1)} ms`} sub="per frame" />
        <Stat label="Steps measured" value={summary[0]?.steps ?? 0} />
      </section>

      <section className="card">
        <h2>Per node</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Node</th>
                <th className="num">Sent</th>
                <th className="num">Received</th>
                <th className="num">Sent / step</th>
                <th className="num">Avg send (ms)</th>
                <th className="num">Throughput (MB/s)</th>
              </tr>
            </thead>
            <tbody>
              {summary.map((s) => (
                <tr key={s.id}>
                  <td>
                    <span className="dot" style={{ background: nodeColor(state, s.id), marginRight: 6 }} />
                    {s.id}
                  </td>
                  <td className="num">{fmtBytes(s.sent)}</td>
                  <td className="num">{fmtBytes(s.received)}</td>
                  <td className="num">{fmtBytes(s.perStep)}</td>
                  <td className="num">{fmtNum(s.avgSend, 2)}</td>
                  <td className="num">{fmtNum(s.mbps, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid cols-2">
        <div className="card">
          <h2>MB sent per step</h2>
          <NodeLines state={state} ids={ids} data={sent} unit="MB" />
        </div>
        <div className="card">
          <h2>Average send time per frame (ms)</h2>
          <NodeLines state={state} ids={ids} data={latency} unit="ms" />
        </div>
      </section>
      <section className="card">
        <h2>Send throughput (MB/s)</h2>
        <NodeLines state={state} ids={ids} data={speed} unit="MB/s" />
        <p className="muted small">
          Relay vs TCP and injected-latency comparisons are measured by the benchmark scripts (benchmarks/results/results.md).
        </p>
      </section>
    </>
  );
}
