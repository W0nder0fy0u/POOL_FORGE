// Training page: pick a model, preview the split, start/pause/stop, watch it learn.
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";
import SplitPlan from "../components/SplitPlan.jsx";
import { ChartTip, Stat, StatusPill, axisProps, fmtNum } from "../components/ui.jsx";
import { nodeColor, useRoom } from "../room.jsx";

const DEFAULTS = {
  model: "mnist_cnn",
  dataset: "",
  batch_size: 64,
  microbatches: 1,
  epochs: 1,
  steps_per_epoch: 100,
  lr: 0.001,
  optimizer: "adam",
  checkpoint_every: 20,
  width: 2048,
  depth: 8,
};

const ACTIVE = ["running", "paused", "waiting"];

export default function Training() {
  const { state, isAdmin, admin, log } = useRoom();
  const [models, setModels] = useState([]);
  const [form, setForm] = useState(DEFAULTS);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const job = state.job;
  const active = job && ACTIVE.includes(job.status);

  useEffect(() => {
    api("/api/models").then(setModels).catch((e) => setError(e.message));
  }, []);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value });

  const config = () => {
    const c = {
      model: form.model,
      batch_size: form.batch_size,
      microbatches: form.microbatches,
      epochs: form.epochs,
      steps_per_epoch: form.steps_per_epoch,
      lr: form.lr,
      optimizer: form.optimizer,
      checkpoint_every: form.checkpoint_every,
    };
    if (form.dataset) c.dataset = form.dataset;
    if (form.model === "large_mlp") c.model_options = { width: form.width, depth: form.depth };
    return c;
  };

  const run = async (fn) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const doPreview = () => run(async () => setPreview(await admin("/plan", config())));
  const doStart = () =>
    run(async () => {
      await admin("/jobs", config());
      setPreview(null);
      log("Start requested");
    });
  const doAction = (action) => run(() => admin(`/jobs/${job.job_id}/${action}`));

  return (
    <>
      <section className="card">
        <div className="card-head">
          <h2>Configure training</h2>
          {!isAdmin && <span className="muted small">admin token needed to start</span>}
        </div>
        <div className="form-grid">
          <label className="field">
            Model
            <select value={form.model} onChange={set("model")}>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Dataset
            <select value={form.dataset} onChange={set("dataset")}>
              <option value="">model default</option>
              <option value="mnist">MNIST</option>
              <option value="synthetic">synthetic</option>
            </select>
          </label>
          {form.model === "large_mlp" && (
            <>
              <label className="field">
                Width (neurons)
                <input type="number" min="8" max="16384" value={form.width} onChange={set("width")} />
              </label>
              <label className="field">
                Depth (hidden layers)
                <input type="number" min="1" max="64" value={form.depth} onChange={set("depth")} />
              </label>
            </>
          )}
          <label className="field">
            Batch size
            <input type="number" min="1" value={form.batch_size} onChange={set("batch_size")} />
          </label>
          <label className="field">
            Micro-batches
            <input type="number" min="1" max="64" value={form.microbatches} onChange={set("microbatches")} />
          </label>
          <label className="field">
            Epochs
            <input type="number" min="1" value={form.epochs} onChange={set("epochs")} />
          </label>
          <label className="field">
            Steps per epoch
            <input type="number" min="1" value={form.steps_per_epoch} onChange={set("steps_per_epoch")} />
          </label>
          <label className="field">
            Learning rate
            <input type="number" step="0.0001" min="0" value={form.lr} onChange={set("lr")} />
          </label>
          <label className="field">
            Optimizer
            <select value={form.optimizer} onChange={set("optimizer")}>
              <option value="adam">Adam</option>
              <option value="sgd">SGD (momentum)</option>
            </select>
          </label>
          <label className="field">
            Checkpoint every N steps
            <input type="number" min="1" value={form.checkpoint_every} onChange={set("checkpoint_every")} />
          </label>
        </div>
        <p className="muted small" style={{ marginTop: "0.6rem" }}>
          {models.find((m) => m.name === form.model)?.description}
        </p>
        <div className="row">
          <button onClick={doPreview} disabled={!isAdmin || busy}>
            Preview split
          </button>
          <button className="primary" onClick={doStart} disabled={!isAdmin || busy || active}>
            Start
          </button>
          {job && active && (
            <>
              {job.status === "paused" ? (
                <button onClick={() => doAction("resume")} disabled={!isAdmin || busy}>
                  Resume
                </button>
              ) : (
                <button onClick={() => doAction("pause")} disabled={!isAdmin || busy || job.status === "waiting"}>
                  Pause
                </button>
              )}
              <button className="danger" onClick={() => doAction("stop")} disabled={!isAdmin || busy}>
                Stop
              </button>
            </>
          )}
        </div>
        {error && (
          <div className="banner error" style={{ marginTop: "0.8rem" }}>
            {error}
          </div>
        )}
        {preview && (
          <div style={{ marginTop: "1rem" }}>
            <h3>
              Proposed split ({preview.transport}, {preview.dataset})
            </h3>
            <SplitPlan plan={preview.plan} layerTypes={preview.layer_types} />
          </div>
        )}
      </section>

      {job ? <JobView job={job} /> : <div className="card empty">No training job yet. Configure one above and press Start.</div>}
    </>
  );
}

function JobView({ job }) {
  const { state } = useRoom();
  const metrics = state.metrics;
  const last = metrics[metrics.length - 1];
  const recent = metrics.slice(-10);
  const avgMs = recent.length ? recent.reduce((s, m) => s + m.step_time_ms, 0) / recent.length : null;
  const nodeIds = job.plan.map((p) => p.node_id);

  // per-node agent RAM at every step: [{step, nodeA: MB, nodeB: MB}]
  const ramData = useMemo(() => {
    const byStep = new Map();
    state.nodeMetrics.forEach((r) => {
      const row = byStep.get(r.step) || { step: r.step };
      row[r.node_id] = r.proc_ram_mb;
      byStep.set(r.step, row);
    });
    return [...byStep.values()].sort((a, b) => a.step - b.step);
  }, [state.nodeMetrics]);

  return (
    <>
      <section className="grid cols-4">
        <Stat label="Status" value={<StatusPill status={job.status} />} sub={`${job.job_id} - ${job.model} on ${job.dataset}`} />
        <Stat label="Epoch" value={last ? `${last.epoch} / ${job.config.epochs}` : "-"} />
        <Stat label="Step" value={`${job.last_step} / ${job.total_steps}`} />
        <Stat label="Loss" value={last ? fmtNum(last.loss, 4) : "-"} />
        <Stat label="Step time" value={avgMs ? `${fmtNum(avgMs, 0)} ms` : "-"} sub="avg of last 10" />
        <Stat label="Steps / sec" value={avgMs ? fmtNum(1000 / avgMs, 2) : "-"} />
        <Stat label="Samples / sec" value={avgMs ? fmtNum((job.config.batch_size * 1000) / avgMs, 0) : "-"} />
        <Stat label="Transport" value={job.transport} sub={`${job.plan.length} stage(s), ${job.config.microbatches} micro-batch(es)`} />
      </section>

      {job.missing_nodes && job.missing_nodes.length > 0 && (
        <div className="banner warning">Waiting for node(s) {job.missing_nodes.join(", ")} to come back. Training resumes from the last checkpoint.</div>
      )}

      <section className="card">
        <div className="card-head">
          <h2>Split plan</h2>
          <span className="muted small">{job.num_layers} layers over {job.plan.length} node(s)</span>
        </div>
        <SplitPlan plan={job.plan} layerTypes={job.layer_types} />
      </section>

      <section className="grid cols-2">
        <div className="card">
          <h2>Training loss</h2>
          <ChartOrEmpty data={metrics}>
            <LineChart data={metrics} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="step" {...axisProps} />
              <YAxis {...axisProps} width={48} domain={["auto", "auto"]} />
              <Tooltip content={<ChartTip labelFormatter={(s) => `step ${s}`} valueFormatter={(v) => fmtNum(v, 4)} />} />
              <Line type="monotone" dataKey="loss" name="loss" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ChartOrEmpty>
        </div>
        <div className="card">
          <h2>Step time (ms)</h2>
          <ChartOrEmpty data={metrics}>
            <LineChart data={metrics} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="step" {...axisProps} />
              <YAxis {...axisProps} width={48} />
              <Tooltip content={<ChartTip labelFormatter={(s) => `step ${s}`} valueFormatter={(v) => `${fmtNum(v, 0)} ms`} />} />
              <Line type="monotone" dataKey="step_time_ms" name="step time" stroke="var(--series-2)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ChartOrEmpty>
        </div>
      </section>

      <section className="card">
        <h2>Per-node RAM used by the agent (MB)</h2>
        <p className="muted small">Each laptop only holds its own layers. Includes the PyTorch runtime itself (several hundred MB per process) and, on the first stage, the dataset.</p>
        <ChartOrEmpty data={ramData}>
          <LineChart data={ramData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="step" {...axisProps} />
            <YAxis {...axisProps} width={56} />
            <Tooltip content={<ChartTip labelFormatter={(s) => `step ${s}`} valueFormatter={(v) => `${fmtNum(v, 0)} MB`} />} />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
            {nodeIds.map((id) => (
              <Line
                key={id}
                type="monotone"
                dataKey={id}
                name={id}
                stroke={nodeColor(state, id)}
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ChartOrEmpty>
      </section>

      <section className="card">
        <h2>Event log</h2>
        <div className="log">
          {state.log.length === 0 && <div className="muted">No events yet.</div>}
          {state.log.map((l) => (
            <div key={l.t + l.text}>
              <span className="muted">{new Date(l.t).toLocaleTimeString()}</span> {l.text}
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

export function ChartOrEmpty({ data, children, height = 240 }) {
  if (!data || data.length === 0) return <div className="empty">Waiting for data...</div>;
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>{children}</ResponsiveContainer>
    </div>
  );
}
