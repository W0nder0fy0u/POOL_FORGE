// Reports page: every run in this room, with CSV / JSON export.
import { useEffect, useState } from "react";
import { api } from "../api.js";
import { StatusPill, fmtNum, fmtTime } from "../components/ui.jsx";
import { useRoom } from "../room.jsx";

export default function Reports() {
  const { code, state } = useRoom();
  const [jobs, setJobs] = useState([]);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const jobStatus = state.job?.status;

  useEffect(() => {
    api(`/api/rooms/${code}/jobs`).then(setJobs).catch((e) => setError(e.message));
  }, [code, jobStatus]);

  const open = (id) => api(`/api/rooms/${code}/jobs/${id}`).then(setDetail).catch((e) => setError(e.message));

  return (
    <>
      <section className="card">
        <h2>Runs</h2>
        {error && <div className="banner error">{error}</div>}
        {jobs.length === 0 ? (
          <div className="empty">No runs yet.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Model</th>
                  <th>Transport</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Export</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.job_id}>
                    <td>
                      <button onClick={() => open(j.job_id)}>{j.job_id}</button>
                    </td>
                    <td>{j.model}</td>
                    <td>{j.transport}</td>
                    <td title={j.detail}>
                      <StatusPill status={j.status} />
                    </td>
                    <td>{fmtTime(j.created_at)}</td>
                    <td className="row">
                      <a className="button" href={`/api/rooms/${code}/jobs/${j.job_id}/export?format=csv`}>CSV</a>
                      <a className="button" href={`/api/rooms/${code}/jobs/${j.job_id}/export?format=json`}>JSON</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {detail && <JobSummary job={detail} />}
    </>
  );
}

function JobSummary({ job }) {
  const m = job.metrics;
  const avg = (k) => (m.length ? m.reduce((s, x) => s + x[k], 0) / m.length : null);
  const nodes = [...new Set(job.node_metrics.map((x) => x.node_id))];
  const peakRam = (id) => Math.max(...job.node_metrics.filter((x) => x.node_id === id).map((x) => x.proc_ram_mb));
  return (
    <section className="card stack">
      <h2>{job.job_id}</h2>
      <div className="table-wrap">
        <table>
          <tbody>
            <tr><th>Status</th><td>{job.status} {job.detail && <span className="muted">({job.detail})</span>}</td></tr>
            <tr><th>Model</th><td>{job.model}</td></tr>
            <tr><th>Config</th><td className="mono small" style={{ whiteSpace: "normal" }}>{JSON.stringify(job.config)}</td></tr>
            <tr><th>Steps recorded</th><td>{m.length}</td></tr>
            <tr><th>First / last loss</th><td>{m.length ? `${fmtNum(m[0].loss, 4)} / ${fmtNum(m[m.length - 1].loss, 4)}` : "-"}</td></tr>
            <tr><th>Avg step time</th><td>{fmtNum(avg("step_time_ms"), 1)} ms</td></tr>
            <tr><th>Avg samples / s</th><td>{fmtNum(avg("samples_per_s"), 1)}</td></tr>
            {nodes.map((id) => (
              <tr key={id}><th>Peak agent RAM: {id}</th><td>{fmtNum(peakRam(id), 0)} MB</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
