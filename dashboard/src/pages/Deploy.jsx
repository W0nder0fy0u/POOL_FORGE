// Deploy page: run an allow-listed app on a pool node and get its URL.
import { useEffect, useState } from "react";
import { api } from "../api.js";
import { StatusPill, fmtTime } from "../components/ui.jsx";
import { useRoom } from "../room.jsx";

export default function Deploy() {
  const { state, isAdmin, admin } = useRoom();
  const [apps, setApps] = useState([]);
  const [appId, setAppId] = useState("static_site");
  const [nodeId, setNodeId] = useState("");
  const [error, setError] = useState("");
  const online = Object.values(state.nodes).filter((n) => n.online);
  const deployments = Object.values(state.deployments).sort((a, b) => b.created_at - a.created_at);

  useEffect(() => {
    api("/api/apps").then(setApps).catch((e) => setError(e.message));
  }, []);

  const act = async (path, body) => {
    setError("");
    try {
      await admin(path, body);
    } catch (e) {
      setError(e.message);
    }
  };

  const selectedApp = apps.find((a) => a.app_id === appId);
  return (
    <>
      <section className="card">
        <h2>Deploy an app on the pool</h2>
        <p className="secondary small">
          Only apps on the allow-list can run (no arbitrary commands). The node gets a public Cloudflare quick-tunnel URL if it
          has <code>cloudflared</code>, otherwise its local IP:port.
        </p>
        <div className="form-grid">
          <label className="field">
            App
            <select value={appId} onChange={(e) => setAppId(e.target.value)}>
              {apps.map((a) => (
                <option key={a.app_id} value={a.app_id}>
                  {a.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Node
            <select value={nodeId} onChange={(e) => setNodeId(e.target.value)}>
              <option value="">auto (most free RAM)</option>
              {online.map((n) => (
                <option key={n.node_id} value={n.node_id}>
                  {n.node_id} ({n.hostname})
                </option>
              ))}
            </select>
          </label>
        </div>
        {selectedApp && <p className="muted small" style={{ marginTop: "0.5rem" }}>{selectedApp.description}</p>}
        {selectedApp?.kind === "docker" && nodeId && state.nodes[nodeId] && !state.nodes[nodeId].docker && (
          <div className="banner warning">This node reported that Docker is not available; the deploy will fail.</div>
        )}
        <button className="primary" disabled={!isAdmin || online.length === 0} onClick={() => act("/deployments", { app_id: appId, node_id: nodeId || null })}>
          Deploy
        </button>
        {error && <div className="banner error" style={{ marginTop: "0.8rem" }}>{error}</div>}
      </section>

      <section className="card">
        <h2>Deployments</h2>
        {deployments.length === 0 ? (
          <div className="empty">Nothing deployed yet.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>App</th>
                  <th>Node</th>
                  <th>Status</th>
                  <th>URL</th>
                  <th>Started</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {deployments.map((d) => (
                  <tr key={d.id}>
                    <td>
                      {d.app_id} <span className="muted small mono">{d.id}</span>
                    </td>
                    <td>{d.node_id}</td>
                    <td title={d.detail}>
                      <StatusPill status={d.status} />
                      {d.detail && <div className="muted small" style={{ whiteSpace: "normal", maxWidth: 320 }}>{d.detail}</div>}
                    </td>
                    <td>{d.url && d.status === "running" ? <a href={d.url} target="_blank" rel="noreferrer">{d.url}</a> : "-"}</td>
                    <td>{fmtTime(d.created_at)}</td>
                    <td>
                      <div className="row">
                        <button disabled={!isAdmin || d.status === "stopped"} onClick={() => act(`/deployments/${d.id}/stop`)}>
                          Stop
                        </button>
                        <button disabled={!isAdmin} onClick={() => act(`/deployments/${d.id}/restart`)}>
                          Restart
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
