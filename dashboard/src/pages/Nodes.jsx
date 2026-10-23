// Nodes page: one live card per laptop in the pool.
import { nodeColor, useRoom } from "../room.jsx";
import { CopyBox, Meter, Stat, StatusPill, fmtNum } from "../components/ui.jsx";

export default function Nodes() {
  const { code, state, conn, isAdmin, admin } = useRoom();
  const nodes = state.nodeOrder.map((id) => state.nodes[id]).filter(Boolean);
  const online = nodes.filter((n) => n.online);
  const freeGb = online.reduce((s, n) => s + (n.ram_free_gb || 0), 0);
  const totalGb = online.reduce((s, n) => s + (n.ram_total_gb || 0), 0);
  const joinCmd = `poolforge join ${code} --server ${window.location.origin}`;

  return (
    <>
      <div className="grid cols-4">
        <Stat label="Nodes online" value={online.length} sub={`${nodes.length} joined in total`} />
        <Stat label="Pool free RAM (reported)" value={`${fmtNum(freeGb, 1)} GB`} sub="after --ram-cap-gb limits" />
        <Stat label="Pool total RAM" value={`${fmtNum(totalGb, 1)} GB`} />
        <Stat label="Room" value={state.locked ? "Locked" : "Open"} sub={state.locked ? "no new nodes can join" : "anyone with the code can join"} />
      </div>

      {nodes.length === 0 && conn === "live" && (
        <div className="card stack">
          <h2>Waiting for laptops...</h2>
          <p className="secondary">Run this on each laptop that should join the pool:</p>
          <CopyBox text={joinCmd} />
          <p className="muted small">
            Share link: <a href={`/join/${code}`}>{window.location.origin}/join/{code}</a>
          </p>
        </div>
      )}

      <div className="grid cols-3">
        {nodes.map((n) => (
          <NodeCard key={n.node_id} node={n} color={nodeColor(state, n.node_id)} />
        ))}
      </div>

      {isAdmin && (
        <div className="row">
          <button onClick={() => admin("/lock", { locked: !state.locked })}>{state.locked ? "Unlock room" : "Lock room (no new joins)"}</button>
          <span className="muted small">Join command: </span>
          <code className="small">{joinCmd}</code>
        </div>
      )}
    </>
  );
}

function NodeCard({ node, color }) {
  const status = node.online ? node.status || "idle" : "offline";
  const usedGb = node.ram_total_gb - (node.ram_free_gb || 0);
  return (
    <div className="card" style={{ opacity: node.online ? 1 : 0.55, borderTop: `3px solid ${color}` }}>
      <div className="card-head">
        <div>
          <h2 style={{ marginBottom: 0 }}>{node.hostname}</h2>
          <div className="muted small mono">{node.node_id}</div>
        </div>
        <StatusPill status={status} />
      </div>
      <div className="muted small">{node.os}</div>
      {node.ram_total_gb !== undefined ? (
        <>
          <Meter
            label={node.ram_cap_gb ? `RAM free (capped at ${node.ram_cap_gb} GB)` : "RAM free"}
            value={node.ram_free_gb}
            max={node.ram_cap_gb || node.ram_total_gb}
            text={`${fmtNum(node.ram_free_gb, 2)} / ${fmtNum(node.ram_cap_gb || node.ram_total_gb, 1)} GB`}
            color="var(--series-3)"
          />
          <Meter label="CPU" value={node.cpu_percent} max={100} text={`${fmtNum(node.cpu_percent, 0)} %`} color="var(--series-1)" />
          <div className="row small secondary" style={{ marginTop: "0.6rem" }}>
            <span>Agent RAM: {fmtNum(node.proc_ram_mb, 0)} MB</span>
            <span>Machine used: {fmtNum(usedGb, 1)} GB</span>
          </div>
          <div className="row" style={{ marginTop: "0.5rem" }}>
            <span className="pill">transport: {node.transport}</span>
            <span className="pill">{node.docker ? "docker ok" : "no docker"}</span>
            <span className="pill">{node.cloudflared ? "cloudflared ok" : "no cloudflared"}</span>
          </div>
        </>
      ) : (
        <p className="muted small">No stats yet.</p>
      )}
    </div>
  );
}
