// Visualises which node owns which layers: one bar segment per stage, sized by
// number of layers, plus the list of layers coloured by owner.
import { nodeColor, useRoom } from "../room.jsx";

function deviceLabel(state, p) {
  if (p.device === "cuda" || p.device === "mps") return `GPU (${state.nodes[p.node_id]?.gpu_name || p.device})`;
  return "CPU";
}

export default function SplitPlan({ plan, layerTypes }) {
  const { state } = useRoom();
  if (!plan || !plan.length) return <div className="empty">No split plan yet.</div>;
  const total = plan[plan.length - 1].end_layer;
  const owner = (i) => plan.find((p) => i >= p.start_layer && i < p.end_layer);

  return (
    <div>
      <div className="plan" role="img" aria-label="Layer split across nodes">
        {plan.map((p) => (
          <div
            key={p.node_id}
            className="seg"
            title={`${p.node_id}: layers ${p.start_layer}-${p.end_layer - 1}`}
            style={{ flex: p.end_layer - p.start_layer, background: nodeColor(state, p.node_id) }}
          >
            {p.end_layer - p.start_layer} layers
          </div>
        ))}
      </div>
      <div className="plan-legend">
        {plan.map((p) => (
          <span key={p.node_id}>
            <span className="dot" style={{ background: nodeColor(state, p.node_id), marginRight: 6 }} />
            Stage {p.stage_index + 1}: <b>{p.node_id}</b> on {deviceLabel(state, p)} - layers {p.start_layer}-{p.end_layer - 1}, ~
            {p.estimated_mb} MB of {Math.round(p.budget_mb)} MB {p.device === "cuda" ? "VRAM" : "RAM"}
          </span>
        ))}
      </div>
      {layerTypes && (
        <div className="layers" aria-label="Layers">
          {layerTypes.slice(0, total).map((t, i) => (
            <span key={i} style={{ borderLeftColor: nodeColor(state, owner(i)?.node_id) }} title={`layer ${i}`}>
              {i}:{t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
