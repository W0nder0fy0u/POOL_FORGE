// Live room state: one WebSocket to /ws/dashboard/{code} feeds every page.
//
// The coordinator first sends a "snapshot" (everything so far), then small
// events: node, job, step, node_step, deployment, locked. This file folds
// them into one state object with a reducer.

import { createContext, useContext, useEffect, useReducer, useRef, useState } from "react";
import { api, getAdminToken, setAdminToken, wsUrl } from "./api.js";

const RoomContext = createContext(null);
export const useRoom = () => useContext(RoomContext);

const initial = {
  nodes: {}, // node_id -> latest stats
  nodeOrder: [], // join order -> fixed colour per node
  job: null,
  metrics: [], // per-step loss/time from the last stage
  nodeMetrics: [], // per-step, per-node network + RAM
  deployments: {},
  locked: false,
  log: [],
};

function addLog(state, text) {
  if (!text) return state.log;
  return [{ t: Date.now(), text }, ...state.log].slice(0, 50);
}

function withNode(state, node) {
  const nodes = { ...state.nodes, [node.node_id]: node };
  const nodeOrder = state.nodeOrder.includes(node.node_id) ? state.nodeOrder : [...state.nodeOrder, node.node_id];
  return { nodes, nodeOrder };
}

function reducer(state, ev) {
  switch (ev.type) {
    case "snapshot": {
      const nodes = {};
      ev.nodes.forEach((n) => (nodes[n.node_id] = n));
      const deployments = {};
      ev.deployments.forEach((d) => (deployments[d.id] = d));
      return {
        ...initial,
        nodes,
        nodeOrder: ev.nodes.map((n) => n.node_id),
        job: ev.job,
        metrics: ev.metrics,
        nodeMetrics: ev.node_metrics,
        deployments,
        locked: ev.locked,
        log: state.log,
      };
    }
    case "node":
      return { ...state, ...withNode(state, ev.node) };
    case "job": {
      const isNew = !state.job || state.job.job_id !== ev.job.job_id;
      return {
        ...state,
        job: ev.job,
        metrics: isNew ? [] : state.metrics,
        nodeMetrics: isNew ? [] : state.nodeMetrics,
        log: addLog(state, ev.message),
      };
    }
    case "step": {
      if (!state.job || state.job.job_id !== ev.job_id) return state;
      // after a resume from checkpoint, steps repeat: replace instead of duplicating
      const metrics = state.metrics.filter((m) => m.step < ev.step);
      metrics.push(ev);
      return { ...state, metrics, job: { ...state.job, last_step: ev.step } };
    }
    case "node_step": {
      if (!state.job || state.job.job_id !== ev.job_id) return state;
      const nodeMetrics = state.nodeMetrics.filter((m) => !(m.node_id === ev.node_id && m.step >= ev.step));
      nodeMetrics.push(ev);
      return { ...state, nodeMetrics: nodeMetrics.slice(-4000) };
    }
    case "deployment":
      return { ...state, deployments: { ...state.deployments, [ev.deployment.id]: ev.deployment } };
    case "locked":
      return { ...state, locked: ev.locked };
    case "log":
      return { ...state, log: addLog(state, ev.text) };
    default:
      return state;
  }
}

export function RoomProvider({ code, children }) {
  const [state, dispatch] = useReducer(reducer, initial);
  const [conn, setConn] = useState("connecting"); // connecting | live | reconnecting | missing
  const [token, setToken] = useState(() => getAdminToken(code));
  const [isAdmin, setIsAdmin] = useState(false);
  const wsRef = useRef(null);

  // Accept "#admin=TOKEN" in the URL once (used by the dev scripts), then hide it.
  useEffect(() => {
    const m = window.location.hash.match(/admin=([^&]+)/);
    if (m) {
      setAdminToken(code, m[1]);
      setToken(m[1]);
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, [code]);

  // Check whether our token really is this room's admin token.
  useEffect(() => {
    if (!token) return setIsAdmin(false);
    api(`/api/rooms/${code}/admin/check`, { method: "POST", token })
      .then(() => setIsAdmin(true))
      .catch(() => setIsAdmin(false));
  }, [code, token]);

  // Live WebSocket with automatic reconnect (1s, 2s, 4s ... max 10s).
  useEffect(() => {
    let closed = false;
    let delay = 1000;
    let timer = null;
    let ping = null;

    function connect() {
      const ws = new WebSocket(wsUrl(`/ws/dashboard/${code}`));
      wsRef.current = ws;
      ws.onopen = () => {
        setConn("live");
        delay = 1000;
        ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
      };
      ws.onmessage = (e) => {
        try {
          dispatch(JSON.parse(e.data));
        } catch {
          /* ignore malformed event */
        }
      };
      ws.onclose = (e) => {
        clearInterval(ping);
        if (closed) return;
        if (e.code === 4404) return setConn("missing");
        setConn("reconnecting");
        timer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 10000);
      };
    }
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      clearInterval(ping);
      wsRef.current && wsRef.current.close();
    };
  }, [code]);

  const saveToken = (t) => {
    setAdminToken(code, t);
    setToken(t);
  };
  const log = (text) => dispatch({ type: "log", text });
  // Admin action helper: shows errors in the event log and rethrows.
  const admin = async (path, body) => {
    try {
      return await api(`/api/rooms/${code}${path}`, { method: "POST", body, token });
    } catch (err) {
      log(`Error: ${err.message}`);
      throw err;
    }
  };

  const value = { code, state, conn, token, isAdmin, saveToken, admin, log };
  return <RoomContext.Provider value={value}>{children}</RoomContext.Provider>;
}

// Fixed colour per node (by join order), so a node keeps its colour everywhere.
export function nodeColor(state, nodeId) {
  const i = state.nodeOrder.indexOf(nodeId);
  return `var(--series-${(Math.max(i, 0) % 8) + 1})`;
}
