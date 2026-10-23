// Frame around every room page: top bar with tabs, connection state,
// admin/viewer status, and the event log banner.
import { useState } from "react";
import { NavLink, Outlet, useParams } from "react-router-dom";
import { RoomProvider, useRoom } from "../room.jsx";
import ThemeToggle from "./ThemeToggle.jsx";
import { StatusPill } from "./ui.jsx";

export default function RoomLayout() {
  const { code } = useParams();
  return (
    <RoomProvider code={code.toUpperCase()}>
      <Frame />
    </RoomProvider>
  );
}

const CONN_LABEL = { connecting: "Connecting...", live: "Live", reconnecting: "Reconnecting...", missing: "Room not found" };

function Frame() {
  const { code, conn, isAdmin, state } = useRoom();
  const tabs = [
    ["nodes", "Nodes"],
    ["training", "Training"],
    ["network", "Network"],
    ["deploy", "Deploy"],
    ["reports", "Reports"],
  ];
  const online = Object.values(state.nodes).filter((n) => n.online).length;
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/">
          <img src="/favicon.svg" alt="" /> PoolForge
        </a>
        <span className="pill mono" title="Room code">{code}</span>
        <nav className="tabs">
          {tabs.map(([path, label]) => (
            <NavLink key={path} to={`/room/${code}/${path}`}>
              {label}
            </NavLink>
          ))}
        </nav>
        <span className="spacer" />
        <span className="pill">{online} node{online === 1 ? "" : "s"} online</span>
        <StatusPill status={isAdmin ? "online" : "idle"} label={isAdmin ? "Admin" : "Viewer"} />
        <StatusPill status={conn} label={CONN_LABEL[conn]} />
        <ThemeToggle />
      </header>
      <main className="stack">
        {conn === "missing" && <div className="banner error">Room {code} does not exist. Check the code or create a new pool.</div>}
        {conn === "reconnecting" && <div className="banner warning">Lost connection to the coordinator. Reconnecting...</div>}
        {!isAdmin && conn !== "missing" && <AdminTokenForm />}
        <Outlet />
      </main>
    </>
  );
}

function AdminTokenForm() {
  const { saveToken, token } = useRoom();
  const [value, setValue] = useState("");
  return (
    <div className="banner info">
      <span className="secondary small">
        {token ? "The saved admin token is not valid for this room." : "Viewer mode: you can watch but not control."} Paste the
        admin token to Start/Stop/Deploy:
      </span>
      <input value={value} onChange={(e) => setValue(e.target.value)} placeholder="admin token" aria-label="Admin token" style={{ flex: "1 1 200px" }} />
      <button onClick={() => saveToken(value.trim())} disabled={!value.trim()}>
        Use token
      </button>
    </div>
  );
}
