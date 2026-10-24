// Frame around every room page: top bar with tabs, connection state,
// admin/viewer status, and the Chat & Call side panel.
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useParams } from "react-router-dom";
import { useCall } from "../call.js";
import { RoomProvider, useRoom } from "../room.jsx";
import TalkPanel from "./TalkPanel.jsx";
import ThemeToggle from "./ThemeToggle.jsx";
import { StatusPill } from "./ui.jsx";

// Open by default on wide screens; on phones it is a bottom sheet, so start closed.
function panelWasOpen() {
  try {
    const saved = localStorage.getItem("pf_talk_open");
    if (saved !== null) return saved === "1";
  } catch {
    /* storage blocked */
  }
  return window.innerWidth > 900;
}

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
  const call = useCall(); // lives here so the call survives tab switches
  const [talkOpen, setTalkOpen] = useState(panelWasOpen);
  const [unread, setUnread] = useState(0);
  const seen = useRef(state.chat.length);

  // Count messages that arrive while the panel is closed.
  useEffect(() => {
    if (talkOpen) setUnread(0);
    else setUnread((u) => u + Math.max(0, state.chat.length - seen.current));
    seen.current = state.chat.length;
  }, [state.chat.length, talkOpen]);

  const toggleTalk = (open) => {
    setTalkOpen(open);
    try {
      localStorage.setItem("pf_talk_open", open ? "1" : "0");
    } catch {
      /* storage blocked */
    }
  };

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
        <button className={`talk-toggle ${talkOpen ? "active" : ""}`} onClick={() => toggleTalk(!talkOpen)} aria-pressed={talkOpen}>
          {call.inCall && <span className="dot good" />}
          Chat &amp; Call
          {unread > 0 && <span className="badge">{unread}</span>}
        </button>
        <ThemeToggle />
      </header>
      <div className={`room-body ${talkOpen ? "with-talk" : ""}`}>
        <main className="stack">
          {conn === "missing" && <div className="banner error">Room {code} does not exist. Check the code or create a new pool.</div>}
          {conn === "reconnecting" && <div className="banner warning">Lost connection to the coordinator. Reconnecting...</div>}
          {!isAdmin && conn !== "missing" && <AdminTokenForm />}
          <Outlet />
        </main>
        {talkOpen && <TalkPanel call={call} onClose={() => toggleTalk(false)} />}
      </div>
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
