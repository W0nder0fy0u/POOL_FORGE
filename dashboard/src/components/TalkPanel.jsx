// "Chat & Call" side panel, shown next to every room page.
// The call itself lives in RoomLayout (useCall), so it keeps running while you
// switch tabs or close this panel.
import { useEffect, useRef, useState } from "react";
import { useRoom } from "../room.jsx";
import { fmtTime } from "./ui.jsx";

export function savedName() {
  try {
    return localStorage.getItem("pf_name") || "";
  } catch {
    return "";
  }
}

function saveName(name) {
  try {
    localStorage.setItem("pf_name", name);
  } catch {
    /* storage blocked: name only lives for this page view */
  }
}

export default function TalkPanel({ call, onClose }) {
  const { state, send, conn } = useRoom();
  const [name, setName] = useState(savedName);
  const [text, setText] = useState("");
  const listRef = useRef(null);
  const who = name.trim();

  // Keep the newest message in view.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [state.chat.length]);

  const submit = (e) => {
    e.preventDefault();
    if (!who || !text.trim()) return;
    send({ type: "chat", name: who, text: text.trim().slice(0, 500) });
    setText("");
  };

  return (
    <aside className="talk card" aria-label="Chat and voice call">
      <div className="card-head">
        <h2 style={{ margin: 0 }}>Chat &amp; Call</h2>
        <button onClick={onClose} aria-label="Close chat panel">Close</button>
      </div>

      <label className="field">
        Your name
        <input
          value={name}
          maxLength={30}
          placeholder="e.g. Asha"
          onChange={(e) => setName(e.target.value)}
          onBlur={() => saveName(who)}
          disabled={call.inCall}
        />
      </label>

      <section className="talk-call">
        <div className="row">
          {!call.inCall ? (
            <button className="primary" disabled={!who || conn !== "live"} onClick={() => (saveName(who), call.join(who))}>
              Join voice call
            </button>
          ) : (
            <>
              <button onClick={call.toggleMute}>{call.muted ? "Unmute" : "Mute"}</button>
              <button className="danger" onClick={call.leave}>Leave call</button>
            </>
          )}
          <span className="muted small">{state.call.length} in call</span>
        </div>
        {call.error && <div className="banner error small">{call.error}</div>}
        {state.call.length > 0 && (
          <ul className="talk-people">
            {state.call.map((p) => (
              <li key={p.peer_id}>
                <span className={`dot ${call.speaking[p.peer_id] ? "good" : "off"}`} />
                {p.name}
                {p.peer_id === state.myPeerId && <span className="muted small"> (you{call.muted ? ", muted" : ""})</span>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="talk-messages" ref={listRef} aria-live="polite">
        {state.chat.length === 0 && <div className="empty small">No messages yet. Say hi!</div>}
        {state.chat.map((m, i) => (
          <div key={i} className={`talk-msg ${m.peer_id === state.myPeerId || (who && m.name === who) ? "mine" : ""}`}>
            <div className="small">
              <b>{m.name}</b> <span className="muted">{fmtTime(m.t)}</span>
            </div>
            <div className="talk-text">{m.text}</div>
          </div>
        ))}
      </div>

      <form className="row talk-input" onSubmit={submit}>
        <input
          value={text}
          maxLength={500}
          onChange={(e) => setText(e.target.value)}
          placeholder={who ? "Message the pool..." : "Enter your name first"}
          disabled={!who || conn !== "live"}
          aria-label="Chat message"
        />
        <button type="submit" disabled={!who || !text.trim() || conn !== "live"}>
          Send
        </button>
      </form>
    </aside>
  );
}
