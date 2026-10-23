// Home: create a pool (you become its admin) or open an existing room.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setAdminToken } from "../api.js";
import ThemeToggle from "../components/ThemeToggle.jsx";
import { CopyBox } from "../components/ui.jsx";

export default function Home() {
  const [room, setRoom] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [code, setCode] = useState("");
  const navigate = useNavigate();

  const create = async () => {
    setBusy(true);
    setError("");
    try {
      const r = await api("/api/rooms", { method: "POST" });
      setAdminToken(r.room_code, r.admin_token);
      setRoom(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="topbar">
        <a className="brand" href="/">
          <img src="/favicon.svg" alt="" /> PoolForge
        </a>
        <span className="spacer" />
        <ThemeToggle />
      </header>
      <main className="stack">
        <section className="hero">
          <h1>Pool your laptops. Train bigger models.</h1>
          <p>
            PoolForge splits a neural network by layers across several laptops, so together they can train a model that does not
            fit in one laptop&apos;s RAM. It can also host small web apps on the pool.
          </p>
          <button className="primary" onClick={create} disabled={busy}>
            {busy ? "Creating..." : "Create Pool"}
          </button>
          {error && <div className="banner error" style={{ marginTop: "1rem" }}>Could not create a pool: {error}</div>}
        </section>

        {room && (
          <section className="card stack">
            <div className="card-head">
              <h2>Pool {room.room_code} is ready</h2>
              <button className="primary" onClick={() => navigate(`/room/${room.room_code}/nodes`)}>
                Open dashboard
              </button>
            </div>
            <div>
              <h3>1. Send this link to your friends</h3>
              <CopyBox text={room.join_url} />
            </div>
            <div>
              <h3>2. They run this command on their laptop</h3>
              <CopyBox text={room.join_command} />
            </div>
            <div>
              <h3>Your admin token (needed for Start / Stop / Deploy - keep it private)</h3>
              <CopyBox text={room.admin_token} />
              <p className="muted small">It is saved in this browser. To control the pool from another browser, paste it there.</p>
            </div>
          </section>
        )}

        <section className="card">
          <h2>Open an existing pool</h2>
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault();
              if (code.trim()) navigate(`/room/${code.trim().toUpperCase()}/nodes`);
            }}
          >
            <input value={code} onChange={(e) => setCode(e.target.value)} placeholder="Room code, e.g. X7K2AB" maxLength={6} aria-label="Room code" />
            <button type="submit">Open</button>
          </form>
        </section>

        <section className="grid cols-3">
          <div className="card">
            <h3>Pipeline parallelism</h3>
            <p className="secondary small">Each laptop holds only its own contiguous block of layers. Activations go forward and gradients go back over the network.</p>
          </div>
          <div className="card">
            <h3>No port forwarding</h3>
            <p className="secondary small">Agents connect outbound to this coordinator. Tensors are relayed through it, so any home Wi-Fi works.</p>
          </div>
          <div className="card">
            <h3>Honest trade-off</h3>
            <p className="secondary small">RAM is not literally merged, and every step pays network latency. You can fit bigger models, but each step is slower.</p>
          </div>
        </section>
      </main>
    </>
  );
}
