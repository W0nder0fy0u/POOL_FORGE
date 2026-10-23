// Join page (what the shared link opens): shows the one command to run.
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { CopyBox } from "../components/ui.jsx";

export default function Join() {
  const code = useParams().code.toUpperCase();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api(`/api/rooms/${code}`).then(setInfo).catch((e) => setError(e.message));
  }, [code]);

  const online = info ? info.nodes.filter((n) => n.online).length : 0;
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/">
          <img src="/favicon.svg" alt="" /> PoolForge
        </a>
      </header>
      <main className="stack" style={{ maxWidth: 820 }}>
        <h1>Join pool {code}</h1>
        {error && <div className="banner error">{error}</div>}
        {info && (
          <>
            <p className="secondary">
              {online} laptop{online === 1 ? " is" : "s are"} in this pool right now. Run the command below to add yours. It connects
              out to the coordinator, so no router or firewall changes are needed.
            </p>
            <section className="card stack">
              <div>
                <h3>Step 1 - one-time setup (Python 3.10+)</h3>
                <CopyBox text={"git clone <this project's repo URL> poolforge\ncd poolforge\npip install -e ."} />
                <p className="muted small">Already set up? Skip to step 2. On Windows use PowerShell or cmd; the commands are the same.</p>
              </div>
              <div>
                <h3>Step 2 - join</h3>
                <CopyBox text={info.join_command} />
              </div>
              <details>
                <summary className="secondary">Optional flags</summary>
                <ul className="secondary small">
                  <li>
                    <code>--ram-cap-gb 2</code> share at most 2 GB of RAM
                  </li>
                  <li>
                    <code>--node-id my-laptop</code> pick a name (also lets the node resume after restarting the agent)
                  </li>
                  <li>
                    <code>--transport tcp</code> direct connections (same Wi-Fi / Tailscale only, faster)
                  </li>
                  <li>
                    <code>--encrypt-passphrase SECRET</code> encrypt tensors (everyone must use the same passphrase)
                  </li>
                </ul>
              </details>
            </section>
            <Link to={`/room/${code}/nodes`}>Watch the pool dashboard</Link>
          </>
        )}
      </main>
    </>
  );
}
