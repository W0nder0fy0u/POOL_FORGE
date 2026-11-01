// Join page (what the shared link opens): download the PoolForge app, open it,
// paste this page's link. The app installs everything it needs on first run.
// When the server hands out source (POOLFORGE_SERVE_SOURCE=1, for developers) the
// old pip commands are shown too. They go into a "poolforge-env" virtual
// environment run by its full path, so they work without admin rights, PATH
// changes or "activate" (blocked by PowerShell's default policy).
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import ThemeToggle from "../components/ThemeToggle.jsx";
import { CopyBox } from "../components/ui.jsx";

const CUDA_TORCH = "https://download.pytorch.org/whl/cu126";
const isWindows = () => /Windows/i.test(navigator.userAgent);
const OS_LABEL = { windows: "Windows", linux: "Linux" };

function commands(os, info, code, nvidia) {
  if (os === "windows") {
    const bin = ".\\poolforge-env\\Scripts\\";
    return {
      setup: [
        "py -m venv poolforge-env",
        // Windows pip gives the CPU-only PyTorch, so GPU users install the CUDA build first.
        ...(nvidia ? [`${bin}python -m pip install torch --index-url ${CUDA_TORCH}`] : []),
        `${bin}python -m pip install ${info.install_url}`,
      ].join("\n"),
      join: `${bin}poolforge join ${code} --server ${info.server}`,
    };
  }
  const bin = "./poolforge-env/bin/";
  return {
    setup: ["python3 -m venv poolforge-env", `${bin}python -m pip install ${info.install_url}`].join("\n"),
    join: `${bin}poolforge join ${code} --server ${info.server}`,
  };
}

function PythonInstall({ info, code }) {
  const [os, setOs] = useState(() => (isWindows() ? "windows" : "unix"));
  const [nvidia, setNvidia] = useState(false);
  const cmd = commands(os, info, code, nvidia);
  return (
    <details className="card">
      <summary className="secondary">Advanced: install with Python instead (developers)</summary>
      <div className="stack" style={{ marginTop: "0.8rem" }}>
        <div className="row os-tabs" role="group" aria-label="Operating system">
          <button className={os === "windows" ? "active" : ""} aria-pressed={os === "windows"} onClick={() => setOs("windows")}>
            Windows
          </button>
          <button className={os === "unix" ? "active" : ""} aria-pressed={os === "unix"} onClick={() => setOs("unix")}>
            Linux / macOS
          </button>
        </div>
        <div>
          <h3>One-time setup (Python 3.10+)</h3>
          {os === "windows" && (
            <label className="row small secondary" style={{ marginBottom: "0.5rem" }}>
              <input type="checkbox" checked={nvidia} onChange={(e) => setNvidia(e.target.checked)} />
              This laptop has an NVIDIA GPU (installs the GPU build of PyTorch, ~2.5 GB)
            </label>
          )}
          <CopyBox text={cmd.setup} />
        </div>
        <div>
          <h3>Join (from the same folder)</h3>
          <CopyBox text={cmd.join} />
          <p className="muted small">
            Optional flags: <code>--ram-cap-gb 2</code>, <code>--vram-cap-gb 2</code>, <code>--device cpu</code>,{" "}
            <code>--node-id my-laptop</code>, <code>--transport tcp</code>, <code>--encrypt-passphrase SECRET</code>.
          </p>
        </div>
      </div>
    </details>
  );
}

export default function Join() {
  const code = useParams().code.toUpperCase();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api(`/api/rooms/${code}`).then(setInfo).catch((e) => setError(e.message));
  }, [code]);

  const online = info ? info.nodes.filter((n) => n.online).length : 0;
  const mine = isWindows() ? "windows" : "linux";
  // The visitor's own OS first, as the big button.
  const downloads = info ? Object.entries(info.downloads || {}).sort(([a]) => (a === mine ? -1 : 1)) : [];
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/">
          <img src="/favicon.svg" alt="" /> PoolForge
        </a>
        <span className="spacer" />
        <ThemeToggle />
      </header>
      <main className="stack" style={{ maxWidth: 820 }}>
        <h1>Join pool {code}</h1>
        {error && <div className="banner error">{error}</div>}
        {info && (
          <>
            <p className="secondary">
              {online} laptop{online === 1 ? " is" : "s are"} in this pool right now. Add yours in two steps. It connects out to
              the pool server, so no router or firewall changes are needed.
            </p>
            <section className="card stack">
              <div>
                <h3>Step 1 - download PoolForge</h3>
                {downloads.length ? (
                  <div className="row" style={{ flexWrap: "wrap", gap: "0.6rem" }}>
                    {downloads.map(([os, url], i) => (
                      <a key={os} className={`button${i === 0 ? " primary" : ""}`} href={url}>
                        Download for {OS_LABEL[os] || os}
                      </a>
                    ))}
                  </div>
                ) : (
                  <div className="banner warning">App downloads aren't set up on this server yet. Ask the pool owner for the PoolForge app.</div>
                )}
                <p className="muted small">
                  About 20 MB. The first time you open it, it sets itself up once: about 200 MB, or about 2.5 GB if your laptop has an
                  NVIDIA GPU (it then trains on the GPU). No Python or other installs needed.
                  {mine === "windows"
                    ? ' If Windows says "Windows protected your PC", click More info, then Run anyway.'
                    : " On Linux, make the file runnable first: chmod +x PoolForge-linux, then ./PoolForge-linux"}
                </p>
              </div>
              <div>
                <h3>Step 2 - open it and paste this link</h3>
                <CopyBox text={info.join_url} />
                <p className="muted small">Click Join and keep the window open. Close it (or click Leave) to leave the pool.</p>
              </div>
            </section>
            {info.install_url && <PythonInstall info={info} code={code} />}
            <Link to={`/room/${code}/nodes`}>Watch the pool dashboard</Link>
          </>
        )}
      </main>
    </>
  );
}
