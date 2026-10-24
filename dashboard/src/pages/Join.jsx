// Join page (what the shared link opens): shows the commands to run, for
// Windows or Linux/macOS. Everything goes into a "poolforge-env" virtual
// environment and is run by its full path, so it works without admin rights,
// without PATH changes, without "activate" (blocked by PowerShell's default
// policy) and on distros that forbid system-wide pip installs (Ubuntu/Debian).
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import ThemeToggle from "../components/ThemeToggle.jsx";
import { CopyBox } from "../components/ui.jsx";

const CUDA_TORCH = "https://download.pytorch.org/whl/cu126";
const isWindows = () => /Windows/i.test(navigator.userAgent);

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

export default function Join() {
  const code = useParams().code.toUpperCase();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  const [os, setOs] = useState(() => (isWindows() ? "windows" : "unix"));
  const [nvidia, setNvidia] = useState(false);

  useEffect(() => {
    api(`/api/rooms/${code}`).then(setInfo).catch((e) => setError(e.message));
  }, [code]);

  const online = info ? info.nodes.filter((n) => n.online).length : 0;
  const cmd = info && commands(os, info, code, nvidia);
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
              {online} laptop{online === 1 ? " is" : "s are"} in this pool right now. Run the command below to add yours. It connects
              out to the coordinator, so no router or firewall changes are needed.
            </p>
            <section className="card stack">
              <div className="row os-tabs" role="group" aria-label="Operating system">
                <button className={os === "windows" ? "active" : ""} aria-pressed={os === "windows"} onClick={() => setOs("windows")}>
                  Windows
                </button>
                <button className={os === "unix" ? "active" : ""} aria-pressed={os === "unix"} onClick={() => setOs("unix")}>
                  Linux / macOS
                </button>
              </div>
              <div>
                <h3>Step 1 - one-time setup (Python 3.10+)</h3>
                {os === "windows" && (
                  <label className="row small secondary" style={{ marginBottom: "0.5rem" }}>
                    <input type="checkbox" checked={nvidia} onChange={(e) => setNvidia(e.target.checked)} />
                    This laptop has an NVIDIA GPU (installs the GPU build of PyTorch, ~2.5 GB)
                  </label>
                )}
                <CopyBox text={cmd.setup} />
                <p className="muted small">
                  {os === "windows"
                    ? "Run in PowerShell or cmd, in the folder where you want PoolForge. No Python yet? Install it from python.org (the py command comes with it)."
                    : "Run in a terminal. On Ubuntu/Debian, if venv is missing: sudo apt install python3-venv. An NVIDIA GPU works out of the box on Linux; Apple Silicon GPUs too."}{" "}
                  It downloads PoolForge from this server (no GitHub account needed). Already set up? Skip to step 2.
                </p>
              </div>
              <div>
                <h3>Step 2 - join (run from the same folder)</h3>
                <CopyBox text={cmd.join} />
                <p className="muted small">The agent trains on your GPU automatically if it has one. Press Ctrl+C to leave the pool.</p>
              </div>
              <details>
                <summary className="secondary">Optional flags</summary>
                <ul className="secondary small">
                  <li>
                    <code>--ram-cap-gb 2</code> share at most 2 GB of RAM
                  </li>
                  <li>
                    <code>--vram-cap-gb 2</code> share at most 2 GB of GPU memory
                  </li>
                  <li>
                    <code>--device cpu</code> do not use the GPU
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
