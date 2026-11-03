// Download buttons for the PoolForge desktop app. Always shown (home and join pages),
// on any device: phones can't run the app, so they get a note to use a laptop instead.
import { useEffect, useState } from "react";
import { api } from "../api.js";

const OS_LABEL = { windows: "Windows", linux: "Linux" };
const isPhone = () => /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);
// Android reports "Linux" too, so only a desktop Linux browser gets Linux first.
export const visitorOs = () => (/Linux/i.test(navigator.userAgent) && !isPhone() ? "linux" : "windows");

export default function AppDownloads({ downloads }) {
  // Pages that already have the links (the join page's room info) pass them in.
  const [fetched, setFetched] = useState(null);
  useEffect(() => {
    if (!downloads) api("/api/downloads").then(setFetched).catch(() => setFetched({}));
  }, [downloads]);
  const links = downloads || fetched;

  const mine = visitorOs();
  // The visitor's own OS first, as the big button.
  const entries = Object.entries(links || {}).sort(([a]) => (a === mine ? -1 : 1));
  return (
    <div className="stack" style={{ gap: "0.6rem" }}>
      {isPhone() && (
        <div className="banner info">PoolForge runs on a Windows or Linux laptop. Download it there, or send this page to your computer.</div>
      )}
      <div className="row" style={{ flexWrap: "wrap", gap: "0.6rem" }}>
        {entries.map(([os, url], i) => (
          <a key={os} className={`button${i === 0 ? " primary" : ""}`} href={url}>
            Download for {OS_LABEL[os] || os}
          </a>
        ))}
        {links && !entries.length && <span className="muted small">Couldn't load the download links. Refresh the page in a moment.</span>}
      </div>
      <p className="muted small">
        About 20 MB. The first time you open it, it sets itself up once: about 200 MB, or about 2.5 GB if your laptop has an NVIDIA
        GPU (it then trains on the GPU). No Python or other installs needed.
        {mine === "windows"
          ? ' If Windows says "Windows protected your PC", click More info, then Run anyway.'
          : " On Linux, make the file runnable first: chmod +x PoolForge-linux, then ./PoolForge-linux"}
      </p>
    </div>
  );
}
