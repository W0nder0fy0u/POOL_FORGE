// Light / dark / system theme switch. The choice is remembered in this browser.
import { useState } from "react";

const ORDER = ["system", "light", "dark"];
const LABEL = { system: "Theme: auto", light: "Theme: light", dark: "Theme: dark" };

export function applyTheme(choice) {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
}

export function savedTheme() {
  try {
    return localStorage.getItem("pf_theme") || "system";
  } catch {
    return "system";
  }
}

export default function ThemeToggle() {
  const [choice, setChoice] = useState(savedTheme);
  const next = () => {
    const c = ORDER[(ORDER.indexOf(choice) + 1) % ORDER.length];
    setChoice(c);
    applyTheme(c);
    try {
      localStorage.setItem("pf_theme", c);
    } catch {
      /* storage blocked: still switches for this page view */
    }
  };
  return (
    <button className="theme-toggle" onClick={next} title="Switch light / dark theme">
      {LABEL[choice]}
    </button>
  );
}
