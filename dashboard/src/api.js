// Small helpers for talking to the coordinator's REST API.

// Admin tokens are kept per room in localStorage, so a refresh keeps you admin.
export function getAdminToken(code) {
  try {
    return localStorage.getItem(`pf_admin_${code}`) || "";
  } catch {
    return "";
  }
}

export function setAdminToken(code, token) {
  try {
    if (token) localStorage.setItem(`pf_admin_${code}`, token);
    else localStorage.removeItem(`pf_admin_${code}`);
  } catch {
    /* private mode: token only lives for this page view */
  }
}

// fetch wrapper: JSON in/out, throws Error(detail) on HTTP errors.
export async function api(path, { method = "GET", body, token } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers["X-Admin-Token"] = token;
  const res = await fetch(path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    let detail = data && data.detail ? data.detail : `HTTP ${res.status}`;
    if (Array.isArray(detail)) detail = detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
    throw new Error(detail);
  }
  return data;
}

export function wsUrl(path) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}
