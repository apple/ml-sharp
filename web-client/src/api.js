/**
 * Thin transport layer for talking to the local SHARP Engine.
 *
 * Why native fetch instead of axios: when the page is served over HTTPS (Vercel)
 * and the engine runs at http://localhost on the visitor's machine, Chrome 142+
 * gates the request behind the "Local Network Access" permission. fetch() lets us
 * pass `targetAddressSpace: 'local'`, which axios (XHR-based) does not forward.
 * The job/poll state machine in App.jsx is unchanged — only the transport differs.
 */

import { getApiBase, isLocalBackend } from './config';

export const apiUrl = (path) =>
  `${getApiBase()}${path.startsWith('/') ? path : `/${path}`}`;

// Only hint `local` for loopback backends. Setting it for a non-local address
// makes the request fail by spec, so we gate it on isLocalBackend().
const withLocalHint = (init) =>
  isLocalBackend() ? { ...init, targetAddressSpace: 'local' } : init;

/**
 * fetch() with a timeout (AbortController) and the local-network hint applied.
 * Returns the raw Response; callers decide how to read the body.
 */
export async function apiFetch(path, { timeout, ...init } = {}) {
  const controller = new AbortController();
  const timer = timeout
    ? setTimeout(() => controller.abort(), timeout)
    : null;
  try {
    return await fetch(
      apiUrl(path),
      withLocalHint({ ...init, signal: controller.signal })
    );
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** GET that parses JSON, throwing on non-2xx (mirrors axios behavior). */
export async function getJson(path, { timeout } = {}) {
  const res = await apiFetch(path, { method: 'GET', timeout });
  if (!res.ok) {
    const err = new Error(`Request failed (${res.status})`);
    err.status = res.status;
    try {
      err.detail = (await res.json())?.detail;
    } catch {
      /* no body */
    }
    throw err;
  }
  return res.json();
}

/** POST FormData (multipart) and parse the JSON response. */
export async function postForm(path, formData, { timeout } = {}) {
  // Do NOT set Content-Type — the browser sets the multipart boundary.
  const res = await apiFetch(path, { method: 'POST', body: formData, timeout });
  if (!res.ok) {
    const err = new Error(`Upload failed (${res.status})`);
    err.status = res.status;
    try {
      err.detail = (await res.json())?.detail;
    } catch {
      /* no body */
    }
    throw err;
  }
  return res.json();
}

/** GET a binary result and return an object URL for it. */
export async function getBlobUrl(path, { timeout } = {}) {
  const res = await apiFetch(path, { method: 'GET', timeout });
  if (!res.ok) {
    const err = new Error(`Download failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  const blob = await res.blob();
  return window.URL.createObjectURL(blob);
}

/** Lightweight health probe. Returns true iff the engine answers as running. */
export async function checkEngineHealth() {
  try {
    const data = await getJson('/', { timeout: 3000 });
    return data?.status === 'running';
  } catch {
    return false;
  }
}
