/**
 * Backend ("SHARP Engine") base URL resolution.
 *
 * In the hosted model, the website is served from Vercel but the heavy ML work
 * runs on the *visitor's own* machine. Every visitor runs the engine locally, so
 * the correct default is http://localhost:8000 for everyone.
 *
 * Resolution priority (highest first):
 *   1. ?backend=<url>     query param (also persisted to localStorage)
 *   2. localStorage        ("sharp_api_base")
 *   3. VITE_API_BASE       build-time env var
 *   4. http://localhost:8000
 */

const STORAGE_KEY = 'sharp_api_base';
const DEFAULT_API_BASE = 'http://localhost:8000';

const stripTrailingSlash = (url) => url.replace(/\/+$/, '');

const fromQueryParam = () => {
  try {
    const params = new URLSearchParams(window.location.search);
    const raw = params.get('backend');
    if (raw) {
      const value = stripTrailingSlash(raw.trim());
      // Persist so the override survives navigation / param-less reloads.
      window.localStorage.setItem(STORAGE_KEY, value);
      return value;
    }
  } catch {
    // Ignore (SSR / disabled storage).
  }
  return null;
};

const fromStorage = () => {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value ? stripTrailingSlash(value) : null;
  } catch {
    return null;
  }
};

const fromEnv = () => {
  const value = import.meta.env?.VITE_API_BASE;
  return value ? stripTrailingSlash(value) : null;
};

export const getApiBase = () =>
  fromQueryParam() || fromStorage() || fromEnv() || DEFAULT_API_BASE;

export const setApiBase = (url) => {
  const value = stripTrailingSlash(url.trim());
  try {
    if (value) {
      window.localStorage.setItem(STORAGE_KEY, value);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Ignore.
  }
  return value || DEFAULT_API_BASE;
};

export const isLocalBackend = (base = getApiBase()) =>
  /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i.test(base);

export { DEFAULT_API_BASE };
