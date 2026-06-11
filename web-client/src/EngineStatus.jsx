import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Cpu, CheckCircle2, AlertTriangle, RefreshCw, Copy, Check,
  Download, Settings2, X,
} from 'lucide-react';
import { checkEngineHealth } from './api';
import { getApiBase, setApiBase, isLocalBackend, DEFAULT_API_BASE } from './config';

/**
 * SHARP Engine connection status + setup guidance.
 *
 * In the hosted model the visitor must run the engine on their own machine. This
 * component auto-detects it, shows a green pill when connected, and an actionable
 * setup card (start command + copy button + retry) when it is not — replacing the
 * old blocking alert() dialogs.
 */

// Safari/WebKit blocks HTTPS pages from reaching http://localhost (mixed content),
// so the local-compute model cannot work there. Detect it to warn explicitly.
const isSafari = () => {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /^((?!chrome|android|crios|fxios|edg).)*safari/i.test(ua);
};

// We're "remote" (HTTPS deployment, not localhost dev) when the page itself is
// served over https. That's when the Chrome permission prompt / Safari block apply.
const isRemoteOrigin = () =>
  typeof window !== 'undefined' && window.location.protocol === 'https:';

const POLL_CONNECTED_MS = 15000;
const POLL_DISCONNECTED_MS = 4000;

// Where to get the engine. Override with VITE_ENGINE_REPO at build time.
const ENGINE_REPO =
  import.meta.env?.VITE_ENGINE_REPO || 'https://github.com/apple/ml-sharp';

const macCommand = './start-engine.command';
const shellCommand = './start-engine.sh';

export default function EngineStatus({ onConnectedChange, onClose, forceOpen = false }) {
  const [status, setStatus] = useState('checking'); // checking | connected | disconnected
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [urlDraft, setUrlDraft] = useState(getApiBase());
  const timerRef = useRef(null);

  const safari = isSafari();
  const remote = isRemoteOrigin();
  const apiBase = getApiBase();

  const probe = useCallback(async () => {
    const ok = await checkEngineHealth();
    setStatus(ok ? 'connected' : 'disconnected');
    onConnectedChange?.(ok);
    return ok;
  }, [onConnectedChange]);

  // Poll: fast while disconnected, slow while connected (to notice drops).
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      const ok = await probe();
      if (cancelled) return;
      timerRef.current = setTimeout(
        tick,
        ok ? POLL_CONNECTED_MS : POLL_DISCONNECTED_MS
      );
    };
    tick();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [probe]);

  // The panel is shown when the user opens it, or when the parent forces it open
  // (e.g. after a failed pre-flight). Derived — no setState-in-effect needed.
  const panelOpen = open || forceOpen;

  const closePanel = () => {
    setOpen(false);
    onClose?.(); // let the parent clear its forceOpen flag
  };

  const copyCommand = (cmd) => {
    navigator.clipboard?.writeText(cmd).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const applyUrl = () => {
    setApiBase(urlDraft);
    setShowSettings(false);
    setStatus('checking');
    probe();
  };

  const pill = {
    checking: { dot: 'bg-yellow-400 animate-pulse', text: 'Checking engine...', cls: 'text-yellow-300 border-yellow-500/30 bg-yellow-500/10' },
    connected: { dot: 'bg-green-400', text: 'Engine connected', cls: 'text-green-300 border-green-500/30 bg-green-500/10' },
    disconnected: { dot: 'bg-red-400', text: 'Engine not running', cls: 'text-red-300 border-red-500/30 bg-red-500/10' },
  }[status];

  return (
    <>
      {/* Status pill (always visible) */}
      <button
        onClick={() => (panelOpen ? closePanel() : setOpen(true))}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-all ${pill.cls}`}
        title="SHARP Engine status"
      >
        <span className={`w-2 h-2 rounded-full ${pill.dot}`} />
        {pill.text}
      </button>

      {/* Setup / detail panel */}
      {panelOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-3xl border border-white/10 bg-neutral-900 shadow-2xl p-6 space-y-5 relative">
            <button
              onClick={closePanel}
              className="absolute top-4 right-4 p-2 rounded-lg text-neutral-500 hover:text-white hover:bg-white/10 transition-colors"
            >
              <X size={18} />
            </button>

            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <Cpu size={22} className="text-blue-400" />
              </div>
              <div>
                <h3 className="text-lg font-medium text-white">SHARP Engine</h3>
                <p className="text-xs text-neutral-500">
                  Runs on <span className="font-mono">{apiBase}</span> — your own machine does the processing.
                </p>
              </div>
            </div>

            {/* Connected */}
            {status === 'connected' && (
              <div className="flex items-start gap-3 rounded-2xl bg-green-500/5 border border-green-500/20 p-4">
                <CheckCircle2 size={20} className="text-green-400 shrink-0 mt-0.5" />
                <div className="text-sm text-neutral-300">
                  Connected and ready. Drop in a photo to generate a 3D scene — it runs entirely on this computer.
                </div>
              </div>
            )}

            {/* Safari hard-block warning (only relevant on a remote HTTPS origin) */}
            {status !== 'connected' && remote && safari && (
              <div className="flex items-start gap-3 rounded-2xl bg-red-500/5 border border-red-500/20 p-4">
                <AlertTriangle size={20} className="text-red-400 shrink-0 mt-0.5" />
                <div className="text-sm text-neutral-300 space-y-1">
                  <p className="font-medium text-red-300">Safari can't reach a local engine.</p>
                  <p className="text-neutral-400">
                    Safari blocks secure pages from talking to <span className="font-mono">localhost</span>.
                    Please open this site in <span className="text-white">Chrome, Edge, or Firefox</span>.
                  </p>
                </div>
              </div>
            )}

            {/* Disconnected setup steps */}
            {status !== 'connected' && !(remote && safari) && (
              <div className="space-y-4">
                <div className="rounded-2xl bg-neutral-950/60 border border-white/5 p-4 space-y-3">
                  <div className="text-sm font-medium text-white">Start the engine on this computer</div>

                  <ol className="text-sm text-neutral-400 space-y-2 list-decimal list-inside">
                    <li>
                      Download the engine:{' '}
                      <a
                        href={ENGINE_REPO}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:text-blue-300 inline-flex items-center gap-1"
                      >
                        <Download size={12} /> SHARP Engine
                      </a>
                    </li>
                    <li>From the project folder, run:</li>
                  </ol>

                  {/* macOS */}
                  <CommandRow
                    label="macOS (or double-click start-engine.command)"
                    cmd={macCommand}
                    copied={copied}
                    onCopy={copyCommand}
                  />
                  {/* Linux */}
                  <CommandRow
                    label="Linux"
                    cmd={shellCommand}
                    copied={copied}
                    onCopy={copyCommand}
                  />
                  <p className="text-xs text-neutral-600">
                    First run installs dependencies and downloads the model (~once). Windows: run <span className="font-mono">start-engine.bat</span>.
                  </p>
                </div>

                {/* Chrome permission explainer (remote origin only) */}
                {remote && (
                  <div className="text-xs text-neutral-500 rounded-xl bg-white/[0.02] border border-white/5 p-3">
                    Your browser may ask for <span className="text-neutral-300">"Local Network Access"</span> permission
                    so this site can talk to the engine on your machine. Click <span className="text-neutral-300">Allow</span>.
                  </div>
                )}
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-between gap-3 pt-1">
              <button
                onClick={() => setShowSettings((s) => !s)}
                className="flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
              >
                <Settings2 size={14} /> Advanced
              </button>
              <button
                onClick={() => { setStatus('checking'); probe(); }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
              >
                <RefreshCw size={14} /> Retry connection
              </button>
            </div>

            {/* Advanced: custom engine URL */}
            {showSettings && (
              <div className="rounded-2xl bg-neutral-950/60 border border-white/5 p-4 space-y-2">
                <label className="text-xs text-neutral-400">Engine URL (advanced — e.g. a different port)</label>
                <div className="flex gap-2">
                  <input
                    value={urlDraft}
                    onChange={(e) => setUrlDraft(e.target.value)}
                    placeholder={DEFAULT_API_BASE}
                    className="flex-1 px-3 py-2 rounded-lg bg-black/40 border border-white/10 text-sm text-white font-mono focus:outline-none focus:border-blue-500/50"
                  />
                  <button
                    onClick={applyUrl}
                    className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-sm text-white transition-colors"
                  >
                    Apply
                  </button>
                </div>
                {!isLocalBackend(urlDraft) && (
                  <p className="text-xs text-yellow-400/80">
                    Non-local URL — the browser's local-network hint will be skipped.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

const CommandRow = ({ label, cmd, copied, onCopy }) => (
  <div className="space-y-1">
    <div className="text-[11px] uppercase tracking-wider text-neutral-600">{label}</div>
    <div className="flex items-center gap-2 rounded-lg bg-black/50 border border-white/5 px-3 py-2">
      <code className="flex-1 text-sm text-green-300 font-mono truncate">{cmd}</code>
      <button
        onClick={() => onCopy(cmd)}
        className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-white/10 transition-colors shrink-0"
        title="Copy"
      >
        {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
      </button>
    </div>
  </div>
);
