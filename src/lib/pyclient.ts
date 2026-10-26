// Main-thread side of the Pyodide worker: boot once, run tests one at a time with a timeout,
// and replace the worker when learner code never finishes.
import type { RunResult } from "../../pipeline/pyrun.ts";

export type { RunResult };
export type RunOutcome =
  | { kind: "ok"; result: RunResult }
  | { kind: "timeout"; seconds: number }
  | { kind: "error"; message: string };

export interface Runtime { pyodide: string; hash: string }

const RUN_TIMEOUT_MS = 5000;
const BOOT_TIMEOUT_MS = 90_000;
let worker: Worker | null = null;
let booting: Promise<string> | null = null;
let runtime: Runtime | null = null;
let seq = 0;
let queue: Promise<unknown> = Promise.resolve();
const pending = new Map<number, (m: { json?: string; error?: string }) => void>();
export let pythonVersion: string | null = null;

function retire(w: Worker) {
  w.terminate();
  if (worker === w) { worker = null; booting = null; pythonVersion = null; }
}

async function fetchText(url: string) {
  const r = await fetch(url);
  if (r.status === 404) throw new Error("the course was updated since this page loaded. Reload the page to get the new version");
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.text();
}

function boot(rt: Runtime) {
  runtime = rt;
  const w = new Worker("/py-worker.js", { type: "module" });
  worker = w;
  // mini-torch and the harness are static files (cached by the service worker), not page props.
  const sources = Promise.all([fetchText(`/runtime/${rt.hash}/torch.py`), fetchText(`/runtime/${rt.hash}/harness.py`)]);
  const p = new Promise<string>((resolve, reject) => {
    sources.catch(reject);
    const timer = setTimeout(() => reject(new Error("loading took longer than 90 seconds")), BOOT_TIMEOUT_MS);
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") { clearTimeout(timer); if (worker === w) pythonVersion = m.python; resolve(m.python); }
      else if (m.type === "init-error") { clearTimeout(timer); reject(new Error(m.message)); }
      else if (m.type === "result") { pending.get(m.id)?.(m); pending.delete(m.id); }
    };
    w.onerror = (e) => { clearTimeout(timer); reject(new Error(e.message || "The Python worker failed to start")); };
  });
  booting = p;
  p.catch(() => retire(w));
  sources.then(([torch, harness]) => w.postMessage({ type: "init", version: rt.pyodide, torch, harness })).catch(() => {});
  return p;
}

export const isReady = () => pythonVersion !== null && worker !== null;

export function ensurePython(rt: Runtime) {
  return booting ?? boot(rt);
}

/** Runs are queued, so a run's timeout only counts its own execution time. */
export function runTests(rt: Runtime, code: string, tests: string, file: string, snippet?: string): Promise<RunOutcome> {
  const job = queue.then(() => runNow(rt, code, tests, file, snippet));
  queue = job.catch(() => {});
  return job;
}

/** Runs the learner's file quietly, then `snippet`; the result holds only the snippet's output. */
export const runSnippet = (rt: Runtime, code: string, snippet: string, file: string) => runTests(rt, code, "", file, snippet);

async function runNow(rt: Runtime, code: string, tests: string, file: string, snippet?: string): Promise<RunOutcome> {
  try {
    await ensurePython(rt);
  } catch (e) {
    return { kind: "error", message: `Couldn't load the Python runtime (${(e as Error).message}). Check your connection: the first run loads Pyodide, then it is cached for offline use.` };
  }
  const w = worker;
  if (!w) return { kind: "error", message: "The Python runtime restarted. Run the tests again." };
  const id = ++seq;
  return new Promise<RunOutcome>((resolve) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      retire(w);
      if (runtime && !worker) boot(runtime).catch(() => {});
      resolve({ kind: "timeout", seconds: RUN_TIMEOUT_MS / 1000 });
    }, RUN_TIMEOUT_MS);
    pending.set(id, (m) => {
      clearTimeout(timer);
      if (m.error) resolve({ kind: "error", message: m.error });
      else resolve({ kind: "ok", result: JSON.parse(m.json!) });
    });
    w.postMessage({ type: "run", id, code, tests, file, snippet });
  });
}
