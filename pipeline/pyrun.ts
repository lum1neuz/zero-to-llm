// Runs learner code + lesson tests through python/harness.py in Pyodide under Node.
// Same harness and shim the browser worker uses, so a passing check here means the same in the app.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadPyodide, type PyodideInterface } from "pyodide";

const root = fileURLToPath(new URL("..", import.meta.url));
let py: Promise<PyodideInterface> | null = null;

export function runtimeSources() {
  return {
    torch: readFileSync(`${root}python/torch.py`, "utf8"),
    harness: readFileSync(`${root}python/harness.py`, "utf8"),
  };
}

async function boot() {
  const p = await loadPyodide();
  await p.loadPackage("numpy", { messageCallback: () => {} });
  const src = runtimeSources();
  p.FS.mkdirTree("/course");
  p.FS.writeFile("/course/torch.py", src.torch);
  p.FS.writeFile("/course/harness.py", src.harness);
  p.runPython("import sys; sys.path.insert(0, '/course'); import harness");
  return p;
}

export interface RunResult {
  stdout: string;
  scriptError: null | { type: string; message: string; line: number | null; diagnosis: string; plain: boolean; traceback: string };
  tests: { id: string; name: string; status: "pass" | "fail" | "skip" | "error"; message: string }[];
  warnings: string[];
}

export async function runChallenge(code: string, tests: string, filename: string): Promise<RunResult> {
  py ??= boot();
  const p = await py;
  p.globals.set("_code", code);
  p.globals.set("_tests", tests);
  p.globals.set("_file", filename);
  return JSON.parse(p.runPython("harness.run_json(_code, _tests, _file)"));
}

/** Same as the "Start here" Run button: the learner's file quietly, then the snippet. */
export async function trySnippet(code: string, snippet: string, filename: string): Promise<RunResult> {
  py ??= boot();
  const p = await py;
  p.globals.set("_code", code);
  p.globals.set("_snippet", snippet);
  p.globals.set("_file", filename);
  return JSON.parse(p.runPython("harness.try_json(_code, _snippet, _file)"));
}
