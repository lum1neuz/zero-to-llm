// Pyodide runs here, off the main thread (module worker). Pyodide is self-hosted under
// /pyodide/<version>/. The page sends the harness and mini-torch sources once ("init"), then learner
// code + lesson tests per run. The page terminates and replaces this worker if a run takes
// too long (an infinite loop can't be interrupted otherwise).
let py = null;
let ready = null;

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    ready = (async () => {
      const base = `/pyodide/${m.version}/`;
      const { loadPyodide } = await import(`${base}pyodide.mjs`);
      py = await loadPyodide({ indexURL: base });
      await py.loadPackage("numpy", { messageCallback: () => {} });
      py.FS.mkdirTree("/course");
      py.FS.writeFile("/course/torch.py", m.torch);
      py.FS.writeFile("/course/harness.py", m.harness);
      py.runPython("import sys; sys.path.insert(0, '/course'); import harness");
      return py.runPython("import sys; sys.version.split()[0]");
    })();
    try {
      const python = await ready;
      self.postMessage({ type: "ready", python });
    } catch (err) {
      self.postMessage({ type: "init-error", message: String((err && err.message) || err) });
    }
    return;
  }
  if (m.type === "run") {
    try {
      await ready;
      py.globals.set("_code", m.code);
      py.globals.set("_tests", m.tests);
      py.globals.set("_file", m.file);
      // A "Start here" snippet runs after the learner's file instead of the lesson tests.
      if (m.snippet !== undefined) py.globals.set("_snippet", m.snippet);
      const json = py.runPython(m.snippet !== undefined ? "harness.try_json(_code, _snippet, _file)" : "harness.run_json(_code, _tests, _file)");
      self.postMessage({ type: "result", id: m.id, json });
    } catch (err) {
      self.postMessage({ type: "result", id: m.id, error: String((err && err.message) || err) });
    }
  }
};
