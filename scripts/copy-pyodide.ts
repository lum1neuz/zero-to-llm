// Self-host the Pyodide core + NumPy under public/pyodide/<version>/ so the browser loads Python from
// the same origin (works offline once cached, and in contexts that block cross-origin workers).
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const src = join(root, "node_modules/pyodide");
const version = JSON.parse(readFileSync(join(src, "package.json"), "utf8")).version as string;
// Versioned path: the service worker caches it forever, so an upgrade must change the URL.
const out = join(root, "public/pyodide", version);
if (!existsSync(join(out, "pyodide.mjs"))) rmSync(join(root, "public/pyodide"), { recursive: true, force: true });
const lock = JSON.parse(readFileSync(join(src, "pyodide-lock.json"), "utf8"));
const NEED = ["numpy"];

mkdirSync(out, { recursive: true });
for (const f of ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) {
  copyFileSync(join(src, f), join(out, f));
}
for (const name of NEED) {
  const pkg = lock.packages[name];
  const deps: string[] = pkg.depends ?? [];
  for (const n of [name, ...deps]) {
    const file = lock.packages[n].file_name as string;
    const dest = join(out, file);
    if (existsSync(dest)) continue;
    const cached = join(src, file);
    if (existsSync(cached)) { copyFileSync(cached, dest); continue; }
    const url = `https://cdn.jsdelivr.net/pyodide/v${version}/full/${file}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Couldn't download ${url}: ${res.status}`);
    writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  }
}
console.log(`✓ Pyodide ${version} + ${NEED.join(", ")} in public/pyodide/${version}/`);
