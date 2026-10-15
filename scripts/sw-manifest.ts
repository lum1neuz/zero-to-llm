// After `astro build`: fill dist/sw.js with the list of files to precache and a version hash.
// Pyodide (~16 MB) isn't precached; the service worker caches it on the first test run.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const files: string[] = [];
const walk = (d: string) => {
  for (const name of readdirSync(d)) {
    const p = join(d, name);
    if (statSync(p).isDirectory()) walk(p);
    else files.push(relative(dist, p).split(sep).join("/"));
  }
};
walk(dist);
const precache = files
  // Build pages are large and desktop-only; the service worker caches each one when it's first opened.
  .filter((f) => !f.startsWith("pyodide/") && !f.startsWith("build/") && f !== "sw.js" && f !== "_headers")
  .map((f) => "/" + f.replace(/(^|\/)index\.html$/, "$1"));
const hash = createHash("sha256");
for (const f of files.filter((x) => !x.startsWith("pyodide/")).sort()) hash.update(f).update(readFileSync(join(dist, f)));
const version = hash.digest("hex").slice(0, 12);
const swPath = join(dist, "sw.js");
const sw = readFileSync(swPath, "utf8")
  .replace('const VERSION = "dev";', `const VERSION = "${version}";`)
  .replace("const PRECACHE = [];", `const PRECACHE = ${JSON.stringify(precache)};`)
  .replace('const PYODIDE = "dev";', `const PYODIDE = "${JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).devDependencies.pyodide}";`);
writeFileSync(swPath, sw);
console.log(`✓ service worker: ${precache.length} files precached, version ${version}`);
