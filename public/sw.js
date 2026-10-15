// Offline support. The build replaces VERSION, PRECACHE and PYODIDE (scripts/sw-manifest.ts).
// Pages and assets are precached; Pyodide and fonts are cached the first time they load.
const VERSION = "dev";
const PRECACHE = [];
const PYODIDE = "dev";
const CACHE = `z2l-${VERSION}`;
const RUNTIME = "z2l-runtime-v1";
// Pages you've opened (Build pages aren't precached). Kept across deploys so they still open offline;
// online, navigation is network-first, so they refresh on the next visit.
const PAGES = "z2l-pages-v1";

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE && k !== RUNTIME && k !== PAGES).map((k) => caches.delete(k))))
      .then(() => caches.open(RUNTIME))
      .then(async (c) => {
        // drop Pyodide files from older versions (paths are /pyodide/<version>/...)
        for (const req of await c.keys()) {
          const p = new URL(req.url).pathname;
          if (p.startsWith("/pyodide/") && PYODIDE !== "dev" && !p.startsWith(`/pyodide/${PYODIDE}/`)) await c.delete(req);
        }
      })
      .then(() => self.clients.claim()),
  );
});

const isRuntime = (url) =>
  url.pathname.startsWith("/pyodide/") || url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com";

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (isRuntime(url)) {
    // cache-first: these files are versioned or immutable
    e.respondWith(caches.open(RUNTIME).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok || res.type === "opaque") c.put(req, res.clone());
      return res;
    }));
    return;
  }
  if (url.origin !== self.location.origin) return;
  if (req.mode === "navigate") {
    // network-first for pages so content updates show, cached copy when offline
    e.respondWith(fetch(req).then((res) => {
      const copy = res.clone();
      if (res.ok) caches.open(PAGES).then((c) => c.put(req, copy));
      return res;
    }).catch(async () => (await caches.match(req, { ignoreSearch: true })) || (await caches.match(url.pathname.replace(/\/?$/, "/"))) || caches.match("/")));
    return;
  }
  e.respondWith(caches.match(req).then((hit) => hit || fetch(req)));
});
