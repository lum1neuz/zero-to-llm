// Fold reader: one text at four depths. The level control folds the whole document;
// tapping a section (or its key sentence) unfolds just that section one level deeper,
// so a key always leads back to the text it summarises.
import { useEffect, useState } from "react";
import type { FoldDoc, FoldSection } from "../../../pipeline/schema.ts";
import { inlineMd } from "../../lib/text.ts";
import { MdCharts } from "../Chart.tsx";
import { Html, I } from "../ui.tsx";

export const LEVELS = ["Gist", "Map", "Keys", "Full"] as const;
export type Level = 0 | 1 | 2 | 3;

const words = (s?: string) => (s ? s.split(/\s+/).filter(Boolean).length : 0);
const storeKey = (scope: string, doc: string) => `z2l.fold.${scope}.${doc}`;

function readLevel(key: string, fallback: Level): Level {
  try { const v = Number(localStorage.getItem(key)); return v >= 0 && v <= 3 && localStorage.getItem(key) !== null ? (v as Level) : fallback; } catch { return fallback; }
}

export function FoldReader({ doc, scope, learnDone, learnHref }: { doc: FoldDoc; scope: string; learnDone: boolean; learnHref: string }) {
  const key = storeKey(scope, doc.id);
  const startLevel: Level = doc.kind === "primer" ? 3 : 1;
  const [level, setLevelState] = useState<Level>(() => readLevel(key, startLevel));
  const [open, setOpen] = useState<Record<string, Level>>({});
  useEffect(() => { setLevelState(readLevel(key, startLevel)); setOpen({}); }, [key, startLevel]);

  const setLevel = (l: Level) => {
    setLevelState(l);
    setOpen({});
    try { localStorage.setItem(key, String(l)); } catch { /* per-device convenience only */ }
  };
  // A section's own depth overrides the document level in either direction, so a fully
  // unfolded section can fold back even when the whole document is at Full.
  const depth = (s: FoldSection) => (open[s.id] ?? level) as Level;
  const deepen = (s: FoldSection, to?: Level) => {
    const cur = depth(s);
    const next = (to ?? (cur >= 3 ? 1 : cur + 1)) as Level;
    setOpen({ ...open, [s.id]: next });
  };
  const counts = [words(doc.gist), doc.sections.reduce((a, s) => a + words(s.title), 0),
    doc.sections.reduce((a, s) => a + words(s.title) + words(s.key), 0),
    doc.sections.reduce((a, s) => a + words(s.title) + words(s.key) + words(s.body), 0)];

  return (
    <article className="fold" aria-label={doc.title}>
      <header className="fold-h">
        <div className="eyebrow" style={{ margin: 0 }}>{doc.kind === "paper" ? "Paper notes" : doc.kind === "primer" ? `Primer${doc.minutes ? ` · about ${doc.minutes} min` : ""}` : "Lesson at a glance"}</div>
        <h2>{doc.title}</h2>
        {doc.source && (
          <p className="fine" style={{ margin: 0 }}>
            {doc.source.citation}{doc.source.url && <> · <a href={doc.source.url} target="_blank" rel="noopener">Original ↗</a></>}
            {doc.kind === "paper" && <> · Notes written for this course, not the paper's text.</>}
          </p>
        )}
        <div className="fold-levels seg" role="group" aria-label="Fold level">
          {LEVELS.map((name, i) => (
            <button key={name} aria-pressed={level === i} onClick={() => setLevel(i as Level)} title={`${counts[i]} words`}>
              {name}<small>{counts[i] < 1000 ? counts[i] : `${(counts[i] / 1000).toFixed(1)}k`}</small>
            </button>
          ))}
        </div>
      </header>

      <p className="fold-gist">{doc.gist}</p>

      {level === 0 && (
        <button className="btn sm" onClick={() => setLevel(1)}>Unfold to the map · {doc.sections.length} sections</button>
      )}
      {level >= 1 && (
        <ol className="fold-list">
          {doc.sections.map((s, i) => {
            const d = depth(s);
            const locked = s.locked === "learn" && !learnDone;
            return (
              <li key={s.id} className={`fold-sec d${d}`} id={`${doc.id}-${s.id}`}>
                <button className="fold-title" onClick={() => deepen(s)} aria-expanded={d >= 2}
                  aria-label={`${s.title}. ${d < 3 ? "Unfold" : "Fold back"}`}>
                  <span className="fold-n">{i + 1}</span>
                  <span className="fold-t">{s.title}</span>
                  {s.ref && <span className="chip">{s.ref}</span>}
                  <span className="fold-more" aria-hidden="true">{d < 3 ? "+" : "−"}</span>
                </button>
                {d >= 2 && s.key && (
                  <button className="fold-key" onClick={() => deepen(s, d >= 3 ? 2 : 3)} title={d >= 3 ? "Fold back to the key" : "Unfold the text this key comes from"}>
                    <Html html={inlineMd(s.key)} />
                  </button>
                )}
                {d >= 3 && (locked ? (
                  <p className="fold-locked">{I.lock} The explanation gives the answer away. It opens after you finish <a href={learnHref}>Learn</a>.</p>
                ) : s.body ? (
                  <MdCharts className="fold-body md" md={s.body} charts={s.charts} />
                ) : null)}
              </li>
            );
          })}
        </ol>
      )}
    </article>
  );
}
