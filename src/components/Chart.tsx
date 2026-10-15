// Charts from ```chart fences, drawn as inline SVG at the container's real pixel width so text stays
// 11–12px on a phone and on a laptop. Colours come from the --viz-* tokens in global.css (light and dark
// steps are checked for colour-blind separation). Hover or arrow keys show values; every number is also
// in the "Show the numbers" table, so the tooltip never gates anything.
import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Chart as ChartT } from "../../pipeline/schema.ts";
import { blockMd } from "../lib/text.ts";
import { Html } from "./ui.tsx";

const fmt = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e4) return `${+(v / 1e3).toPrecision(3)}K`;
  return String(+v.toPrecision(3));
};
const fmtX = (v: number | string) => (typeof v === "number" ? fmt(v) : v);

function niceTicks(lo: number, hi: number, n: number) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw * 0.999)!;
  const out: number[] = [];
  for (let v = Math.floor(lo / step + 1e-9) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(12));
  if (out[0] > lo + step * 1e-6) out.unshift(+(out[0] - step).toPrecision(12));
  return out;
}

/** Plain-language summary for screen readers: what is plotted and where it goes. */
export function chartSummary(c: ChartT) {
  const xs = c.x.values;
  const kind = { bar: "Bar chart", line: "Line chart", scatter: "Scatter plot" }[c.kind];
  const parts = [`${kind}${c.illustrative ? " of illustrative numbers" : ""}: ${c.title}.`];
  if (xs.length <= 8) {
    for (const s of c.series) parts.push(`${s.name} by ${c.x.label}: ${xs.map((x, i) => `${fmtX(x)} → ${fmt(s.values[i])}`).join(", ")}.`);
  } else {
    parts.push(`${c.x.label} runs from ${fmtX(xs[0])} to ${fmtX(xs[xs.length - 1])}.`);
    for (const s of c.series) parts.push(`${s.name} goes from ${fmt(s.values[0])} to ${fmt(s.values[s.values.length - 1])}, lowest ${fmt(Math.min(...s.values))}, highest ${fmt(Math.max(...s.values))}.`);
  }
  if (c.threshold) parts.push(`A line marks ${c.threshold.label} at ${fmt(c.threshold.y)}.`);
  if (c.highlight) parts.push(`Marked: ${c.highlight.label}, at ${c.x.label} ${fmtX(c.highlight.x)}.`);
  return parts.join(" ");
}

const COLORS = ["var(--viz-1)", "var(--viz-2)", "var(--viz-3)"];
const CHAR = 6.4; // average width of an 11px label, for spacing ticks without measuring

function barPath(x: number, w: number, base: number, top: number, r: number) {
  const h = Math.abs(base - top);
  r = Math.min(r, w / 2, h);
  const d = top < base ? 1 : -1; // positive bars grow up
  return `M${x},${base}V${top + d * r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + d * r}V${base}Z`;
}

export function Chart({ chart: c }: { chart: ChartT }) {
  const wrap = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(320);
  const [hi, setHi] = useState<number | null>(null);
  const titleId = useId();
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const xs = c.x.values;
  const n = xs.length;
  const numericX = typeof xs[0] === "number" && c.kind !== "bar";
  const all = c.series.flatMap((s) => s.values);
  const H = W < 480 ? 210 : 250;

  // y domain: the author's range wins; otherwise nice ticks around the data (and the threshold).
  let lo = Math.min(...all, c.threshold?.y ?? Infinity), hiV = Math.max(...all, c.threshold?.y ?? -Infinity);
  if (c.kind === "bar" || (lo >= 0 && lo < hiV * 0.25)) lo = Math.min(lo, 0);
  if (c.kind === "bar") hiV = Math.max(hiV, 0);
  const plotGuess = H - 70;
  let yTicks = niceTicks(c.y?.min ?? lo, c.y?.max ?? hiV, Math.max(3, Math.round(plotGuess / 40)));
  const y0 = c.y?.min ?? yTicks[0], y1 = c.y?.max ?? yTicks[yTicks.length - 1];
  yTicks = yTicks.filter((t) => t >= y0 - 1e-12 && t <= y1 + 1e-12);

  const m = { t: c.y?.label ? 26 : 12, r: 14, b: 42, l: Math.ceil(Math.max(...yTicks.map((t) => fmt(t).length)) * CHAR) + 12 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const Y = (v: number) => m.t + ph - ((v - y0) / (y1 - y0)) * ph;
  const xMax = numericX ? Math.max(...(xs as number[])) : 1;
  let xMin = numericX ? Math.min(...(xs as number[])) : 0;
  // Start the x axis at 0 when the data nearly does (chances from 0.01, context from 8k).
  if (numericX && xMin > 0 && xMin <= (xMax - xMin) * 0.1) xMin = 0;
  const band = pw / n;
  const X = (i: number) => numericX ? m.l + (((xs[i] as number) - xMin) / (xMax - xMin || 1)) * pw
    : c.kind === "bar" ? m.l + band * (i + 0.5) : m.l + (n === 1 ? pw / 2 : (pw * i) / (n - 1));

  // x ticks: numbers get nice ticks; labels thin out when they would collide.
  const xTicks: { x: number; text: string }[] = [];
  // A few x values that don't crowd each other (k = 1, 2, 4, 8) get a tick each; otherwise nice round ticks.
  const own = numericX && n <= 8 && xs.every((v, i) => i === 0 || X(i) - X(i - 1) >= (Math.max(fmtX(v).length, fmtX(xs[i - 1]).length) * CHAR) + 8);
  if (own) xs.forEach((v, i) => xTicks.push({ x: X(i), text: fmtX(v) }));
  else if (numericX) {
    for (const t of niceTicks(xMin, xMax, Math.max(2, Math.floor(pw / 64)))) {
      if (t < xMin - 1e-12 || t > xMax + 1e-12) continue;
      xTicks.push({ x: m.l + ((t - xMin) / (xMax - xMin || 1)) * pw, text: fmt(t) });
    }
  } else {
    const widest = Math.max(...xs.map((v) => fmtX(v).length)) * CHAR + 8;
    const slot = c.kind === "bar" ? band : pw / Math.max(1, n - 1);
    const every = Math.max(1, Math.ceil(widest / slot));
    xs.forEach((v, i) => { if (i % every === 0 || (c.kind !== "bar" && i === n - 1 && (n - 1) % every >= every / 2)) xTicks.push({ x: X(i), text: fmtX(v) }); });
  }

  const k = c.series.length;
  const barW = Math.min(24, (band * 0.72 - (k - 1) * 2) / k);
  const groupW = barW * k + (k - 1) * 2;
  const base = Y(Math.max(y0, Math.min(0, y1)));
  const hl = c.highlight ? { i: xs.findIndex((v) => v === c.highlight!.x), s: Math.max(0, c.series.findIndex((s) => s.name === c.highlight!.series)) } : null;
  const dots = c.kind === "scatter" || n <= 12;

  const pick = (e: PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W, py = ((e.clientY - r.top) / r.height) * H;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < n; i++) {
      const d = c.kind === "scatter" ? Math.min(...c.series.map((s) => Math.hypot(X(i) - px, Y(s.values[i]) - py))) : Math.abs(X(i) - px);
      if (d < bestD) { bestD = d; best = i; }
    }
    setHi(best);
  };
  const keys = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      setHi((h) => Math.max(0, Math.min(n - 1, h === null ? 0 : h + (e.key === "ArrowRight" ? 1 : -1))));
    } else if (e.key === "Escape") setHi(null);
  };
  // The tooltip sits beside the hovered point (or bar), on whichever side it fits, so it never covers it.
  const tipGap = c.kind === "bar" ? groupW / 2 + 10 : 14;
  useLayoutEffect(() => {
    const el = tip.current;
    if (!el || hi === null) return;
    const w = el.offsetWidth, x = X(hi);
    const left = x + tipGap + w <= W ? x + tipGap : x - tipGap - w >= 0 ? x - tipGap - w : Math.max(0, W - w);
    el.style.left = `${left}px`;
  });

  return (
    <figure className="chart" aria-labelledby={titleId}>
      <div className="chart-h">
        <div className="chart-title" id={titleId}>{c.title}</div>
        {c.illustrative && <span className="chip" title="Made-up numbers that show the shape, not a measurement">Illustrative</span>}
      </div>
      {k > 1 && (
        <ul className="chart-legend">
          {c.series.map((s, i) => <li key={s.name}><i className={`key key-${c.kind}`} style={{ background: COLORS[i] }} aria-hidden="true" />{s.name}</li>)}
        </ul>
      )}
      <div className="chart-plot" ref={wrap}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={chartSummary(c)} tabIndex={0}
          onKeyDown={keys} onBlur={() => setHi(null)}>
          {c.y?.label && <text className="chart-axis" x={m.l - 4} y={12} textAnchor="start">↑ {c.y.label}</text>}
          {yTicks.map((t) => (
            <g key={t}>
              <line className="chart-grid" x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} />
              <text className="chart-tick" x={m.l - 6} y={Y(t)} dy="0.32em" textAnchor="end">{fmt(t)}</text>
            </g>
          ))}
          <line className="chart-base" x1={m.l} x2={W - m.r} y1={base} y2={base} />
          {xTicks.map((t, i) => <text key={i} className="chart-tick" x={t.x} y={m.t + ph + 16} textAnchor="middle">{t.text}</text>)}
          <text className="chart-axis" x={m.l + pw / 2} y={H - 6} textAnchor="middle">{c.x.label} →</text>

          {c.threshold && (
            <g>
              <line className="chart-limit" x1={m.l} x2={W - m.r} y1={Y(c.threshold.y)} y2={Y(c.threshold.y)} />
              <text className="chart-note" x={W - m.r} y={Y(c.threshold.y) - 5} textAnchor="end">{c.threshold.label}</text>
            </g>
          )}

          {c.kind === "bar" && c.series.map((s, si) => s.values.map((v, i) => (
            <path key={`${si}-${i}`} d={barPath(X(i) - groupW / 2 + si * (barW + 2), barW, base, Y(v), 4)} fill={COLORS[si]}
              opacity={hl && k === 1 && hl.i !== i ? 0.45 : 1} />
          )))}
          {c.kind === "line" && c.series.map((s, si) => (
            <path key={si} className="chart-line" stroke={COLORS[si]} d={s.values.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join("")} />
          ))}
          {hi !== null && c.kind !== "bar" && <line className="chart-cross" x1={X(hi)} x2={X(hi)} y1={m.t} y2={m.t + ph} />}
          {c.kind !== "bar" && dots && c.series.map((s, si) => s.values.map((v, i) => (
            <circle key={`${si}-${i}`} className="chart-dot" cx={X(i)} cy={Y(v)} r={hi === i ? 5.5 : 4} fill={COLORS[si]} />
          )))}
          {c.kind !== "bar" && !dots && hi !== null && c.series.map((s, si) => (
            <circle key={si} className="chart-dot" cx={X(hi)} cy={Y(s.values[hi])} r={5.5} fill={COLORS[si]} />
          ))}
          {hl && hl.i >= 0 && (() => {
            const v = c.series[hl.s].values[hl.i];
            const cx = c.kind === "bar" ? X(hl.i) - groupW / 2 + hl.s * (barW + 2) + barW / 2 : X(hl.i), cy = Y(v);
            // Try spots around the point and take the first whose text box stays in the plot and clear of every line
            // (or, when none is, the one that crosses the lines least).
            const tw = c.highlight!.label.length * 6 + 4, th = 14;
            const spots: [number, number, "start" | "middle" | "end"][] = c.kind === "bar" ? [[cx, cy - 8, "middle"], [cx, cy + 18, "middle"]]
              : [[cx, cy - 12, "middle"], [cx + 10, cy - 10, "start"], [cx - 10, cy - 10, "end"], [cx, cy + 20, "middle"], [cx + 10, cy + 18, "start"], [cx - 10, cy + 18, "end"]];
            const box = ([x, y, a]: (typeof spots)[number]) => { const l = a === "start" ? x : a === "end" ? x - tw : x - tw / 2; return { l, r: l + tw, t: y - th + 3, b: y + 3 }; };
            const hits = (b: ReturnType<typeof box>) => {
              let n = b.l >= m.l - 4 && b.r <= W - 2 && b.t >= m.t - 12 && b.b <= m.t + ph ? 0 : 1000;
              if (c.kind === "line") for (const s of c.series) for (let i = 1; i < s.values.length; i++) {
                const x1 = X(i - 1), y1 = Y(s.values[i - 1]), x2 = X(i), y2 = Y(s.values[i]);
                const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1) / 3);
                for (let t = 0; t <= steps; t++) {
                  const px = x1 + ((x2 - x1) * t) / steps, py = y1 + ((y2 - y1) * t) / steps;
                  if (px > b.l - 3 && px < b.r + 3 && py > b.t - 3 && py < b.b + 3) n++;
                }
              }
              return n;
            };
            const scored = spots.map((sp) => ({ sp, n: hits(box(sp)) }));
            const [lx, ly, anchor] = (scored.find((x) => x.n === 0) ?? scored.reduce((a, b) => (b.n < a.n ? b : a))).sp;
            return (
              <g>
                {c.kind !== "bar" && <circle className="chart-dot hl" cx={cx} cy={cy} r={6} fill={COLORS[hl.s]} />}
                <text className="chart-note strong" x={lx} y={ly} textAnchor={anchor}>{c.highlight!.label}</text>
              </g>
            );
          })()}
          {c.kind === "bar" && hi !== null && <rect className="chart-band" x={X(hi) - band / 2} y={m.t} width={band} height={ph} />}
          <rect x={m.l} y={m.t} width={pw} height={ph} fill="transparent" onPointerMove={pick} onPointerDown={pick} onPointerLeave={() => setHi(null)} />
        </svg>
        {hi !== null && (
          <div className="chart-tip" ref={tip} style={{ top: m.t }} aria-live="polite">
            <b>{c.x.label} {fmtX(xs[hi])}</b>
            {c.series.map((s, si) => <span key={s.name}>{k > 1 && <i className={`key key-${c.kind}`} style={{ background: COLORS[si] }} aria-hidden="true" />}{s.name}: {fmt(s.values[hi])}</span>)}
          </div>
        )}
      </div>
      <figcaption className="fine">{c.caption}{c.illustrative && " (Illustrative numbers.)"}</figcaption>
      <details className="chart-data">
        <summary>Show the numbers</summary>
        <div className="chart-table">
          <table>
            <thead><tr><th scope="col">{c.x.label}</th>{c.series.map((s) => <th scope="col" key={s.name}>{s.name}</th>)}</tr></thead>
            <tbody>{xs.map((x, i) => <tr key={i}><th scope="row">{fmtX(x)}</th>{c.series.map((s) => <td key={s.name}>{fmt(s.values[i])}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

/** Authored Markdown with its charts dropped in where the `<!-- chart:N -->` markers sit. */
export function MdCharts({ md, charts, className }: { md: string; charts?: ChartT[]; className?: string }) {
  if (!charts?.length) return <Html as="div" className={className} html={blockMd(md)} />;
  const parts = md.split(/^<!-- chart:(\d+) -->$/m);
  return (
    <div className={className}>
      {parts.map((p, i) => i % 2 ? (charts[Number(p)] ? <Chart key={i} chart={charts[Number(p)]} /> : null)
        : p.trim() ? <Html key={i} as="div" html={blockMd(p)} /> : null)}
    </div>
  );
}
