// Charts in reading text: a ```chart fence holding YAML. The pipeline checks each one and moves its data
// into a `charts` array beside the Markdown, leaving a `<!-- chart:N -->` line where the fence was.
// course.json carries chart data, never drawings, so any client can render it its own way; a plain
// Markdown renderer just skips the comment.
//
//   ```chart
//   kind: line                     # bar | line | scatter
//   title: Surprise against chance
//   x: { label: chance p, values: [0.05, 0.1, 0.25, 0.5, 1] }   # numbers, or text labels (bar, line)
//   y: { label: surprise (nats), min: 0, max: 3 }              # optional; label, min and max all optional
//   series:
//     - { name: −log p, values: [3.0, 2.3, 1.39, 0.69, 0] }
//   highlight: { x: 0.5, label: a coin flip }                   # optional point (series defaults to the first)
//   threshold: { y: 1, label: pass limit }                      # optional horizontal line
//   illustrative: true                                          # optional: made-up numbers that show a shape
//   caption: Rare things are surprising; certain things are not.
//   ```
import { parse as parseYaml } from "yaml";
import { z } from "zod";

export const CHART_KINDS = ["bar", "line", "scatter"] as const;
/** At most three series: more than three hues stop being reliably distinct for colour-blind readers. */
export const MAX_SERIES = 3;
export const chartMarker = (i: number) => `<!-- chart:${i} -->`;
export const CHART_MARKER = /^<!-- chart:(\d+) -->$/;

const Num = z.number().refine(Number.isFinite, "must be a finite number");
const Text = z.string().trim().min(1);
export const ChartSrc = z.strictObject({
  kind: z.enum(CHART_KINDS, { error: (i) => `kind must be one of ${CHART_KINDS.join(", ")}, got ${JSON.stringify(i.input)}` }),
  title: Text,
  x: z.strictObject({ label: Text, values: z.union([z.array(Num).min(2), z.array(Text).min(2)], { error: "two or more finite numbers, or two or more text labels (not a mix)" }) }),
  y: z.strictObject({ label: Text.optional(), min: Num.optional(), max: Num.optional() }).optional(),
  series: z.array(z.strictObject({ name: Text, values: z.array(Num) })).min(1).max(MAX_SERIES),
  highlight: z.strictObject({ x: z.union([Num, Text]), series: Text.optional(), label: Text }).optional(),
  threshold: z.strictObject({ y: Num, label: Text }).optional(),
  illustrative: z.boolean().default(false),
  caption: Text,
});

export type ChartIssue = { line?: number; message: string };

/** Check one chart's YAML. Returns the chart (camelCase, ready for course.json) or the problems. */
export function checkChart(yamlSrc: string): { chart?: z.infer<typeof ChartSrc>; issues: string[] } {
  let raw: unknown;
  try { raw = parseYaml(yamlSrc); } catch (e) { return { issues: [`YAML: ${(e as Error).message.split("\n")[0]}`] }; }
  const res = ChartSrc.safeParse(raw);
  if (!res.success) return { issues: res.error.issues.map((i) => `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`) };
  const c = res.data;
  const issues: string[] = [];
  const xs = c.x.values;
  const numericX = typeof xs[0] === "number";
  if (c.kind === "scatter" && !numericX) issues.push("x.values: a scatter chart needs numbers on x");
  if (c.kind === "line" && numericX && !(xs as number[]).every((v, i) => i === 0 || v > (xs as number[])[i - 1])) issues.push("x.values: a line chart's numbers must go up from left to right");
  if (c.kind !== "scatter" && new Set<unknown>(xs).size !== xs.length) issues.push("x.values: each x appears once");
  const names = new Set<string>();
  for (const s of c.series) {
    if (names.has(s.name)) issues.push(`series "${s.name}" appears twice`);
    names.add(s.name);
    if (s.values.length !== xs.length) issues.push(`series "${s.name}" has ${s.values.length} values but x has ${xs.length}`);
  }
  const all = c.series.flatMap((s) => s.values);
  const { min, max } = c.y ?? {};
  if (min !== undefined && max !== undefined && !(min < max)) issues.push(`y: min (${min}) must be below max (${max})`);
  if (min !== undefined && all.some((v) => v < min)) issues.push(`y.min is ${min} but a value is ${Math.min(...all)}; it would fall off the chart`);
  if (max !== undefined && all.some((v) => v > max)) issues.push(`y.max is ${max} but a value is ${Math.max(...all)}; it would fall off the chart`);
  // Bars are read by their length, so they must start at zero.
  if (c.kind === "bar" && ((min !== undefined && min > 0) || (max !== undefined && max < 0))) issues.push("y: a bar chart's range must include 0, because bars are read by their length");
  if (c.threshold) {
    if (min !== undefined && c.threshold.y < min) issues.push(`threshold.y (${c.threshold.y}) is below y.min (${min})`);
    if (max !== undefined && c.threshold.y > max) issues.push(`threshold.y (${c.threshold.y}) is above y.max (${max})`);
  }
  if (c.highlight) {
    if (!xs.some((v) => v === c.highlight!.x)) issues.push(`highlight.x ${JSON.stringify(c.highlight.x)} isn't one of x.values`);
    if (c.highlight.series !== undefined && !names.has(c.highlight.series)) issues.push(`highlight.series "${c.highlight.series}" isn't one of the series`);
  }
  const cap = c.caption.trim();
  if (!/[.!?]$/.test(cap)) issues.push("caption: one full sentence, ending in . ! or ?");
  else if (/[.!?]\s+[A-Z]/.test(cap)) issues.push("caption: keep it to one sentence");
  return issues.length ? { issues } : { chart: c, issues };
}
export type Chart = z.infer<typeof ChartSrc>;

/**
 * Pull every ```chart fence out of a Markdown string. Fences nested inside other fences (a code sample
 * showing chart syntax) are left alone. `line` is the file line of the Markdown's first line, when known.
 */
export function extractCharts(md: string, line?: number): { body: string; charts: Chart[]; issues: ChartIssue[] } {
  const lines = md.split("\n");
  const out: string[] = [];
  const charts: Chart[] = [];
  const issues: ChartIssue[] = [];
  let open: { char: string; len: number; chart: boolean; at: number; body: string[] } | null = null;
  const at = (i: number) => (line === undefined ? undefined : line + i);
  lines.forEach((l, i) => {
    const m = /^(\s{0,3})(`{3,}|~{3,})\s*([\w-]*)\s*$/.exec(l);
    if (!open) {
      if (m && m[3] === "chart") { open = { char: m[2][0], len: m[2].length, chart: true, at: i, body: [] }; return; }
      if (m) open = { char: m[2][0], len: m[2].length, chart: false, at: i, body: [] };
      out.push(l);
      return;
    }
    const closes = m && m[2][0] === open.char && m[2].length >= open.len && m[3] === "";
    if (!open.chart) { out.push(l); if (closes) open = null; return; }
    if (!closes) { open.body.push(l); return; }
    const { chart, issues: problems } = checkChart(open.body.join("\n"));
    problems.forEach((p) => issues.push({ line: at(open!.at), message: `chart: ${p}` }));
    if (chart) { out.push(chartMarker(charts.length)); charts.push(chart); }
    open = null;
  });
  const left = open as { chart: boolean; at: number } | null;
  if (left?.chart) issues.push({ line: at(left.at), message: "chart: unclosed ```chart fence" });
  return { body: out.join("\n"), charts, issues };
}
