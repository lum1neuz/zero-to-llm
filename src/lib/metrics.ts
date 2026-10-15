import type { MetricDef } from "../../pipeline/schema.ts";

/** value: the one input. pct_change: (b / a − 1) × 100 with the baseline first. ratio: b / a. */
export function computeMetric(m: MetricDef, values: Record<string, number>): number | null {
  if (!m.inputs || !m.compute) return null;
  const v = m.inputs.map((i) => values[i.id]);
  if (v.some((x) => !(typeof x === "number" && Number.isFinite(x)))) return null;
  if (m.compute === "value") return v[0];
  if (!(v[0] > 0)) return null;
  return m.compute === "pct_change" ? (v[1] / v[0] - 1) * 100 : v[1] / v[0];
}

export function metricVerdict(m: MetricDef, value: number | null) {
  if (value === null || !m.pass) return null;
  const pass = (m.pass.max === undefined || value <= m.pass.max) && (m.pass.min === undefined || value >= m.pass.min);
  return { pass };
}

export function formatMetric(m: MetricDef, v: number | null) {
  if (v === null) return "–";
  const sign = m.compute === "pct_change" && v >= 0 ? "+" : "";
  return m.unit === "%" ? `${sign}${v.toFixed(2)}%` : `${sign}${+v.toFixed(2)} ${m.unit}`;
}
