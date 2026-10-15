// The learner runs it themselves, then enters their numbers. They're compared against the
// reference run and the pass limit from the lesson's frontmatter.
import { useState } from "react";
import type { Lesson } from "../../lib/course.ts";
import { computeMetric, formatMetric, metricVerdict } from "../../lib/metrics.ts";
import { lessonRec, touchDay, update, useProgress } from "../../lib/progress.ts";
import { inlineMd } from "../../lib/text.ts";
import { Html, toast } from "../ui.tsx";

export function MetricsStep({ lesson, title, intro }: { lesson: Lesson; title: string; intro?: string }) {
  const p = useProgress();
  const saved = p.lessons[lesson.id]?.metrics?.values ?? {};
  const metrics = [lesson.qualityMetric, lesson.systemsMetric];
  const inputs = metrics.flatMap((m) => m.inputs ?? []).filter((x, i, all) => all.findIndex((y) => y.id === x.id) === i);
  const [raw, setRaw] = useState<Record<string, string>>(() => Object.fromEntries(inputs.map((x) => [x.id, saved[x.id] !== undefined ? String(saved[x.id]) : ""])));
  const values = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, parseFloat(v.replace(",", "."))]));
  const missing = inputs.filter((x) => !(values[x.id] > 0));

  const saveIt = () => {
    if (missing.length) { toast(`Fill in ${missing.length === 1 ? missing[0].label : `all ${inputs.length} numbers`} first.`); return; }
    update((d) => { lessonRec(d, lesson.id).metrics = { values, at: Date.now() }; touchDay(d); });
    toast("Saved. It shows under Me › Metrics too.");
  };

  return (
    <>
      <div className="stage-h"><h2>{title}</h2></div>
      {intro && <Html as="p" className="fine" html={inlineMd(intro)} />}
      <form className="mform" onSubmit={(e) => { e.preventDefault(); saveIt(); }}>
        {inputs.map((x) => (
          <label key={x.id}>{x.label}
            <input inputMode="decimal" autoComplete="off" value={raw[x.id]} placeholder="0" onChange={(e) => setRaw({ ...raw, [x.id]: e.target.value })} />
            {x.hint && <small>{x.hint}</small>}
          </label>
        ))}
      </form>
      {metrics.map((m) => <Compare key={m.id} m={m} value={computeMetric(m, values)} />)}
      <div className="row-gap">
        <button className="btn primary" onClick={saveIt}>{p.lessons[lesson.id]?.metrics ? "Update progress" : "Save to progress"}</button>
        {missing.length > 0 && <span className="fine" style={{ margin: 0 }}>Fill in {missing.map((x) => x.label).join(", ")} to save.</span>}
      </div>
    </>
  );
}

function Compare({ m, value }: { m: Lesson["qualityMetric"]; value: number | null }) {
  const limit = m.pass?.max ?? m.pass?.min;
  const ref = m.reference;
  const scale = Math.max(Math.abs(value ?? 0), Math.abs(ref ?? 0), Math.abs(limit ?? 0)) * 1.25 || 1;
  const w = (v: number) => `${Math.max(0, Math.min(100, (Math.abs(v) / scale) * 100))}%`;
  const verdict = metricVerdict(m, value);
  const limitText = m.pass?.max !== undefined ? `at most ${formatMetric(m, m.pass.max)}` : m.pass?.min !== undefined ? `at least ${formatMetric(m, m.pass.min)}` : "";
  return (
    <div className="cmp">
      <h3>{m.label}</h3>
      <div className="cmp-row"><span>You</span>
        <div className="cmp-track"><i className={verdict && !verdict.pass ? "over" : ""} style={{ width: value === null ? 0 : w(value) }} />{limit !== undefined && <span className="limit" style={{ left: w(limit) }} title={`Pass limit ${limitText}`} />}</div>
        <b>{formatMetric(m, value)}</b></div>
      {ref !== undefined && (
        <div className="cmp-row"><span>Reference</span>
          <div className="cmp-track"><i className="ref" style={{ width: w(ref) }} />{limit !== undefined && <span className="limit" style={{ left: w(limit) }} />}</div>
          <b>{formatMetric(m, ref)}</b></div>
      )}
      <div className={`verdict ${!verdict ? "wait" : verdict.pass ? "ok" : "no"}`}>
        {!verdict ? `Enter your numbers to compare. Pass limit: ${limitText}. Dashed line = limit.` : verdict.pass ? `Within the limit (${limitText}).` : `Outside the limit (${limitText}). Check for an outlier layer before moving on.`}
      </div>
      {m.note && <p className="fine" style={{ margin: 0 }}>{m.note}</p>}
    </div>
  );
}
