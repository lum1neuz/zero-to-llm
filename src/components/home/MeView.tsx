import { useRef, useState } from "react";
import type { LiteCourse, LiteLesson } from "../../lib/course.ts";
import { computeMetric, formatMetric, metricVerdict } from "../../lib/metrics.ts";
import {
  exportJson, getProgress, hoursLogged, lessonState, merge, parseImport, replaceAll, resetAll, streak, totalXp, update, useProgress,
} from "../../lib/progress.ts";
import { toast } from "../ui.tsx";

type Theme = "system" | "light" | "dark";
const readTheme = (): Theme => { try { const t = localStorage.getItem("z2l.theme"); return t === "light" || t === "dark" ? t : "system"; } catch { return "system"; } };

export function MeView({ course }: { course: LiteCourse }) {
  const p = useProgress();
  const file = useRef<HTMLInputElement>(null);
  const [theme, setThemeState] = useState<Theme>(readTheme);
  const [confirmReset, setConfirmReset] = useState(false);
  const core = course.lessons.filter((l) => !course.phases.find((ph) => ph.id === l.phase)!.optional);
  const done = core.filter((l) => lessonState(l, p) === "done").length;
  const withMetrics = course.lessons.filter((l) => p.lessons[l.id]?.metrics);

  const setTheme = (t: Theme) => {
    setThemeState(t);
    try { t === "system" ? localStorage.removeItem("z2l.theme") : localStorage.setItem("z2l.theme", t); } catch { /* ignore */ }
    if (t === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
  };
  const download = () => {
    const blob = new Blob([exportJson(getProgress())], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `zero-to-llm-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast("Progress saved as a file. Open it on your other device under Me › Import.");
  };
  const copy = () => navigator.clipboard?.writeText(exportJson(getProgress())).then(() => toast("Progress JSON copied."), () => toast("Copying isn't allowed here. Use Download instead."));
  const onImport = async (f: File | undefined) => {
    if (!f) return;
    try {
      const incoming = parseImport(await f.text());
      const before = totalXp(getProgress());
      replaceAll(merge(getProgress(), incoming));
      toast(`Merged. XP ${before.toLocaleString()} → ${totalXp(getProgress()).toLocaleString()}.`);
    } catch (e) {
      toast(e instanceof SyntaxError ? "That file isn't valid JSON." : (e as Error).message);
    }
    if (file.current) file.current.value = "";
  };

  return (
    <>
      <div className="metric-card">
        <div>Lessons done<b>{done} / {core.length}</b></div>
        <div>Hours logged<b>{hoursLogged(course, p)} / {course.totals.coreHours} h</b></div>
        <div>Experience<b>{totalXp(p).toLocaleString()} XP</b></div>
        <div>Day streak<b>{streak(p)}</b></div>
      </div>

      <div className="sec-h">Metrics</div>
      {withMetrics.length ? withMetrics.map((l) => <MetricRow key={l.id} lesson={l} values={p.lessons[l.id].metrics!.values} />)
        : <div className="empty">No metrics yet. Record them in a lesson's Build steps.</div>}

      <div className="sec-h">Sync between devices</div>
      <p className="fine">No account and no server. Download your progress here and import it on the other device. Imports merge: Learn work from your phone and Build work from your laptop both survive.</p>
      <div className="row-gap" style={{ marginTop: 10 }}>
        <button className="btn" onClick={download}>Download progress</button>
        <button className="btn" onClick={copy}>Copy as JSON</button>
        <button className="btn" onClick={() => file.current?.click()}>Import…</button>
        <input ref={file} type="file" accept="application/json,.json" hidden onChange={(e) => onImport(e.target.files?.[0])} />
      </div>

      <div className="sec-h">Settings</div>
      <div style={{ display: "grid", gap: 14 }}>
        <div className="row-gap"><span className="field" style={{ minWidth: 80 }}>Theme</span>
          <div className="seg" role="group" aria-label="Theme">
            {(["system", "light", "dark"] as Theme[]).map((t) => <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}
          </div>
        </div>
        <label className="field">Course start date (week 1)
          <input type="date" value={p.startDate} max={new Date().toISOString().slice(0, 10)}
            onChange={(e) => e.target.value && update((d) => { d.startDate = e.target.value; d.startSetAt = Date.now(); })} />
        </label>
        <div>
          {confirmReset ? (
            <div className="row-gap">
              <span className="fine" style={{ margin: 0 }}>This erases all progress on this device. Download it first if you want to keep it.</span>
              <button className="btn sm danger" onClick={() => { resetAll(); setConfirmReset(false); toast("Progress reset."); }}>Erase progress</button>
              <button className="btn sm" onClick={() => setConfirmReset(false)}>Cancel</button>
            </div>
          ) : <button className="linkbtn" onClick={() => setConfirmReset(true)}>Reset progress…</button>}
        </div>
      </div>
    </>
  );
}

function MetricRow({ lesson: l, values }: { lesson: LiteLesson; values: Record<string, number> }) {
  return (
    <div style={{ marginTop: 8 }}>
      <div className="fine" style={{ fontWeight: 700, color: "var(--ink)" }}>{l.id} · {l.title}</div>
      <div className="metric-card" style={{ marginTop: 4 }}>
        {[l.qualityMetric, l.systemsMetric].map((m) => {
          const v = computeMetric(m, values);
          const verdict = metricVerdict(m, v);
          return <div key={m.id}>{m.label}<b>{formatMetric(m, v)}</b>{verdict && <span className={`chip ${verdict.pass ? "good" : "bad"}`} style={{ marginTop: 6 }}>{verdict.pass ? "within limit" : "over limit"}</span>}</div>;
        })}
      </div>
    </div>
  );
}
