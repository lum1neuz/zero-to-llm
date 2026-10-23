import { useEffect, useRef, useState } from "react";
import type { Lesson } from "../../lib/course.ts";
import { MdCharts } from "../Chart.tsx";
import { phaseLabel } from "../../lib/course.ts";
import type { Runtime } from "../../lib/pyclient.ts";
import type { LiteCourse } from "../../lib/course.ts";
import { checkValue, lessonRec, lessonState, prereqsMet, touchDay, update, useProgress } from "../../lib/progress.ts";
import { blockMd, inlineMd } from "../../lib/text.ts";
import { Html, I, toast } from "../ui.tsx";
import { Challenge, stepNote } from "./Challenge.tsx";
import { MetricsStep } from "./MetricsStep.tsx";
import { Replay } from "./Replay.tsx";

export default function Workspace({ lesson, course, runtime }: { lesson: Lesson; course: LiteCourse; runtime: Runtime }) {
  const p = useProgress();
  const rec = p.lessons[lesson.id];
  const lite = course.lessons.find((l) => l.id === lesson.id)!;
  const phase = course.phases.find((ph) => ph.id === lesson.phase)!;
  const state = lessonState(lite, p);
  const pre = prereqsMet(course, p, lite);
  const title = (id: string) => course.lessons.find((l) => l.id === id)?.title ?? id;

  const lockedReason = (i: number) => {
    const s = lesson.build[i];
    if (s.kind !== "challenge" || !("from" in s.start)) return null;
    const from = s.start.from;
    const n = lesson.build.findIndex((x) => x.id === from) + 1;
    return rec?.challenges[from]?.phase === "solved" ? null : `Locked · solve step ${n} first`;
  };
  const firstOpen = Math.max(0, lesson.build.findIndex((s, i) => s.kind === "challenge" && rec?.challenges[s.id]?.phase !== "solved" && !lockedReason(i)));
  const [stepIdx, setStepIdx] = useState(firstOpen);
  const step = lesson.build[stepIdx];

  // celebrate the moment the lesson completes
  const prevState = useRef(state);
  useEffect(() => {
    if (prevState.current !== "done" && state === "done") {
      const next = course.lessons.find((l) => l.order === lite.order + 1);
      toast(`${lesson.id} complete.${next ? ` ${next.id} ${next.title} is next on your path.` : ""}`);
    }
    prevState.current = state;
  }, [state]);

  const pick = (i: number) => {
    const why = lockedReason(i);
    if (why) { toast(`${why.replace("Locked · s", "S")}. This step builds on that code.`); return; }
    setStepIdx(i);
  };
  const autoNote = (auto: string) => {
    const [kind, target] = auto.split(":");
    if (kind === "learn") return "Ticks itself when Learn is done";
    if (kind === "metrics") return "Ticks itself when you save your metrics";
    const n = lesson.build.findIndex((s) => s.id === target) + 1;
    return `Ticks itself when step ${n}'s tests pass`;
  };
  const toggle = (id: string, v: boolean) => update((d) => { lessonRec(d, lesson.id).checks[id] = { v, at: Date.now() }; if (v) touchDay(d); });

  const startCode = (s: Lesson["build"][number]) => {
    if (s.kind !== "challenge") return "";
    if ("code" in s.start) return s.start.code;
    const from = lesson.build.find((x) => x.kind === "challenge" && x.id === (s.start as { from: string }).from);
    return rec?.challenges[(s.start as { from: string }).from]?.code ?? (from?.kind === "challenge" ? from.solution : "");
  };

  const learnDone = !!rec?.learn?.done;
  // The concept recap comes from the primer: every section has a one-line key idea and the text
  // behind it. Locked sections (answers to Learn exercises) never show here.
  const primer = lesson.reading.find((d) => d.kind === "primer");
  const concepts = (primer?.sections ?? []).filter((s) => !s.locked && (s.key || s.body));
  return (
    <div className="build">
      <header className="bh">
        <div>
          <div className="crumbs"><a href="/">{phaseLabel(phase)}</a> › {lesson.id}</div>
          <h1>{lesson.title}</h1>
          <div className="chips">
            <span className="chip">{lesson.hours} h</span>
            {lesson.protected && <span className="chip prot">{I.shield} protected · never cut</span>}
            {lesson.prerequisites.map((id) => <span key={id} className={`chip ${pre.missing.includes(id) ? "gold" : ""}`}>needs {id} {title(id)}{pre.usingFallback.includes(id) ? " (cut)" : ""}</span>)}
            {lesson.reading.length > 0 && <a className="chip prot" href={`/read/${lesson.id}/`}>Read: lesson{lesson.reading.length > 1 ? " + paper notes" : ""}</a>}
            {lesson.papers.map((pp) => pp.url ? <a key={pp.title} className="chip" href={pp.url} target="_blank" rel="noopener">Paper: {pp.title} ↗</a> : <span key={pp.title} className="chip">Paper: {pp.title}</span>)}
          </div>
        </div>
        <a className="learn-status" href={`/learn/${lesson.id}/`}>
          {learnDone ? <><span className="chip good">{I.check} Learn done</span><span>{lesson.learn.length}/{lesson.learn.length} cards</span></>
            : <><span className="chip gold">Learn {rec?.session ? `${rec.session.ci}/${lesson.learn.length}` : "not started"}</span><span>Open on your phone</span></>}
        </a>
      </header>
      <div className="bgrid">
        <aside className="rail">
          <p className="narrow-note">Build is made for a laptop. It works here, but writing code is easier with a keyboard.</p>
          {pre.usingFallback.map((id) => <p key={id} className="fine" style={{ margin: 0 }}><strong>{id} is cut:</strong> {lesson.fallbacks[id]}</p>)}
          <div className="rail-h">Build steps</div>
          <ol className="steps">
            {lesson.build.map((s, i) => {
              const solved = s.kind === "challenge" && rec?.challenges[s.id]?.phase === "solved";
              const note = s.kind === "challenge" ? stepNote(rec?.challenges[s.id], lockedReason(i), s.minutes)
                : s.kind === "replay" ? `Recording · your GPU` : rec?.metrics ? "Saved" : "A few minutes";
              return (
                <li key={s.id}>
                  <button onClick={() => pick(i)} aria-current={stepIdx === i ? "step" : undefined}>
                    <span className={`n ${solved || (s.kind === "metrics" && rec?.metrics) ? "ok" : ""}`}>{solved || (s.kind === "metrics" && rec?.metrics) ? "✓" : i + 1}</span>
                    <b>{s.title}</b><small>{note}</small>
                  </button>
                </li>
              );
            })}
          </ol>
          <div className="rail-h" style={{ marginTop: 12 }}>Done when</div>
          <ul className="checks">
            {lesson.doneWhen.map((d) => {
              const v = checkValue(lite, rec, d.id);
              return (
                <li key={d.id}>
                  <label className={d.auto ? "auto" : ""}>
                    <input type="checkbox" checked={v} disabled={!!d.auto} onChange={(e) => toggle(d.id, e.target.checked)} />
                    <span>{d.text}<small>{d.auto ? autoNote(d.auto) : "Tick it after your own run"}</small></span>
                  </label>
                </li>
              );
            })}
          </ul>
          <p className="fine">These boxes and Learn decide when the lesson is done. They travel to your phone with the progress export.</p>
          {lesson.goal && <details><summary>Goal</summary><Html as="div" html={blockMd(lesson.goal)} /></details>}
          {lesson.hints && <details><summary>Stuck on the GPU part?</summary><Html as="div" html={blockMd(lesson.hints)} /></details>}
        </aside>
        <main className="stage">
          {state === "done" && <div className="complete">{I.check} {lesson.id} complete. The next lesson is open on your path.</div>}
          {step.kind === "challenge" && <Challenge key={step.id} lessonId={lesson.id} step={step} startCode={startCode(step)} runtime={runtime} />}
          {step.kind === "replay" && <Replay key={step.id} step={step} />}
          {step.kind === "metrics" && <MetricsStep key={step.id} lesson={lesson} title={step.title} intro={step.intro} />}
          {concepts.length > 0 && (
            <details className="recap">
              <summary>Review the concepts <span>{concepts.length} key ideas from the primer</span></summary>
              {concepts.map((s) => (
                <details key={s.id} className="idea">
                  <summary><b>{s.title}</b>{s.key && <Html className="key" html={inlineMd(s.key)} />}</summary>
                  {s.body && <MdCharts className="md" md={s.body} charts={s.charts} />}
                </details>
              ))}
              <p className="fine"><a href={`/read/${lesson.id}/`}>Open the full primer</a></p>
            </details>
          )}
        </main>
      </div>
    </div>
  );
}
