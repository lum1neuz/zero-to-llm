import { useState } from "react";
import type { LiteCourse, LiteLesson } from "../../lib/course.ts";
import { phaseLabel, weeksLabel } from "../../lib/course.ts";
import { currentLesson, cutIds, lessonState, prereqsMet, useProgress } from "../../lib/progress.ts";
import { I } from "../ui.tsx";
import { LessonSheet } from "./LessonSheet.tsx";

const OFFSETS = [0, 52, -40, 30, -52, 18, -24];

export function PathView({ course }: { course: LiteCourse }) {
  const p = useProgress();
  const [sheet, setSheet] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const cut = cutIds(course, p);
  const cur = currentLesson(course, p);
  const visible = (phase: string) => course.lessons.filter((l) => l.phase === phase && !cut.includes(l.id));
  const phaseDone = (phase: string) => visible(phase).every((l) => lessonState(l, p) === "done");

  // Leading phases that are fully done collapse into one strip.
  const core = course.phases.filter((ph) => !ph.optional);
  let doneLead = 0;
  while (doneLead < core.length && phaseDone(core[doneLead].id)) doneLead++;
  const doneCount = core.slice(0, doneLead).reduce((a, ph) => a + visible(ph.id).length, 0);
  const showDone = open.__done;

  const node = (l: LiteLesson, i: number) => {
    const st = lessonState(l, p);
    const isCur = cur?.id === l.id;
    const locked = !prereqsMet(course, p, l).ok && st === "new";
    const cls = st === "done" ? "done" : isCur ? "cur" : st === "learned" ? "half" : locked ? "" : "open";
    const bubble = isCur ? (st === "learned" ? "Build next" : st === "started" ? "Continue" : "Start") : null;
    return (
      <button key={l.id} className={`node ${cls}`} style={{ transform: `translateX(${OFFSETS[i % OFFSETS.length]}px)` }} onClick={() => setSheet(l.id)}
        aria-label={`${l.id} ${l.title}${st === "done" ? ", done" : locked ? ", locked" : ""}`}>
        {bubble && <span className="bubble">{bubble}</span>}
        <span className="disc">
          {st === "done" ? I.check : locked ? I.lock : l.id}
          {l.protected && <span className="badge" title="Protected: never cut">{I.shield}</span>}
        </span>
        <span className="lbl">{l.id} · {l.title}</span>
      </button>
    );
  };

  const sheetLesson = sheet ? course.lessons.find((l) => l.id === sheet)! : null;
  return (
    <>
      {doneLead > 0 && (
        <button className="done-strip" onClick={() => setOpen({ ...open, __done: !showDone })} aria-expanded={!!showDone}>
          {I.check} {doneLead === 1 ? phaseLabel(core[0]) : `${phaseLabel(core[0])} to ${phaseLabel(core[doneLead - 1])}`} done · {doneCount} lessons
          <span style={{ marginLeft: "auto", fontWeight: 400, fontSize: 13 }}>{showDone ? "Hide" : "Show"}</span>
        </button>
      )}
      {course.phases.map((ph, pi) => {
        const isLeadDone = !ph.optional && pi < doneLead;
        if (isLeadDone && !showDone) return null;
        const ls = visible(ph.id);
        const done = ls.filter((l) => lessonState(l, p) === "done").length;
        const hrs = ls.reduce((a, l) => a + l.hours, 0);
        const containsCur = !!cur && ls.some((l) => l.id === cur.id);
        const expanded = open[ph.id] ?? (containsCur || isLeadDone);
        return (
          <section className="phase" key={ph.id} aria-label={ph.title}>
            <button className="phase-h" onClick={() => setOpen({ ...open, [ph.id]: !expanded })} aria-expanded={expanded}>
              <h3>{phaseLabel(ph)}</h3>
              {ph.protected ? <span className="chip prot">{I.shield} protected</span> : ph.optional ? <span className="chip">optional</span> : <span className="chip">{done}/{ls.length}</span>}
            </button>
            <p className="phase-sub">{weeksLabel(ph.weeks)} · {hrs} h · {ph.blurb}</p>
            {expanded ? (
              <div className="path">{ls.map(node)}</div>
            ) : (
              <button className="phase-collapsed" style={{ width: "100%", background: "none", cursor: "pointer" }} onClick={() => setOpen({ ...open, [ph.id]: true })}>
                <span>{done}/{ls.length} lessons</span>
                <span className="bar"><i style={{ width: `${ls.length ? (done / ls.length) * 100 : 0}%` }} /></span>
                <span>Show</span>
              </button>
            )}
          </section>
        );
      })}
      {sheetLesson && <LessonSheet course={course} lesson={sheetLesson} onClose={() => setSheet(null)} />}
    </>
  );
}
