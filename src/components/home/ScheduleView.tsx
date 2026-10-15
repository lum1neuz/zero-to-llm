import type { LiteCourse } from "../../lib/course.ts";
import { phaseLabel, weeksLabel } from "../../lib/course.ts";
import { courseWeek, currentLesson, cutIds, lessonState, update, useProgress } from "../../lib/progress.ts";
import { I } from "../ui.tsx";

export function ScheduleView({ course }: { course: LiteCourse }) {
  const p = useProgress();
  const max = course.cutOrder.length;
  const cut = cutIds(course, p);
  const core = course.lessons.filter((l) => !course.phases.find((ph) => ph.id === l.phase)!.optional);
  const kept = core.filter((l) => !cut.includes(l.id));
  const hrs = kept.reduce((a, l) => a + l.hours, 0);
  const name = (id: string) => course.lessons.find((l) => l.id === id)!.title;
  const week = courseWeek(p, course.weeks);
  const cur = currentLesson(course, p);
  const curPhase = cur && course.phases.find((ph) => ph.id === cur.phase);
  const behind = curPhase?.weeks && week > curPhase.weeks[1];
  const setCut = (n: number) => update((d) => { d.cut = Math.max(0, Math.min(max, n)); d.cutAt = Date.now(); });

  return (
    <>
      <div className="stepper">
        <div className="stepper-row">
          <b>Behind schedule?</b>
          <div className="ctl">
            <button onClick={() => setCut(p.cut - 1)} disabled={p.cut === 0} aria-label="Cut one lesson fewer">−</button>
            <span aria-live="polite">{p.cut}/{max}</span>
            <button onClick={() => setCut(p.cut + 1)} disabled={p.cut === max} aria-label="Cut one more lesson">+</button>
          </div>
        </div>
        <p>{p.cut === 0 ? "On track. Nothing cut." : `Cutting ${cut.map(name).join(", ")}. Protected lessons never get cut.`}</p>
        <div className="hours">{kept.length} lessons · {hrs} h</div>
        {(p.cut === 0 || p.cut === max) && <p>{p.cut === 0 ? `Cut order: ${course.cutOrder.join(", ")}.` : "That's everything that can be cut."}</p>}
      </div>
      {behind && curPhase && (
        <p className="behind" style={{ marginTop: 12 }}>
          You're in week {week}. {curPhase.title} was planned for weeks {curPhase.weeks![0]}–{curPhase.weeks![1]}.
          {p.cut < max ? " Cutting a lesson is one way to catch up." : ""}
        </p>
      )}
      {course.phases.map((ph) => {
        const now = ph.weeks && week >= ph.weeks[0] && week <= ph.weeks[1];
        return (
          <div className="ph-block" key={ph.id}>
            <div className={`ph-row ${now ? "now" : ""}`}>
              <b>{phaseLabel(ph)}</b>
              <span>{weeksLabel(ph.weeks, true)}{now ? " · now" : ""}</span>
            </div>
            {course.lessons.filter((l) => l.phase === ph.id).map((l) => {
              const isCut = cut.includes(l.id);
              const done = lessonState(l, p) === "done";
              const isCur = cur?.id === l.id;
              const badge = l.protected ? <span className="chip prot">{I.shield} protected</span>
                : l.cut ? <span className={`chip ${isCut ? "bad" : ""}`}>{isCut ? "cut" : `cut #${l.cut}`}</span>
                : l.type === "sidequest" ? <span className="chip">optional</span> : <span />;
              return (
                <div key={l.id} className={`lrow ${done ? "done" : ""} ${isCut ? "cut" : ""} ${isCur ? "cur" : ""}`}>
                  <span className="lid">{done ? "✓ " : ""}{l.id}</span>
                  <span className="lt" title={l.title}>{l.title}</span>
                  {badge}
                  <span className="lh">{l.hours}h</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </>
  );
}
