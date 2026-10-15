import { useEffect, useRef } from "react";
import { learnMinutes, type LiteCourse, type LiteLesson } from "../../lib/course.ts";
import { lessonRec, lessonState, prereqsMet, touchDay, update, useProgress } from "../../lib/progress.ts";
import { plural } from "../../lib/text.ts";
import { toast } from "../ui.tsx";

export function LessonSheet({ course, lesson: l, onClose }: { course: LiteCourse; lesson: LiteLesson; onClose: () => void }) {
  const p = useProgress();
  const ref = useRef<HTMLDivElement>(null);
  const rec = p.lessons[l.id];
  const st = lessonState(l, p);
  const pre = prereqsMet(course, p, l);
  const title = (id: string) => course.lessons.find((x) => x.id === id)?.title ?? id;
  const session = rec?.session;

  useEffect(() => {
    ref.current?.querySelector<HTMLElement>("a, button")?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onClose]);

  const toggleDone = () => {
    const on = !rec?.markedDone?.v;
    update((d) => { lessonRec(d, l.id).markedDone = { v: on, at: Date.now() }; if (on) touchDay(d); });
    toast(on ? `${l.id} marked done.` : `${l.id} is open again.`);
  };

  const learnLabel = rec?.learn?.done ? "Redo Learn cards" : session ? `Continue Learn · card ${session.ci + 1}` : "Start Learn";
  return (
    <>
      <button className="scrim-bg" aria-label="Close" onClick={onClose} />
      <div className="sheet" role="dialog" aria-modal="true" aria-label={`Lesson ${l.id}`} ref={ref}>
        <div className="eyebrow" style={{ margin: 0 }}>
          Lesson {l.id}{l.protected ? " · protected" : ""}{l.cut ? ` · cut #${l.cut} when behind` : ""}{l.type === "sidequest" ? " · optional" : ""}
        </div>
        <h2>{l.title}</h2>
        {l.summary && <p className="fine" style={{ fontSize: 14 }}>{l.summary}</p>}

        {!pre.ok && st !== "done" && (
          <div className="note">Needs {pre.missing.map((id) => `${id} ${title(id)}`).join(", ")} first. You can open it anyway.</div>
        )}
        {pre.usingFallback.map((id) => (
          <div className="note info" key={id}><strong>{id} is cut from your schedule.</strong> {l.fallbacks[id]}</div>
        ))}

        <div className="meta">
          {l.status === "ready" ? (
            <>
              <div><span>Learn · phone</span><b>{l.cards} cards · ~{learnMinutes(l.cards)} min {rec?.learn?.done ? "✓" : ""}</b></div>
              <div><span>Build · laptop</span><b>{plural(l.steps, "step")} · ~{l.hours} h</b></div>
            </>
          ) : (
            <div><span>Time</span><b>~{l.hours} h · {l.type}</b></div>
          )}
          <div><span>Quality metric</span><b>{l.qualityMetric.label}</b></div>
          <div><span>Systems metric</span><b>{l.systemsMetric.label}</b></div>
          {l.papers.length > 0 && <div><span>Paper</span><b>{l.papers.map((pp) => `${pp.title}${pp.authors ? ` · ${pp.authors.replace(" et al.", "")}` : ""}${pp.year ? ` ${pp.year}` : ""}`).join("; ")}</b></div>}
        </div>
        {l.systemsMetric.note && <p className="fine">{l.systemsMetric.note}</p>}

        {l.status === "ready" ? (
          <>
            <a className="btn primary block" href={`/learn/${l.id}/`}>{learnLabel}</a>
            <a className="btn block" href={`/build/${l.id}/`}>Open Build</a>
            <a className="btn block" href={`/read/${l.id}/`}>Skim the lesson{l.paperNotes ? ` · ${l.paperNotes === 1 ? "paper notes" : `${l.paperNotes} papers`}` : ""}</a>
            {st !== "done" && <button className="linkbtn" onClick={toggleDone}>I've already done this. Mark it done</button>}
            {rec?.markedDone?.v && <button className="linkbtn" onClick={toggleDone}>Unmark done</button>}
          </>
        ) : (
          <>
            <div className="note info">This lesson isn't written yet. When you've finished it on your own, mark it done so the path moves on.</div>
            <button className="btn primary block" onClick={toggleDone}>{rec?.markedDone?.v ? "Mark as not done" : "Mark as done"}</button>
          </>
        )}
        <button className="btn block" onClick={onClose}>Close</button>
      </div>
    </>
  );
}

