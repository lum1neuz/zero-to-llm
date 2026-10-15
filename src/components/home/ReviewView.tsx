import { useState } from "react";
import type { LiteCourse } from "../../lib/course.ts";
import { answerReview, daysBetween, dueReviews, today, touchDay, update, useProgress, XP } from "../../lib/progress.ts";
import { inlineMd } from "../../lib/text.ts";
import { Html, I } from "../ui.tsx";

const when = (due: string) => {
  const d = daysBetween(today(), due);
  return d <= 0 ? "today" : d === 1 ? "tomorrow" : `in ${d} days`;
};

export function ReviewView({ course }: { course: LiteCourse }) {
  const p = useProgress();
  const [queue, setQueue] = useState<string[] | null>(null);
  const [i, setI] = useState(0);
  const [pick, setPick] = useState<number | null>(null);
  const [checked, setChecked] = useState(false);
  const [right, setRight] = useState(0);
  const [flash, setFlash] = useState(false);

  const quizOf = (k: string) => {
    const r = p.review[k];
    const l = course.lessons.find((x) => x.id === r.lesson);
    return { r, l, q: l?.quiz.find((q) => q.id === r.qid) };
  };
  const due = dueReviews(p).filter(([k]) => quizOf(k).q);
  const upcoming = Object.entries(p.review).filter(([k, r]) => !r.retired && r.due > today() && quizOf(k).q).sort((a, b) => (a[1].due < b[1].due ? -1 : 1));
  const retired = Object.values(p.review).filter((r) => r.retired).length;

  if (queue) {
    if (i >= queue.length) {
      return (
        <div className="done-screen">
          <div className="xp-ring"><div>+{right * XP.review}<small>XP</small></div></div>
          <h2 style={{ fontSize: 24 }}>Review done</h2>
          <p className="fine">{right} of {queue.length} right. Right answers come back later: 3 days, then 7, then they retire. Misses come back tomorrow.</p>
          <button className="btn primary block" onClick={() => { setQueue(null); setI(0); setRight(0); }}>Back to Review</button>
        </div>
      );
    }
    const k = queue[i];
    const { q, l } = quizOf(k);
    if (!q || !l) { setI(i + 1); return null; }
    const ok = pick === q.answer;
    const check = () => {
      if (pick === null) { setFlash(true); setTimeout(() => setFlash(false), 400); return; }
      setChecked(true);
      if (ok) setRight(right + 1);
      update((d) => { answerReview(d, k, ok); touchDay(d); });
    };
    const next = () => { setI(i + 1); setPick(null); setChecked(false); };
    return (
      <div className="card-body">
        <div className="eyebrow">Review · {i + 1} of {queue.length} · from {l.id}</div>
        <h2>{q.q}</h2>
        <div className="opts">
          {q.options.map((o, j) => (
            <button key={j} className={`opt ${checked && j === q.answer ? "right" : checked && j === pick ? "wrong" : ""}`}
              aria-pressed={pick === j} disabled={checked} onClick={() => setPick(j)}>{o.text}</button>
          ))}
        </div>
        {checked ? (
          <div className={`fb ${ok ? "ok" : "no"}`} role="status" style={{ marginTop: 16, borderRadius: 14 }}>
            <div className="fb-title">{ok ? I.check : I.cross} {ok ? "Right." : "Not quite"}</div>
            <Html as="p" html={inlineMd(q.options[pick!].why)} />
            {!ok && <p><strong>It comes back tomorrow.</strong></p>}
            <button className="btn primary block" onClick={next}>Continue</button>
          </div>
        ) : (
          <div style={{ marginTop: 16 }}>
            {pick === null && <p className={`reason ${flash ? "flash" : ""}`}>Pick an answer first</p>}
            <button className="btn primary block" aria-disabled={pick === null} onClick={check}>Check</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <p className="phase-sub" style={{ marginTop: 0 }}>Quiz questions you miss come back after 1, 3 and 7 days. Get one right at 7 days and it retires.</p>
      <div className="sec-h">Due now</div>
      {due.length ? (
        <div className="rev">
          {due.map(([k]) => { const { q, l, r } = quizOf(k); return (
            <div className="rev-item" key={k}><span>{q!.q}</span><span className="chip gold">today</span><small>From {l!.id} {l!.title} · missed {r.misses}×</small></div>
          ); })}
        </div>
      ) : <div className="empty">Nothing due. Missed quiz questions from Learn show up here.</div>}
      {due.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <button className="btn primary block" onClick={() => { setQueue(due.map(([k]) => k)); setI(0); setRight(0); }}>Review {due.length} due</button>
        </div>
      )}
      {upcoming.length > 0 && (
        <>
          <div className="sec-h">Coming up</div>
          <div className="rev">
            {upcoming.map(([k, r]) => { const { q, l } = quizOf(k); return (
              <div className="rev-item" key={k}><span>{q!.q}</span><span className="chip">{when(r.due)}</span><small>From {l!.id} · step {r.step + 1} of 3</small></div>
            ); })}
          </div>
        </>
      )}
      {retired > 0 && <p className="fine" style={{ marginTop: 14 }}>{retired} question{retired === 1 ? "" : "s"} retired.</p>}
    </>
  );
}
