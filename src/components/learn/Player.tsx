// Learn: Mimo-style card deck. One optional hint per exercise (a correct answer after the hint
// earns 3 XP instead of 5). Check never silently disables: when it's not ready, it says why.
import { useEffect, useState } from "react";
import type { Card, Lesson } from "../../lib/course.ts";
import { lessonRec, scheduleMiss, streak, touchDay, update, useProgress, XP } from "../../lib/progress.ts";
import { quantStats } from "../../lib/quant.ts";
import { hl, inlineMd } from "../../lib/text.ts";
import { Html, I } from "../ui.tsx";
import { MdCharts } from "../Chart.tsx";
import { QuantGrid } from "./QuantGrid.tsx";

interface Answer { slots: (string | null)[]; bits: number; pick: number | null; order: number[]; val: string; hint: boolean }
interface Feedback { ok: boolean; title: string; why: string; practice?: boolean }
const RIGHT = ["Nice.", "Correct.", "Exactly.", "That's it."];

const newAnswer = (c: Card | undefined): Answer => ({
  slots: c?.type === "fill_blank" ? c.answer.map(() => null) : [],
  bits: c?.type === "slider_viz" ? c.params.startBits : 8,
  pick: null, order: [], val: "", hint: false,
});

/** Why Check can't run yet, or null when it can. */
function notReady(c: Card, a: Answer): string | null {
  switch (c.type) {
    case "fill_blank": {
      const left = a.slots.filter((s) => s === null).length;
      return left === 0 ? null : a.slots.length === 2 && left === 2 ? "Fill both blanks first" : left === a.slots.length ? `Fill all ${a.slots.length} blanks first` : `Fill the last ${left === 1 ? "blank" : `${left} blanks`} first`;
    }
    case "predict": case "quiz": return a.pick === null ? "Pick an answer first" : null;
    case "reorder": { const left = c.lines.length - a.order.length; return left ? `Place ${left === 1 ? "the last line" : `${left} more lines`} first` : null; }
    case "calc": return a.val.trim() === "" ? "Type a number first" : !Number.isFinite(parseFloat(a.val.replace(",", "."))) ? "That isn't a number yet" : null;
    default: return null;
  }
}

function grade(c: Card, a: Answer): { ok: boolean; why: string } {
  switch (c.type) {
    case "slider_viz": {
      const e = quantStats(a.bits, c.params.weights).err;
      const lim = c.params.errorLimit;
      if (a.bits === c.answer) {
        const below = quantStats(c.answer - 1, c.params.weights).err;
        return { ok: true, why: `${c.answer} bits gives mean error ${e.toFixed(3)}. At ${c.answer - 1} bits it jumps to ${below.toFixed(3)}. Each bit you drop doubles the step size.` };
      }
      return { ok: false, why: e >= lim ? `At ${a.bits} bits the mean error is ${e.toFixed(3)}, above ${lim}. Add a bit.` : `${a.bits} bits works, but you can go lower and stay under ${lim}.` };
    }
    case "fill_blank": {
      const ok = a.slots.every((s, i) => s === c.answer[i]);
      if (ok) return { ok, why: c.whyOk };
      const wrong = a.slots.find((s, i) => s !== c.answer[i]);
      return { ok, why: (wrong && c.tips[wrong]) || c.whyWrong };
    }
    case "predict": { const o = c.options[a.pick!]; return { ok: o.correct, why: o.why }; }
    case "quiz": return { ok: a.pick === c.answer, why: c.options[a.pick!].why };
    case "reorder": { const ok = a.order.every((v, i) => v === i); return { ok, why: ok ? c.whyOk : c.whyWrong }; }
    case "calc": { const v = parseFloat(a.val.replace(",", ".")); const ok = v >= c.answer.min && v <= c.answer.max; return { ok, why: ok ? c.whyOk : c.whyWrong }; }
    default: return { ok: true, why: "" };
  }
}

export default function Player({ lesson }: { lesson: Lesson }) {
  const cards = lesson.learn;
  const p = useProgress();
  const saved = p.lessons[lesson.id]?.session;
  const [ci, setCi] = useState(() => (saved ? Math.min(saved.ci, cards.length - 1) : 0));
  const [gained, setGained] = useState(saved?.gained ?? 0);
  const [correct, setCorrect] = useState(saved?.correct ?? 0);
  const [a, setA] = useState<Answer>(() => newAnswer(cards[ci]));
  const [fb, setFb] = useState<Feedback | null>(null);
  const [flash, setFlash] = useState(false);
  const [finished, setFinished] = useState<null | { gained: number; correct: number; missed: number }>(null);
  const [missed, setMissed] = useState(0);
  // Cards already scored (XP or a review entry). Going back to one is practice only.
  const [scored, setScored] = useState<number[]>(saved?.scored ?? []);
  const [furthest, setFurthest] = useState(saved?.ci ?? 0);
  const c = cards[ci];
  const exercises = cards.filter((x) => x.type !== "concept").length;

  useEffect(() => { document.querySelector<HTMLElement>(".scroll")?.scrollTo(0, 0); }, [ci]);
  // Every card was scored before a reload: finish the deck with the saved totals.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (saved && saved.ci >= cards.length) next(saved.gained, saved.correct, cards.length - 1); }, []);

  // The saved position never moves backwards, so going back and reloading resumes at the furthest card.
  const persistSession = (nextCi: number, g: number, cr: number, sc = scored) => {
    const far = Math.max(furthest, nextCi);
    setFurthest(far);
    update((d) => { lessonRec(d, lesson.id).session = { ci: far, gained: g, correct: cr, scored: sc, at: Date.now() }; });
  };

  const check = () => {
    if (c.type === "concept") return next(gained, correct);
    const why = notReady(c, a);
    if (why) { setFlash(true); setTimeout(() => setFlash(false), 450); return; }
    const r = grade(c, a);
    const practice = scored.includes(ci);
    let g = gained, cr = correct;
    if (r.ok && !practice) { g += a.hint ? XP.cardWithHint : XP.card; cr++; setGained(g); setCorrect(cr); }
    if (!r.ok && c.type === "quiz" && !practice) { setMissed(missed + 1); update((d) => scheduleMiss(d, lesson.id, c.ref)); }
    setFb({ ok: r.ok, title: r.ok ? RIGHT[ci % RIGHT.length] : "Not quite", why: r.why, practice });
    // Once a card is scored, a reload resumes after it and going back to it is practice, so its XP can't be earned twice.
    if ((r.ok || c.type === "quiz") && !practice) {
      const sc = [...scored, ci];
      setScored(sc);
      persistSession(ci + 1, g, cr, sc);
    }
  };

  const next = (g = gained, cr = correct, from = ci) => {
    const n = from + 1;
    setFb(null);
    if (n >= cards.length) {
      const total = g + XP.learnBonus;
      update((d) => {
        const rec = lessonRec(d, lesson.id);
        rec.learn = { done: true, bestXp: Math.max(rec.learn?.bestXp ?? 0, total), correct: cr, total: exercises, at: Date.now() };
        delete rec.session;
        touchDay(d);
      });
      setFinished({ gained: total, correct: cr, missed });
      return;
    }
    setCi(n);
    setA(newAnswer(cards[n]));
    persistSession(n, g, cr);
  };

  const retry = () => { setFb(null); setA({ ...newAnswer(c), hint: a.hint, bits: a.bits }); };
  const back = () => { if (ci === 0) return; setFb(null); setCi(ci - 1); setA(newAnswer(cards[ci - 1])); };

  if (finished) {
    return (
      <div className="learn-col">
        <div className="scroll"><div className="done-screen">
          <div className="xp-ring"><div>+{finished.gained}<small>XP</small></div></div>
          <h2 style={{ fontSize: 24 }}>{lesson.build.length > 0 ? "Learn done. Now build it." : "Nicely done."}</h2>
          <div className="stats">
            <div><b>{streak(p)}</b>day streak</div>
            <div><b>{finished.correct}/{exercises}</b>correct</div>
            <div><b>{finished.missed}</b>to review</div>
          </div>
          {lesson.build.length > 0 ? (
            <>
              <div className="handoff"><strong>Build is next, on your laptop.</strong><br />{lesson.build.length} steps, about {lesson.hours} hours. {lesson.goal ? lesson.goal.replace(/\*\*/g, "").split(". ")[0] + "." : ""}</div>
              <a className="btn primary block" href={`/build/${lesson.id}/`}>Open Build</a>
            </>
          ) : (
            <div className="handoff"><strong>That's this lesson done.</strong><br />There's no Build part here. Head back to your path for the next lesson.</div>
          )}
          {lesson.reading.length > 0 && <a className="btn block" href={`/read/${lesson.id}/`}>Skim the lesson{lesson.reading.length > 1 ? " and paper notes" : ""}</a>}
          <a className="btn block" href="/">Back to path</a>
        </div></div>
      </div>
    );
  }

  const reason = fb ? null : notReady(c, a);
  const hasHint = "hint" in c && !!c.hint;
  const quizN = cards.filter((x) => x.type === "quiz");
  return (
    <div className="learn-col">
      <div className="ls-head">
        <a className="icon-btn" href="/" aria-label="Leave lesson">{I.cross}</a>
        {ci > 0 && <button className="icon-btn" onClick={back} aria-label="Previous card" title="Previous card">{I.back}</button>}
        <div className="prog" role="progressbar" aria-label="Lesson progress" aria-valuenow={ci} aria-valuemin={0} aria-valuemax={cards.length}><i style={{ width: `${(ci / cards.length) * 100}%` }} /></div>
        <span className="pill">{ci + 1}/{cards.length}</span>
      </div>
      <div className="scroll card-body">
        {ci === 0 && !saved && !fb && lesson.reading.some((d) => d.kind === "primer") && (
          <a className="primer-nudge" href={`/read/${lesson.id}/#primer`}>
            <b>New to this?</b> Read the primer first: the ideas in plain words, with worked examples. {lesson.reading.find((d) => d.kind === "primer")!.minutes ?? 10} min.
          </a>
        )}
        <CardView card={c} a={a} setA={setA} locked={!!fb} quizLabel={c.type === "quiz" ? `Quiz · ${quizN.indexOf(c) + 1} of ${quizN.length}` : undefined} />
        {a.hint && hasHint && <div className="hintbox">{I.bulb}<Html html={inlineMd((c as { hint: string }).hint)} /></div>}
      </div>
      {fb ? (
        <div className={`fb ${fb.ok ? "ok" : "no"}`} role="status">
          <div className="fb-title">{fb.ok ? I.check : I.cross} {fb.title}</div>
          <Html as="p" html={inlineMd(fb.why)} />
          {!fb.ok && c.type === "quiz" && !fb.practice && <p><strong>Added to Review.</strong> It comes back tomorrow.</p>}
          {fb.practice && <p className="fine" style={{ margin: 0 }}>Practice: you already scored this card, so it doesn't change your XP.</p>}
          <div className="fb-actions">
            {fb.ok || c.type === "quiz" ? <button className="btn primary block" onClick={() => next()} autoFocus>Continue</button> : (
              <><button className="btn primary block" onClick={retry} autoFocus>Try again</button><button className="btn block" onClick={() => next()}>Skip for now</button></>
            )}
          </div>
        </div>
      ) : (
        <div className="ls-foot">
          {reason && <p className={`reason ${flash ? "flash" : ""}`} aria-live="polite">{reason}</p>}
          <div className={hasHint ? "row" : ""}>
            {hasHint && (
              <button className="btn" onClick={() => setA({ ...a, hint: true })} aria-disabled={a.hint} title={a.hint ? "Hint shown. A correct answer now earns 3 XP." : "Show a hint. A correct answer then earns 3 XP instead of 5."}>
                {I.bulb} {a.hint ? "Hint shown" : "Hint"}
              </button>
            )}
            <button className="btn primary block" aria-disabled={!!reason} onClick={check}>{c.type === "concept" ? "Continue" : "Check"}</button>
          </div>
        </div>
      )}
    </div>
  );
}

function CardView({ card: c, a, setA, locked, quizLabel }: { card: Card; a: Answer; setA: (a: Answer) => void; locked: boolean; quizLabel?: string }) {
  const eyebrow = quizLabel ?? c.eyebrow;
  const head = <>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h2>{c.type === "quiz" ? c.q : c.title}</h2></>;
  switch (c.type) {
    case "concept": {
      const v = c.visual;
      return (
        <>
          {head}
          {v && (
            <div className="mem" role="img" aria-label={v.bars.map((b) => `${b.label} ${b.text ?? b.value}`).join(", ") + (v.limit ? `; limit ${v.limit.label}` : "")}>
              {v.limit && <div className="vram" style={{ left: `calc(58px + (100% - 58px) * ${v.limit.value / v.max})` }}><span>{v.limit.label}</span></div>}
              {v.bars.map((b) => (
                <div className="mem-row" key={b.label}><b>{b.label}</b>
                  <div className={`mem-bar ${v.limit && b.value > v.limit.value ? "over" : ""}`} style={{ width: `${(b.value / v.max) * 100}%` }}>{b.text ?? b.value}</div></div>
              ))}
            </div>
          )}
          <MdCharts className="md" md={c.body} charts={c.charts} />
        </>
      );
    }
    case "slider_viz": {
      const { minBits, maxBits, weights, errorLimit } = c.params;
      const st = quantStats(a.bits, weights);
      const set = (b: number) => setA({ ...a, bits: Math.max(minBits, Math.min(maxBits, b)) });
      return (
        <>
          {head}
          <QuantGrid bits={a.bits} weights={weights} />
          <label htmlFor="bits" className="phase-sub" style={{ display: "block", margin: "6px 0 0" }}>Bit width</label>
          <div className="bitrow">
            <button className="stepb" onClick={() => set(a.bits - 1)} aria-label="One bit fewer" disabled={locked || a.bits <= minBits}>−</button>
            <input type="range" className="range" id="bits" min={minBits} max={maxBits} step={1} value={a.bits} disabled={locked} onChange={(e) => set(+e.target.value)} />
            <button className="stepb" onClick={() => set(a.bits + 1)} aria-label="One bit more" disabled={locked || a.bits >= maxBits}>+</button>
          </div>
          <div className="readout" aria-live="polite">
            <div><b>{a.bits}</b>bits</div><div><b>±{st.qmax}</b>int range</div>
            <div className={st.err < errorLimit ? "ok" : "bad"}><b>{st.err.toFixed(3)}</b>mean error</div>
          </div>
          {c.body && <Html as="p" html={inlineMd(c.body)} />}
        </>
      );
    }
    case "fill_blank": {
      const nextIdx = a.slots.indexOf(null);
      const parts = c.code.replace(/\n$/, "").split(/(\[\[\d+\]\])/);
      return (
        <>
          {head}
          <div className="code wrap">
            {parts.map((part, k) => {
              const m = /^\[\[(\d+)\]\]$/.exec(part);
              if (!m) return <span key={k} dangerouslySetInnerHTML={{ __html: hl(part) }} />;
              const i = +m[1];
              return (
                <button key={k} className={`slot ${a.slots[i] ? "filled" : ""} ${i === nextIdx ? "next" : ""}`} disabled={locked}
                  aria-label={a.slots[i] ? `Blank ${i + 1}: ${a.slots[i]}. Tap to clear` : `Blank ${i + 1}, empty`}
                  onClick={() => { const s = [...a.slots]; s[i] = null; setA({ ...a, slots: s }); }}>{a.slots[i] ?? " "}</button>
              );
            })}
          </div>
          <div className="tokens">
            {c.tokens.map((t, k) => {
              // A token listed twice is two chips: chip k is used once the slots hold more copies than the chips before it.
              const used = a.slots.filter((s) => s === t).length > c.tokens.slice(0, k).filter((x) => x === t).length;
              return (
                <button key={`${k}-${t}`} className={`tok ${used ? "used" : ""}`} disabled={locked || used}
                  onClick={() => { const i = a.slots.indexOf(null); if (i >= 0) { const s = [...a.slots]; s[i] = t; setA({ ...a, slots: s }); } }}>{t}</button>
              );
            })}
          </div>
        </>
      );
    }
    case "predict": case "quiz": {
      const isRight = (j: number) => (c.type === "quiz" ? j === c.answer : c.options[j].correct);
      const pickedRight = a.pick !== null && isRight(a.pick);
      return (
        <>
          {head}
          {c.type === "predict" && <div className="code" dangerouslySetInnerHTML={{ __html: hl(c.code.replace(/\n$/, "")) }} />}
          <div className="opts">
            {c.options.map((o, j) => {
              let cls = "";
              if (locked) { if (isRight(j) && (pickedRight || c.type === "quiz")) cls = "right"; else if (j === a.pick && !isRight(j)) cls = "wrong"; }
              return <button key={j} className={`opt ${c.type === "predict" ? "mono" : ""} ${cls}`} aria-pressed={a.pick === j} disabled={locked} onClick={() => setA({ ...a, pick: j })}>{o.text}</button>;
            })}
          </div>
        </>
      );
    }
    case "reorder":
      return (
        <>
          {head}
          <div className="order" aria-label="Your order">
            {a.order.map((i) => <button key={i} className="line" disabled={locked} onClick={() => setA({ ...a, order: a.order.filter((x) => x !== i) })} dangerouslySetInnerHTML={{ __html: hl(c.lines[i]) }} />)}
          </div>
          <div className="pool" aria-label="Lines to place">
            {c.shuffle.filter((i) => !a.order.includes(i)).map((i) => <button key={i} className="line" disabled={locked} onClick={() => setA({ ...a, order: [...a.order, i] })} dangerouslySetInnerHTML={{ __html: hl(c.lines[i]) }} />)}
          </div>
        </>
      );
    case "calc":
      return (
        <>
          {head}
          {c.given.length > 0 && <div className="given">{c.given.map((g) => <span key={g.label} style={{ display: "contents" }}><span>{g.label}</span><b>{g.value}</b></span>)}</div>}
          <div className="calc">
            <input inputMode="decimal" autoComplete="off" placeholder={c.placeholder ?? "0"} value={a.val} aria-label={`Answer in ${c.unit}`} disabled={locked}
              onChange={(e) => setA({ ...a, val: e.target.value })}
              onKeyDown={(e) => { if (e.key === "Enter") (document.querySelector(".ls-foot .btn.primary") as HTMLButtonElement | null)?.click(); }} />
            <span>{c.unit}</span>
          </div>
        </>
      );
  }
}
