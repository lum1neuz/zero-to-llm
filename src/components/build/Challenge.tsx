// Scrimba-style challenge: a short narrated setup types only the setup code and a stub with
// `...` placeholders, then stops. The learner writes the code; "Run tests" runs real tests in
// Pyodide; only the first failing test's diagnostic shows. Hints unlock one rung at a time,
// and each rung needs another test run first.
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { ChallengeStep } from "../../../pipeline/schema.ts";
import { blankChallenge, lessonRec, touchDay, update, useProgress, XP, type ChallengeRec } from "../../lib/progress.ts";
import { isReady, pythonVersion, runTests, type RunOutcome, type Runtime } from "../../lib/pyclient.ts";
import { esc, fmtTime, hl, inlineMd, plural } from "../../lib/text.ts";
import { Html, I, toast } from "../ui.tsx";

const CPS = 26;
const SPEEDS = [1, 1.5, 2];
const RUNGS = ["nudge", "concept", "skeleton", "solution"] as const;
const RUNG_LABEL = { nudge: "Nudge", concept: "Concept", skeleton: "Skeleton", solution: "Solution" };

export function stepNote(c: ChallengeRec | undefined, locked: string | null, minutes?: number) {
  if (c?.phase === "solved") return c.hints >= 4 ? "Solved · solution shown" : c.hints ? `Solved · ${plural(c.hints, "hint")}` : "Solved · no hints";
  if (locked) return locked;
  if (c?.phase === "challenge") return `Your turn · ${plural(c.runs, "run")}`;
  return `Challenge · ~${minutes ?? 30} min`;
}

function timeline(step: ChallengeStep) {
  let t = 0;
  const segs = step.setup.map((s) => {
    const start = t;
    const typeEnd = start + (s.type.length / CPS) * 1000;
    const end = typeEnd + s.pause * 1000;
    t = end;
    return { ...s, start, typeEnd, end };
  });
  return { segs, total: t };
}

interface Props { lessonId: string; step: ChallengeStep; startCode: string; runtime: Runtime }

export function Challenge({ lessonId, step, startCode, runtime }: Props) {
  const p = useProgress();
  const saved = p.lessons[lessonId]?.challenges[step.id] ?? blankChallenge();
  const { segs, total } = useMemo(() => timeline(step), [step]);
  const codeAt = useCallback((t: number) => {
    let c = startCode;
    for (const s of segs) {
      if (t >= s.typeEnd) c += s.type;
      else { if (t > s.start) c += s.type.slice(0, Math.floor(((t - s.start) / 1000) * CPS)); break; }
    }
    return c;
  }, [segs, startCode]);

  const [t, setT] = useState(saved.phase === "intro" ? saved.t : total);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [code, setCode] = useState(saved.phase === "intro" ? codeAt(saved.t) : saved.code ?? codeAt(total));
  const [outcome, setOutcome] = useState<RunOutcome | null>(null);
  const [running, setRunning] = useState<false | "boot" | "run">(false);
  const [solAsk, setSolAsk] = useState(false);
  const [runFlash, setRunFlash] = useState(false);
  const ta = useRef<HTMLTextAreaElement>(null);
  const pre = useRef<HTMLPreElement>(null);
  const gut = useRef<HTMLDivElement>(null);
  const raf = useRef(0);
  const intro = saved.phase === "intro";

  const save = useCallback((patch: Partial<ChallengeRec>) => update((d) => {
    const rec = lessonRec(d, lessonId);
    rec.challenges[step.id] = { ...(rec.challenges[step.id] ?? blankChallenge()), ...patch, at: Date.now() };
  }), [lessonId, step.id]);

  // ----- playback -----
  const enterChallenge = useCallback(() => {
    cancelAnimationFrame(raf.current);
    setPlaying(false);
    const c = codeAt(total);
    setT(total); setCode(c);
    save({ phase: "challenge", t: total, code: c });
    requestAnimationFrame(() => {
      const el = ta.current; if (!el) return;
      const at = c.indexOf("...");
      el.focus({ preventScroll: true });
      if (at >= 0) el.setSelectionRange(at, at + 3);
    });
  }, [codeAt, total, save]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let cur = t >= total ? 0 : t;
    const tick = (now: number) => {
      cur = Math.min(total, cur + (now - last) * speed);
      last = now;
      setT(cur);
      setCode(codeAt(cur));
      if (cur >= total) { enterChallenge(); return; }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed]);

  const pause = () => { setPlaying(false); save({ t }); };
  const togglePlay = () => (playing ? pause() : setPlaying(true));

  // keep the overlay scrolled with the textarea
  const sync = () => {
    const el = ta.current; if (!el) return;
    if (pre.current) { pre.current.scrollTop = el.scrollTop; pre.current.scrollLeft = el.scrollLeft; }
    if (gut.current) gut.current.scrollTop = el.scrollTop;
  };
  useEffect(() => { if (playing && ta.current) { ta.current.scrollTop = ta.current.scrollHeight; sync(); } }, [code, playing]);

  // ----- editing -----
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const edit = (v: string) => {
    setCode(v);
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => save({ code: v }), 400);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget;
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); return; }
    if (el.readOnly) return;
    if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      el.setRangeText("    ", el.selectionStart, el.selectionEnd, "end");
      edit(el.value);
    } else if (e.key === "Enter" && !e.shiftKey && el.selectionStart === el.selectionEnd) {
      const before = el.value.slice(0, el.selectionStart);
      const line = before.slice(before.lastIndexOf("\n") + 1);
      const indent = (/^\s*/.exec(line)?.[0] ?? "") + (/:\s*$/.test(line) ? "    " : "");
      if (indent) { e.preventDefault(); el.setRangeText("\n" + indent, el.selectionStart, el.selectionEnd, "end"); edit(el.value); }
    }
  };

  // ----- running -----
  const run = async () => {
    if (intro) { setRunFlash(true); setTimeout(() => setRunFlash(false), 450); return; }
    if (running) return;
    clearTimeout(saveTimer.current);
    const src = ta.current?.value ?? code;
    setRunning(isReady() ? "run" : "boot");
    const out = await runTests(runtime, src, step.tests, step.file);
    setRunning(false);
    setOutcome(out);
    const passed = out.kind === "ok" && out.result.tests.every((x) => x.status === "pass");
    let solvedNow = 0;
    update((d) => {
      const rec = lessonRec(d, lessonId);
      const c = (rec.challenges[step.id] ??= blankChallenge());
      c.code = src; c.at = Date.now();
      // Only a run that actually executed the tests counts as another try (unlocks the next hint).
      if (out.kind === "ok") c.runs++;
      if (passed && c.phase !== "solved") {
        c.phase = "solved"; c.xp = XP.challengeFull;
        solvedNow = c.xp;
        touchDay(d);
      }
    });
    if (solvedNow) {
      const h = p.lessons[lessonId]?.challenges[step.id]?.hints ?? 0;
      toast(`All tests pass. +${solvedNow} XP${h >= 4 ? " (solution shown)" : h ? ` (${plural(h, "hint")})` : ", no hints"}.`);
    }
  };

  // ----- hints -----
  const openHint = () => { if (running) return; save({ hints: saved.hints + 1, lastHintRun: saved.runs }); setSolAsk(false); };

  // ----- render pieces -----
  const seg = segs.find((s) => t < s.end) ?? segs[segs.length - 1];
  const status = intro ? (playing ? ["live", "Setup · playing"] : ["", "Setup · paused"]) : saved.phase === "solved" ? ["done", "Solved"] : ["mine", "Your turn"];
  const caption = intro ? (t === 0 && !playing ? "A short setup plays first. Then the code is yours to write." : seg.say) : saved.phase === "solved" ? step.onSolve : segs[segs.length - 1].say;
  const result = outcome?.kind === "ok" ? outcome.result : null;
  const firstFail = result ? result.tests.findIndex((x) => x.status === "fail" || x.status === "error") : -1;
  const passing = result ? result.tests.filter((x) => x.status === "pass").length : 0;

  return (
    <>
      <div className="stage-h">
        <h2>{step.title}</h2>
        <span className="chip gold">{saved.phase === "solved" ? `Earned ${saved.xp} XP` : `Challenge · up to ${XP.challengeFull} XP`}</span>
      </div>
      {step.intro && <Html as="p" className="fine" html={inlineMd(step.intro)} />}

      <div className="ed-wrap">
        <div className="ed-top">
          <span className="file">{step.file}</span>
          <span className={`chip ${status[0]}`}>{status[1]}</span>
          <span className="sp" />
          {intro && <span className={`run-reason ${runFlash ? "reason flash" : ""}`} style={{ margin: 0 }}>Finish or skip the setup to run tests</span>}
          {intro && <button className="btn sm" onClick={() => { pause(); enterChallenge(); }}>Skip to your turn</button>}
          <button className="btn sm" onClick={run} aria-disabled={intro || !!running}>
            {running ? "Running…" : "Run tests"} <kbd>Ctrl ↵</kbd>
          </button>
        </div>
        <div className="ed">
          <div className="gutter" ref={gut} aria-hidden="true">{code.split("\n").map((_, n) => n + 1).join("\n")}</div>
          <div className="ed-area">
            <pre ref={pre} aria-hidden="true" dangerouslySetInnerHTML={{ __html: hl(code) + (playing ? '<span class="caret"></span>' : "") + "\n" }} />
            <textarea ref={ta} value={code} readOnly={intro} spellCheck={false} wrap="off" autoCapitalize="off" autoComplete="off" autoCorrect="off"
              aria-label={`${step.file} editor`} onChange={(e) => edit(e.target.value)} onKeyDown={onKey} onScroll={sync} />
          </div>
          {intro && t === 0 && !playing && (
            <button className="big-play" onClick={() => setPlaying(true)}>
              <span className="circ">{I.play}</span>
              <span>Watch the setup · {fmtTime(total)}</span>
              <small>Then you write the code. Nothing gets typed for you.</small>
            </button>
          )}
        </div>
        <div className="caption" aria-live="polite"><span className="who">Narration</span><Html html={inlineMd(caption)} /></div>
        {intro && (
          <div className="ctrl">
            <button className="pbtn" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>{playing ? I.pause : I.play}</button>
            <span className="time">{fmtTime(t)} / {fmtTime(total)}</span>
            <div className="scrub">
              <input type="range" min={0} max={Math.round(total)} step={10} value={Math.round(t)} aria-label="Setup position"
                onChange={(e) => { setPlaying(false); const v = +e.target.value; setT(v); setCode(codeAt(v)); save({ t: v }); }} />
            </div>
            <button className="speed" onClick={() => setSpeed(SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length])} aria-label="Playback speed">{speed}×</button>
          </div>
        )}
      </div>

      <div className="duo">
        <div className="pnl">
          <div className="pnl-h">Tests<span>{!outcome ? (saved.phase === "solved" ? "passed" : "not run yet") : outcome.kind !== "ok" ? "couldn't run" : `${passing} / ${result!.tests.length} passing`}</span></div>
          <ul className="tlist">
            {outcome?.kind === "timeout" ? (
              <li className="fail"><span className="ic">!</span><div><b>Your code didn't finish</b><p className="why">It ran for {outcome.seconds} seconds without stopping. Is there a loop that never ends?</p></div></li>
            ) : outcome?.kind === "error" ? (
              <li className="fail"><span className="ic">!</span><div><b>The tests couldn't run</b><p className="why">{outcome.message}</p></div></li>
            ) : result ? result.tests.map((x, k) => (
              <li key={x.id} className={x.status}>
                <span className="ic">{x.status === "pass" ? "✓" : x.status === "skip" ? "·" : "✗"}</span>
                <div><b>{x.name}</b>{k === firstFail && <p className="why"><strong>What's wrong:</strong> <Html html={inlineMd(x.message)} /></p>}</div>
              </li>
            )) : step.testNames.map((x) => (
              <li key={x.id} className={saved.phase === "solved" ? "pass" : "skip"}><span className="ic">{saved.phase === "solved" ? "✓" : "·"}</span><div><b>{x.name}</b></div></li>
            ))}
          </ul>
        </div>
        <HintLadder step={step} rec={saved} intro={intro} running={!!running} solAsk={solAsk} setSolAsk={setSolAsk} openHint={openHint} peek={() => save({ peek: true })} />
      </div>

      <div className="console">
        <div className="con-h">Console<span>{pythonVersion ? `Python ${pythonVersion} · Pyodide ${runtime.pyodide} · mini-torch` : `Pyodide ${runtime.pyodide} · mini-torch`}</span></div>
        <pre aria-live="polite" dangerouslySetInnerHTML={{ __html: consoleHtml(outcome, running) }} />
      </div>
    </>
  );
}

function consoleHtml(o: RunOutcome | null, running: false | "boot" | "run") {
  if (running === "boot") return `<span class="warn">Loading the Python runtime… (Pyodide + NumPy, about 10 MB the first time, then cached for offline use)</span>`;
  if (running === "run") return `<span class="warn">Running tests…</span>`;
  if (!o) return `<span class="dim">Run the tests to see your script's output here.</span>`;
  if (o.kind === "timeout") return `<span class="err">Stopped after ${o.seconds} s. The Python runtime restarted.</span>`;
  if (o.kind === "error") return `<span class="err">${esc(o.message)}</span>`;
  const r = o.result;
  const parts: string[] = [];
  if (r.stdout) parts.push(esc(r.stdout.replace(/\n$/, "")));
  if (r.scriptError) parts.push(`<span class="err">${esc(r.scriptError.traceback)}</span>`);
  for (const w of r.warnings) parts.push(`<span class="warn"># ${esc(w)}</span>`);
  return parts.join("\n") || `<span class="dim">(your script printed nothing)</span>`;
}

function HintLadder({ step, rec: c, intro, running, solAsk, setSolAsk, openHint, peek }: {
  step: ChallengeStep; rec: ChallengeRec; intro: boolean; running: boolean; solAsk: boolean; setSolAsk: (v: boolean) => void; openHint: () => void; peek: () => void;
}) {
  const solved = c.phase === "solved";
  const xpLine = solved ? `earned ${c.xp} XP` : intro ? "opens on your turn" : `${XP.challengeFull} XP if solved now`;
  const block = (e: { preventDefault: () => void }) => e.preventDefault();
  return (
    <div className="pnl">
      <div className="pnl-h">Help<span>{xpLine}</span></div>
      {RUNGS.map((k, i) => {
        const isSol = k === "solution";
        const open = i < c.hints;
        const label = isSol ? "Solution" : `Hint ${i + 1} · ${RUNG_LABEL[k]}`;
        const text = step.hints[k];
        let body;
        if (open && (k === "nudge" || k === "concept")) body = <Html as="p" html={inlineMd(text)} />;
        else if (open && !(isSol && solved)) {
          const guarded = isSol && !solved;
          body = (
            <>
              <pre className={`hint-code ${guarded ? "nocopy" : ""}`} onCopy={guarded ? block : undefined} onCut={guarded ? block : undefined}
                onContextMenu={guarded ? block : undefined} onDragStart={guarded ? block : undefined} dangerouslySetInnerHTML={{ __html: hl(text.replace(/\n$/, "")) }} />
              {guarded && <p className="fine">Type it into the editor yourself. Copying is switched off here on purpose.</p>}
            </>
          );
        } else if (solved) {
          body = isSol
            ? (c.peek ? <><pre className="hint-code" dangerouslySetInnerHTML={{ __html: hl(step.solution.replace(/\n$/, "")) }} /><p className="fine">The reference solution. Yours can differ and still be right.</p></>
              : <button className="btn sm" onClick={peek}>Compare with the reference</button>)
            : open ? <Html as="p" html={inlineMd(text)} /> : <span className="lockr">Not needed</span>;
        } else if (intro) body = <span className="lockr">{I.lock} Opens on your turn</span>;
        else if (i === c.hints && running) body = <span className="lockr">Wait for the tests to finish.</span>;
        else if (i === c.hints && (i === 0 || c.runs > c.lastHintRun)) {
          body = !isSol ? <button className="btn sm" onClick={openHint}>Show hint {i + 1}</button>
            : solAsk ? (
              <>
                <p>Showing the solution marks this step as assisted. You'll still type it yourself.</p>
                <div className="confirm-row"><button className="btn sm warnbtn" onClick={openHint}>Show solution</button><button className="btn sm" onClick={() => setSolAsk(false)}>Not yet</button></div>
              </>
            ) : <button className="btn sm" onClick={() => setSolAsk(true)}>Show solution…</button>;
        } else body = <span className="lockr">{I.lock} {i > c.hints ? `Open hint ${i} first` : "Run the tests once more to unlock"}</span>;
        return <div key={k} className={`tier ${open ? "open" : ""}`}><b>{label}</b>{body}</div>;
      })}
      <p className="fine pnl-foot">The tests point at your actual mistake first. Each hint unlocks only after another run.</p>
    </div>
  );
}
