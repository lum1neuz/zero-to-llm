// GPU-only steps: a recorded run played back, clearly labelled as a recording.
import { useEffect, useRef, useState } from "react";
import type { ReplayStep } from "../../../pipeline/schema.ts";
import { esc, inlineMd } from "../../lib/text.ts";
import { Html } from "../ui.tsx";

export function Replay({ step }: { step: ReplayStep }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const term = useRef<HTMLPreElement>(null);
  const log = step.log;
  const done = i >= log.length;

  useEffect(() => {
    if (!playing) return;
    if (i >= log.length) { setPlaying(false); return; }
    const l = log[i];
    const id = setTimeout(() => setI(i + 1), i === 0 ? 200 : l.delay ?? (l.c === "dim" ? 900 : 500));
    return () => clearTimeout(id);
  }, [playing, i, log]);
  useEffect(() => { if (term.current) term.current.scrollTop = term.current.scrollHeight; }, [i]);

  const toggle = () => {
    if (playing) { setPlaying(false); return; }
    if (done) setI(0);
    setPlaying(true);
  };
  const layers = log.filter((l) => l.layer !== undefined && l.value !== undefined);
  const shown = new Set(log.slice(0, i).filter((l) => l.layer !== undefined).map((l) => l.layer));
  const peak = Math.max(...layers.map((l) => l.value!), step.chart?.flagAbove ?? 0) * 1.1;
  const rec = step.recorded;

  return (
    <>
      <div className="stage-h"><h2>{step.title}</h2></div>
      <Html as="p" className="fine" html={inlineMd(step.intro)} />
      <div className="rec-note" role="note">
        <span className="chip gold">{rec.illustrative ? "Illustrative recording" : "Recording"}</span>
        <span>
          Recorded on {rec.hardware}{rec.date ? `, ${rec.date}` : ""}. Nothing runs here: this plays back a log.
          {rec.note ? ` ${rec.note}` : ""}
        </span>
      </div>
      <div className="term-wrap">
        <div className="ed-top">
          <span className="file">terminal · replay</span>
          <span className="chip rec">{playing ? `Replaying at ${step.speed}×` : done ? "Finished" : i ? "Paused" : "Recording"}</span>
          <span className="sp" />
          <button className="btn sm" onClick={toggle}>{playing ? "Pause" : done ? "Replay again" : i ? "Resume" : "Play replay"}</button>
        </div>
        <pre className="term" ref={term} aria-live="off" dangerouslySetInnerHTML={{
          __html: log.slice(0, i).map((l) => `<span class="${l.c}">${esc(l.t)}</span>`).join("\n") + (playing ? '\n<span class="caret"></span>' : "") || '<span class="dim">Press Play replay to watch the recorded run.</span>',
        }} />
      </div>
      {step.chart && layers.length > 0 && (
        <div className="layers">
          <h3>{step.chart.title}</h3>
          <div className="bars" role="img" aria-label={`${step.chart.title}: ${layers.filter((l) => l.value! > step.chart!.flagAbove).map((l) => l.label).join(", ") || "no outliers"} above ${step.chart.flagAbove}`}>
            {layers.map((l) => (
              <i key={l.layer} className={l.value! > step.chart!.flagAbove ? "out" : ""} title={`${l.label ?? `layer ${l.layer}`}: ${l.value}`}
                style={{ height: shown.has(l.layer) ? `${Math.max(3, (l.value! / peak) * 100)}%` : 0 }} />
            ))}
          </div>
          {step.chart.xLabels && <div className="bars-x"><span>{step.chart.xLabels[0]}</span><span>{step.chart.xLabels[1]}</span></div>}
          {step.chart.caption && <Html as="p" className="fine" html={inlineMd(step.chart.caption)} />}
        </div>
      )}
      {done && step.summary.length > 0 && <div className="sum">{step.summary.map((s) => <div key={s.label}>{s.label}<b>{s.value}</b></div>)}</div>}
      {step.runYourself && <details><summary>Run it yourself</summary><pre>{step.runYourself}</pre></details>}
    </>
  );
}
