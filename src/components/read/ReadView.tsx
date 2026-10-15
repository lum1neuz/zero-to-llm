import { useEffect, useState } from "react";
import type { FoldDoc } from "../../../pipeline/schema.ts";
import { useProgress } from "../../lib/progress.ts";
import { FoldReader } from "./FoldReader.tsx";

interface Props { lesson: { id: string; title: string; status: string; crumb: string }; reading: FoldDoc[] }

export default function ReadView({ lesson, reading }: Props) {
  const p = useProgress();
  const learnDone = !!p.lessons[lesson.id]?.learn?.done;
  const fromHash = () => (typeof location !== "undefined" ? reading.findIndex((d) => `#${d.id}` === location.hash) : -1);
  const [tab, setTab] = useState(() => Math.max(0, fromHash()));
  useEffect(() => {
    const onHash = () => { const i = fromHash(); if (i >= 0) setTab(i); };
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const doc = reading[tab];
  const pick = (i: number) => { setTab(i); history.replaceState(null, "", `#${reading[i].id}`); };

  return (
    <div className="read">
      <div className="crumbs"><a href="/">{lesson.crumb}</a> › {lesson.id}</div>
      <h1 className="page-title" style={{ margin: "4px 0 12px" }}>{lesson.title}</h1>
      {reading.length === 0 ? (
        <div className="empty">This lesson isn't written yet, so there's nothing to read here.</div>
      ) : (
        <>
          {reading.length > 1 && (
            <div className="read-tabs" role="tablist" aria-label="Documents">
              {reading.map((d, i) => (
                <button key={d.id} role="tab" aria-selected={i === tab} className={i === tab ? "on" : ""} onClick={() => pick(i)}>
                  {d.kind === "lesson" ? "The lesson" : d.title}
                </button>
              ))}
            </div>
          )}
          <FoldReader doc={doc} scope={lesson.id} learnDone={learnDone} learnHref={`/learn/${lesson.id}/`} />
          <div className="row-gap" style={{ marginTop: 18 }}>
            {lesson.status === "ready" && <a className="btn" href={`/learn/${lesson.id}/`}>Learn cards</a>}
            {lesson.status === "ready" && <a className="btn" href={`/build/${lesson.id}/`}>Open Build</a>}
          </div>
        </>
      )}
    </div>
  );
}
