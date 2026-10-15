import type { ReactNode } from "react";
import type { LiteCourse } from "../../lib/course.ts";
import { courseWeek, dueReviews, streak, totalXp, useProgress } from "../../lib/progress.ts";
import { I } from "../ui.tsx";

export type Tab = "path" | "schedule" | "review" | "me";
const TABS: [Tab, string, string, keyof typeof I][] = [
  ["path", "Path", "/", "path"], ["schedule", "Schedule", "/schedule/", "plan"], ["review", "Review", "/review/", "review"], ["me", "Me", "/me/", "me"],
];
const TITLES: Record<Tab, string> = { path: "Your path", schedule: "Schedule", review: "Review", me: "Progress" };

export function Shell({ course, tab, children }: { course: LiteCourse; tab: Tab; children: ReactNode }) {
  const p = useProgress();
  const due = dueReviews(p).length;
  const label = (t: Tab, l: string) => (t === "review" && due ? `${l} (${due})` : l);
  return (
    <div className="shell">
      <header className="appbar">
        <div className="appbar-in">
          <a className="brand" href="/"><b>{course.title}</b><span>AI course</span></a>
          <nav className="topnav" aria-label="App sections">
            {TABS.map(([t, l, href]) => <a key={t} href={href} aria-current={tab === t ? "page" : undefined}>{label(t, l)}</a>)}
          </nav>
          <span className="sp" />
          <div className="pills">
            <span className="pill gold" title="Day streak">{I.flame} {streak(p)}</span>
            <span className="pill" title="Experience points">{totalXp(p).toLocaleString()} XP</span>
          </div>
        </div>
      </header>
      <main className="page">
        <div className="page-h">
          <div>
            <div className="eyebrow-s">Week {courseWeek(p, course.weeks)} of {course.weeks}</div>
            <h1 className="page-title">{TITLES[tab]}</h1>
          </div>
        </div>
        {children}
      </main>
      <nav className="tabbar" aria-label="App sections">
        {TABS.map(([t, l, href, icon]) => <a key={t} href={href} aria-current={tab === t ? "page" : undefined}>{I[icon]}{label(t, l)}</a>)}
      </nav>
    </div>
  );
}
