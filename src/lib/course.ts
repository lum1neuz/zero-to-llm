// Course data helpers. Pages import course.json at build time and pass slices to islands.
import courseJson from "../../generated/course.json";
import type { Card, Course, Lesson } from "../../pipeline/schema.ts";

export type { Card, Course, Lesson };
export const course = courseJson as unknown as Course;

/** Everything the home tabs need: no card bodies, challenge sources or runtime. */
export type LiteLesson = Omit<Lesson, "learn" | "build" | "goal" | "hints" | "reading"> & { cards: number; steps: number; stepTitles: string[]; paperNotes: number; primer: { minutes: number } | null };
export interface LiteCourse { title: string; weeks: number; phases: Course["phases"]; cutOrder: string[]; totals: Course["totals"]; lessons: LiteLesson[] }

/** Quiz questions are only needed by Review; other pages leave them out to stay small. */
export function lite(c: Course = course, opts: { quiz?: boolean } = {}): LiteCourse {
  return {
    title: c.title, weeks: c.weeks, phases: c.phases, cutOrder: c.cutOrder, totals: c.totals,
    lessons: c.lessons.map(({ learn, build, goal, hints, reading, ...l }) => ({ ...l, quiz: opts.quiz ? l.quiz : [], cards: learn.length, steps: build.length, stepTitles: build.map((s) => s.title), paperNotes: reading.filter((d) => d.kind === "paper").length,
      primer: reading.some((d) => d.kind === "primer") ? { minutes: reading.find((d) => d.kind === "primer")!.minutes ?? 10 } : null })),
  };
}

export const isOptionalPhase = (c: Pick<LiteCourse, "phases">, phaseId: string) => !!c.phases.find((p) => p.id === phaseId)?.optional;
export const phaseOf = (c: Pick<LiteCourse, "phases">, phaseId: string) => c.phases.find((p) => p.id === phaseId)!;
export const phaseLabel = (p: { id: string; title: string; optional?: boolean }) => (p.optional || !/^\d/.test(p.id) ? p.title : `Phase ${p.id} · ${p.title}`);
export const weeksLabel = (w: [number, number] | null, short = false) =>
  !w ? (short ? "any time" : "Any time") : w[0] === w[1] ? `${short ? "wk" : "Week"} ${w[0]}` : `${short ? "wk" : "Weeks"} ${w[0]}–${w[1]}`;
export const learnMinutes = (cards: number) => Math.max(3, Math.round(cards * 0.75));
