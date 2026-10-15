// On-device progress. No backend: everything lives in localStorage and moves between devices
// as a JSON export that merges record by record (so a phone's Learn work and a laptop's Build
// work combine instead of one overwriting the other). XP and streak are derived, never stored.
import { useSyncExternalStore } from "react";
import type { LiteCourse, LiteLesson } from "./course.ts";

export const KEY = "z2l.progress.v1";
export const REVIEW_DAYS = [1, 3, 7];
export const XP = { card: 5, cardWithHint: 3, learnBonus: 10, challengeFull: 30, hintCost: 5, solutionCap: 10, review: 2 };

export interface ChallengeRec { phase: "intro" | "challenge" | "solved"; t: number; code: string | null; runs: number; hints: number; lastHintRun: number; peek: boolean; xp: number; at: number }
export interface LessonRec {
  learn?: { done: boolean; bestXp: number; correct: number; total: number; at: number };
  session?: { ci: number; gained: number; correct: number; at: number };
  challenges: Record<string, ChallengeRec>;
  checks: Record<string, { v: boolean; at: number }>;
  metrics?: { values: Record<string, number>; at: number };
  markedDone?: { v: boolean; at: number };
}
export interface ReviewRec { lesson: string; qid: string; step: number; due: string; retired: boolean; misses: number; xp: number; at: number }
export interface Progress {
  v: 1;
  startDate: string;
  /** When the learner last set the start date (0 = never). The newer setting wins on merge. */
  startSetAt: number;
  cut: number;
  cutAt: number;
  activeDays: string[];
  lessons: Record<string, LessonRec>;
  review: Record<string, ReviewRec>;
}

// ---------- dates (local calendar days) ----------
export const today = (d = new Date()) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
export const addDays = (day: string, n: number) => { const d = new Date(day + "T12:00:00"); d.setDate(d.getDate() + n); return today(d); };
export const daysBetween = (a: string, b: string) => Math.round((new Date(b + "T12:00:00").getTime() - new Date(a + "T12:00:00").getTime()) / 864e5);

export const fresh = (): Progress => ({ v: 1, startDate: today(), startSetAt: 0, cut: 0, cutAt: 0, activeDays: [], lessons: {}, review: {} });
export const blankChallenge = (): ChallengeRec => ({ phase: "intro", t: 0, code: null, runs: 0, hints: 0, lastHintRun: -1, peek: false, xp: 0, at: 0 });
const blankLesson = (): LessonRec => ({ challenges: {}, checks: {} });

// ---------- store ----------
let cache: Progress | null = null;
const listeners = new Set<() => void>();

function load(): Progress {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) { const p = JSON.parse(raw); if (p && p.v === 1) return { ...fresh(), ...p }; }
  } catch { /* private mode or corrupt: start fresh */ }
  return fresh();
}
export function getProgress(): Progress {
  if (!cache) cache = typeof window === "undefined" ? fresh() : load();
  return cache;
}
function save(p: Progress) {
  cache = p;
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* storage full or blocked; keep in memory */ }
  listeners.forEach((l) => l());
}
export function update(fn: (p: Progress) => void) {
  const next = structuredClone(getProgress());
  fn(next);
  save(next);
}
export function lessonRec(p: Progress, id: string): LessonRec {
  return (p.lessons[id] ??= blankLesson());
}
export function touchDay(p: Progress) {
  const d = today();
  if (!p.activeDays.includes(d)) p.activeDays.push(d);
}
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => { if (e.key === KEY) { cache = load(); listeners.forEach((l) => l()); } });
}
const subscribe = (l: () => void) => { listeners.add(l); return () => listeners.delete(l); };
const serverSnap = fresh();
export function useProgress(): Progress {
  return useSyncExternalStore(subscribe, getProgress, () => serverSnap);
}
/** True after hydration. Islands render a neutral state on the server, then real progress. */
export function useHydrated() {
  return useSyncExternalStore(subscribe, () => true, () => false);
}

// ---------- derived ----------
export function totalXp(p: Progress) {
  let xp = 0;
  for (const l of Object.values(p.lessons)) {
    xp += l.learn?.bestXp ?? 0;
    for (const c of Object.values(l.challenges)) xp += c.xp;
  }
  for (const r of Object.values(p.review)) xp += r.xp;
  return xp;
}
export function streak(p: Progress, now = today()) {
  const days = new Set(p.activeDays);
  let d = days.has(now) ? now : addDays(now, -1);
  let n = 0;
  while (days.has(d)) { n++; d = addDays(d, -1); }
  return n;
}
export function courseWeek(p: Progress, weeks: number, now = today()) {
  return Math.min(weeks, Math.max(1, Math.floor(daysBetween(p.startDate, now) / 7) + 1));
}
export const cutIds = (c: Pick<LiteCourse, "cutOrder">, p: Progress) => c.cutOrder.slice(0, p.cut);

export type LessonState = "done" | "learned" | "started" | "new";
export function checkValue(l: Pick<LiteLesson, "doneWhen">, rec: LessonRec | undefined, id: string) {
  const item = l.doneWhen.find((d) => d.id === id);
  if (!item || !rec) return false;
  if (item.auto) {
    const [kind, target] = item.auto.split(":");
    if (kind === "learn") return !!rec.learn?.done;
    if (kind === "metrics") return !!rec.metrics;
    if (kind === "challenge") return rec.challenges[target]?.phase === "solved";
  }
  return !!rec.checks[id]?.v;
}
export function lessonState(l: LiteLesson, p: Progress): LessonState {
  const rec = p.lessons[l.id];
  if (!rec) return "new";
  if (rec.markedDone?.v) return "done";
  if (l.status === "ready" && rec.learn?.done && l.doneWhen.every((d) => checkValue(l, rec, d.id))) return "done";
  if (rec.learn?.done) return "learned";
  if (rec.session || Object.keys(rec.challenges).length || Object.values(rec.checks).some((c) => c.v)) return "started";
  return "new";
}
export function prereqsMet(c: LiteCourse, p: Progress, l: LiteLesson) {
  const cut = cutIds(c, p);
  const missing = l.prerequisites.filter((id) => !cut.includes(id) && lessonState(c.lessons.find((x) => x.id === id)!, p) !== "done");
  const usingFallback = l.prerequisites.filter((id) => cut.includes(id));
  return { ok: missing.length === 0, missing, usingFallback };
}
/** The next lesson to work on: first core lesson in order that isn't done or cut. */
export function currentLesson(c: LiteCourse, p: Progress) {
  const cut = cutIds(c, p);
  return c.lessons.find((l) => !c.phases.find((ph) => ph.id === l.phase)!.optional && !cut.includes(l.id) && lessonState(l, p) !== "done") ?? null;
}
export function hoursLogged(c: LiteCourse, p: Progress) {
  const core = c.lessons.filter((l) => !c.phases.find((ph) => ph.id === l.phase)!.optional);
  return core.filter((l) => lessonState(l, p) === "done").reduce((a, l) => a + l.hours, 0);
}

// ---------- spaced review: 1, 3, 7 days ----------
export const reviewKey = (lesson: string, qid: string) => `${lesson}/${qid}`;
export function scheduleMiss(p: Progress, lesson: string, qid: string) {
  const k = reviewKey(lesson, qid);
  const prev = p.review[k];
  p.review[k] = { lesson, qid, step: 0, due: addDays(today(), REVIEW_DAYS[0]), retired: false, misses: (prev?.misses ?? 0) + 1, xp: prev?.xp ?? 0, at: Date.now() };
}
export function answerReview(p: Progress, k: string, correct: boolean) {
  const r = p.review[k];
  if (!r) return;
  r.at = Date.now();
  if (!correct) { r.step = 0; r.due = addDays(today(), REVIEW_DAYS[0]); r.misses++; return; }
  r.xp += XP.review;
  if (r.step >= REVIEW_DAYS.length - 1) { r.retired = true; return; }
  r.step++;
  r.due = addDays(today(), REVIEW_DAYS[r.step]);
}
export const dueReviews = (p: Progress, now = today()) => Object.entries(p.review).filter(([, r]) => !r.retired && r.due <= now);

// ---------- export / import ----------
export function exportJson(p: Progress) {
  return JSON.stringify({ app: "zero-to-llm", kind: "progress", v: 1, exportedAt: new Date().toISOString(), progress: p }, null, 2);
}
const isObj = (v: unknown): v is Record<string, any> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown, d = 0) => (typeof v === "number" && Number.isFinite(v) ? v : d);
const day = (v: unknown) => (typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : null);
const stamped = (v: unknown) => (isObj(v) && typeof v.v === "boolean" ? { v: v.v, at: num(v.at) } : undefined);

/** Parse an export and rebuild it field by field, dropping anything malformed. */
export function parseImport(text: string): Progress {
  const data = JSON.parse(text);
  const p = data?.app === "zero-to-llm" || data?.app === "steppe" ? data.progress : data;
  if (!isObj(p) || p.v !== 1 || !isObj(p.lessons) || !isObj(p.review) || !Array.isArray(p.activeDays)) {
    throw new Error("That file isn't a Zero to LLM progress export.");
  }
  const out: Progress = { ...fresh(), startDate: day(p.startDate) ?? today(), startSetAt: num(p.startSetAt), cut: Math.max(0, Math.floor(num(p.cut))), cutAt: num(p.cutAt),
    activeDays: p.activeDays.map(day).filter((d: string | null): d is string => !!d) };
  for (const [id, l] of Object.entries(p.lessons)) {
    if (!isObj(l)) continue;
    const rec: LessonRec = { challenges: {}, checks: {} };
    if (isObj(l.learn)) rec.learn = { done: !!l.learn.done, bestXp: Math.max(0, num(l.learn.bestXp)), correct: num(l.learn.correct), total: num(l.learn.total), at: num(l.learn.at) };
    if (isObj(l.session)) rec.session = { ci: Math.max(0, Math.floor(num(l.session.ci))), gained: Math.max(0, num(l.session.gained)), correct: num(l.session.correct), at: num(l.session.at) };
    for (const [cid, c] of Object.entries(isObj(l.challenges) ? l.challenges : {})) {
      if (!isObj(c) || !["intro", "challenge", "solved"].includes(c.phase)) continue;
      rec.challenges[cid] = { phase: c.phase, t: num(c.t), code: typeof c.code === "string" ? c.code : null, runs: num(c.runs), hints: Math.min(4, Math.max(0, num(c.hints))),
        lastHintRun: num(c.lastHintRun, -1), peek: !!c.peek, xp: Math.min(XP.challengeFull, Math.max(0, num(c.xp))), at: num(c.at) };
    }
    for (const [k, v] of Object.entries(isObj(l.checks) ? l.checks : {})) { const s2 = stamped(v); if (s2) rec.checks[k] = s2; }
    if (isObj(l.metrics) && isObj(l.metrics.values)) {
      const values = Object.fromEntries(Object.entries(l.metrics.values).filter(([, v]) => typeof v === "number" && Number.isFinite(v))) as Record<string, number>;
      rec.metrics = { values, at: num(l.metrics.at) };
    }
    const md = stamped(l.markedDone); if (md) rec.markedDone = md;
    out.lessons[id] = rec;
  }
  for (const [k, r] of Object.entries(p.review)) {
    if (!isObj(r) || typeof r.lesson !== "string" || typeof r.qid !== "string" || !day(r.due)) continue;
    out.review[k] = { lesson: r.lesson, qid: r.qid, step: Math.min(REVIEW_DAYS.length - 1, Math.max(0, Math.floor(num(r.step)))), due: r.due, retired: !!r.retired,
      misses: num(r.misses), xp: Math.max(0, num(r.xp)), at: num(r.at) };
  }
  return out;
}

const RANK = { intro: 0, challenge: 1, solved: 2 } as const;
const newer = <T extends { at: number }>(a: T | undefined, b: T | undefined) => (!a ? b : !b ? a : b.at > a.at ? b : a);

/** Same challenge on two devices: furthest phase and newest code win; hints and runs only ever go up. */
function mergeChallenge(a: ChallengeRec, b: ChallengeRec): ChallengeRec {
  const lead = RANK[a.phase] !== RANK[b.phase] ? (RANK[a.phase] > RANK[b.phase] ? a : b) : newer(a, b)!;
  const solvedXp = [a, b].filter((c) => c.phase === "solved").map((c) => c.xp);
  return {
    ...lead,
    hints: Math.max(a.hints, b.hints),
    runs: Math.max(a.runs, b.runs),
    lastHintRun: Math.max(a.lastHintRun, b.lastHintRun),
    peek: a.peek || b.peek,
    // Never let a merge raise earned XP: if both sides solved, keep the smaller award.
    xp: solvedXp.length ? Math.min(...solvedXp) : 0,
  };
}

/** Merge two progress records. Each record keeps the more advanced or more recent side. */
export function merge(a: Progress, b: Progress): Progress {
  const out: Progress = {
    v: 1,
    startDate: a.startSetAt !== b.startSetAt ? (a.startSetAt > b.startSetAt ? a.startDate : b.startDate) : (a.startDate <= b.startDate ? a.startDate : b.startDate),
    startSetAt: Math.max(a.startSetAt, b.startSetAt),
    cut: b.cutAt > a.cutAt ? b.cut : a.cut,
    cutAt: Math.max(a.cutAt, b.cutAt),
    activeDays: [...new Set([...a.activeDays, ...b.activeDays])].sort(),
    lessons: {},
    review: {},
  };
  for (const id of new Set([...Object.keys(a.lessons), ...Object.keys(b.lessons)])) {
    const x = a.lessons[id] ?? blankLesson();
    const y = b.lessons[id] ?? blankLesson();
    const learn = x.learn && y.learn
      ? { ...newer(x.learn, y.learn)!, done: x.learn.done || y.learn.done, bestXp: Math.max(x.learn.bestXp, y.learn.bestXp) }
      : x.learn ?? y.learn;
    const challenges: Record<string, ChallengeRec> = {};
    for (const cid of new Set([...Object.keys(x.challenges), ...Object.keys(y.challenges)])) {
      const cx = x.challenges[cid], cy = y.challenges[cid];
      challenges[cid] = !cx ? cy : !cy ? cx : mergeChallenge(cx, cy);
    }
    const checks: LessonRec["checks"] = {};
    for (const k of new Set([...Object.keys(x.checks), ...Object.keys(y.checks)])) checks[k] = newer(x.checks[k], y.checks[k])!;
    // A Learn session that ended before the deck was finished elsewhere is stale.
    let session = newer(x.session, y.session);
    if (session && learn?.done && learn.at >= session.at) session = undefined;
    out.lessons[id] = { learn, session, challenges, checks, metrics: newer(x.metrics, y.metrics), markedDone: newer(x.markedDone, y.markedDone) };
    for (const key of Object.keys(out.lessons[id]) as (keyof LessonRec)[]) if (out.lessons[id][key] === undefined) delete out.lessons[id][key];
  }
  for (const k of new Set([...Object.keys(a.review), ...Object.keys(b.review)])) {
    const r = newer(a.review[k], b.review[k])!;
    out.review[k] = { ...r, xp: Math.max(a.review[k]?.xp ?? 0, b.review[k]?.xp ?? 0) };
  }
  return out;
}
export function replaceAll(p: Progress) { save(p); }
export function resetAll() { save(fresh()); }
