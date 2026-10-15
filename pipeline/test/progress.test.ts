// Progress store: merge (phone + laptop sync), import validation, spaced review.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
  addDays, answerReview, blankChallenge, exportJson, fresh, merge, parseImport, scheduleMiss, today, totalXp, type Progress,
} from "../../src/lib/progress.ts";

const base = (): Progress => ({ ...fresh(), startDate: "2026-08-24" });

describe("merge", () => {
  test("Learn from one device and Build from another both survive", () => {
    const phone = base();
    phone.lessons["3.1"] = { challenges: {}, checks: {}, learn: { done: true, bestXp: 38, correct: 6, total: 7, at: 10 } };
    const laptop = base();
    laptop.lessons["3.1"] = { challenges: { quantize_absmax: { ...blankChallenge(), phase: "solved", xp: 30, at: 20 } }, checks: {} };
    const m = merge(laptop, phone);
    assert.equal(m.lessons["3.1"].learn?.done, true);
    assert.equal(m.lessons["3.1"].challenges.quantize_absmax.phase, "solved");
    assert.equal(totalXp(m), 68);
  });
  test("hints opened on one device aren't lost to a newer edit on the other", () => {
    const a = base(), b = base();
    a.lessons["3.1"] = { challenges: { q: { ...blankChallenge(), phase: "challenge", hints: 3, lastHintRun: 4, runs: 4, at: 10 } }, checks: {} };
    b.lessons["3.1"] = { challenges: { q: { ...blankChallenge(), phase: "challenge", hints: 0, runs: 1, code: "new", at: 20 } }, checks: {} };
    const c = merge(a, b).lessons["3.1"].challenges.q;
    assert.equal(c.hints, 3);
    assert.equal(c.code, "new");
    assert.equal(c.runs, 4);
  });
  test("a solved challenge beats a newer unsolved one, and XP never goes up by merging", () => {
    const a = base(), b = base();
    a.lessons["3.1"] = { challenges: { q: { ...blankChallenge(), phase: "solved", hints: 2, xp: 20, at: 10 } }, checks: {} };
    b.lessons["3.1"] = { challenges: { q: { ...blankChallenge(), phase: "solved", hints: 0, xp: 30, at: 20 } }, checks: {} };
    assert.equal(merge(a, b).lessons["3.1"].challenges.q.xp, 20);
    b.lessons["3.1"].challenges.q = { ...blankChallenge(), phase: "intro", at: 99 };
    assert.equal(merge(a, b).lessons["3.1"].challenges.q.phase, "solved");
  });
  test("an old unfinished Learn session doesn't come back after the deck was finished", () => {
    const old = base(), now = base();
    old.lessons["3.1"] = { challenges: {}, checks: {}, session: { ci: 3, gained: 15, correct: 3, at: 5 } };
    now.lessons["3.1"] = { challenges: {}, checks: {}, learn: { done: true, bestXp: 38, correct: 6, total: 7, at: 10 } };
    assert.equal(merge(now, old).lessons["3.1"].session, undefined);
  });
  test("a start date the learner set wins over a default one", () => {
    const a = { ...base(), startDate: "2026-08-01", startSetAt: 0 };
    const b = { ...base(), startDate: "2026-09-01", startSetAt: 500 };
    assert.equal(merge(a, b).startDate, "2026-09-01");
    assert.equal(merge(b, a).startDate, "2026-09-01");
  });
  test("active days are unioned", () => {
    const a = { ...base(), activeDays: ["2026-10-01", "2026-10-02"] }, b = { ...base(), activeDays: ["2026-10-02", "2026-10-03"] };
    assert.deepEqual(merge(a, b).activeDays, ["2026-10-01", "2026-10-02", "2026-10-03"]);
  });
});

describe("import", () => {
  test("round-trips an export", () => {
    const p = base();
    p.lessons["3.1"] = { challenges: { q: { ...blankChallenge(), phase: "solved", xp: 25, at: 1 } }, checks: { m: { v: true, at: 1 } } };
    assert.deepEqual(parseImport(exportJson(p)).lessons, p.lessons);
  });
  test("rejects files that aren't exports", () => {
    assert.throws(() => parseImport('{"hello": 1}'), /isn't a Zero to LLM progress export/);
    assert.throws(() => parseImport('{"v":1,"lessons":null,"review":{},"activeDays":[]}'), /isn't a Zero to LLM progress export/);
  });
  test("drops malformed records instead of crashing later", () => {
    const p = parseImport(JSON.stringify({ v: 1, activeDays: ["2026-10-01", 5], review: { x: { lesson: 1 } },
      lessons: { "3.1": { challenges: null, checks: { a: "yes" }, learn: { done: true, bestXp: "99" } }, "3.2": 7 } }));
    assert.deepEqual(p.activeDays, ["2026-10-01"]);
    assert.deepEqual(p.review, {});
    assert.equal(p.lessons["3.1"].learn?.bestXp, 0);
    assert.deepEqual(p.lessons["3.1"].challenges, {});
    assert.equal(p.lessons["3.2"], undefined);
    assert.equal(totalXp(p), 0);
  });
});

describe("spaced review", () => {
  test("a miss comes back after 1 day, then 3, then 7, then retires", () => {
    const p = base();
    scheduleMiss(p, "3.1", "q");
    const k = "3.1/q";
    assert.equal(p.review[k].due, addDays(today(), 1));
    answerReview(p, k, true);
    assert.equal(p.review[k].due, addDays(today(), 3));
    answerReview(p, k, true);
    assert.equal(p.review[k].due, addDays(today(), 7));
    answerReview(p, k, true);
    assert.equal(p.review[k].retired, true);
    assert.equal(p.review[k].xp, 6);
  });
  test("a wrong answer in review resets to 1 day", () => {
    const p = base();
    scheduleMiss(p, "3.1", "q");
    answerReview(p, "3.1/q", true);
    answerReview(p, "3.1/q", false);
    assert.equal(p.review["3.1/q"].step, 0);
    assert.equal(p.review["3.1/q"].due, addDays(today(), 1));
  });
});
