import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { readContent, runtime } from "../cli.ts";
import { compileCourse } from "../compile.ts";
import { CourseOut } from "../schema.ts";

const BASE = readContent();
const RT = runtime();
const L31 = "lessons/3.1-int8-absmax/lesson.md";
const L32 = "lessons/3.2-group-wise-int4/lesson.md";
const STUB22 = "lessons/2.2-kv-cache/lesson.md";

function compile(edit: (f: Record<string, string>) => void = () => {}) {
  const files = { ...BASE };
  edit(files);
  return compileCourse(files, { runtime: RT, now: "2026-01-01T00:00:00Z" });
}
const sub = (f: Record<string, string>, path: string, from: string | RegExp, to: string) => {
  const before = f[path];
  f[path] = before.replace(from, to);
  assert.notEqual(f[path], before, `fixture edit didn't apply: ${String(from)}`);
};
function expectError(edit: (f: Record<string, string>) => void, pattern: RegExp) {
  const { issues, course } = compile(edit);
  const errors = issues.filter((i) => i.level === "error");
  assert.ok(errors.some((i) => pattern.test(i.message)), `expected an error matching ${pattern}, got:\n${errors.map((e) => e.message).join("\n") || "(none)"}`);
  assert.equal(course, undefined, "course.json must not be emitted when there are errors");
}

describe("real content", () => {
  test("compiles with no errors", () => {
    const { issues, course } = compile();
    assert.deepEqual(issues.filter((i) => i.level === "error"), []);
    assert.ok(course);
    assert.ok(CourseOut.safeParse(course).success);
  });
  test("has 39 core lessons, 203 core hours, 2 sidequests and the agreed cut order", () => {
    const { course } = compile();
    assert.equal(course!.totals.coreLessons, 39);
    assert.equal(course!.totals.coreHours, 203);
    assert.equal(course!.lessons.length, 41);
    assert.deepEqual(course!.cutOrder, ["5.3", "2.4", "1.4", "5.1"]);
  });
  test("lesson 3.1 is ready with 8 cards, 4 build steps and 3 Done-when items", () => {
    const l = compile().course!.lessons.find((x) => x.id === "3.1")!;
    assert.equal(l.status, "ready");
    assert.equal(l.learn.length, 8);
    assert.deepEqual(l.build.map((s) => s.kind), ["challenge", "challenge", "replay", "metrics"]);
    assert.equal(l.doneWhen.length, 3);
  });
  test("lesson 3.2 is ready, protected, builds on 3.1, with 8 cards, 4 build steps and 3 Done-when items", () => {
    const l = compile().course!.lessons.find((x) => x.id === "3.2")!;
    assert.equal(l.status, "ready");
    assert.equal(l.protected, true);
    assert.deepEqual(l.prerequisites, ["3.1"]);
    assert.equal(l.learn.length, 8);
    assert.deepEqual(l.build.map((s) => s.kind), ["challenge", "challenge", "replay", "metrics"]);
    assert.equal(l.doneWhen.length, 3);
  });
  test("2.2's KV cache note uses kv_heads", () => {
    const l = compile().course!.lessons.find((x) => x.id === "2.2")!;
    assert.match(l.systemsMetric.note!, /kv_heads/);
    assert.match(l.systemsMetric.note!, /2 × 32 × 8 × 128 × 8192 × 2 bytes = 1 GiB/);
  });
});

describe("prerequisites", () => {
  test("must exist", () => expectError((f) => sub(f, STUB22, '  - "2.1"', '  - "2.9"'), /Prerequisite 2\.9 doesn't exist/));
  test("can't form a cycle", () => expectError((f) => sub(f, "lessons/2.1-naive-generation-loop/lesson.md", '  - "1.3"', '  - "2.2"'), /Prerequisite cycle/));
  test("can't be the lesson itself", () => expectError((f) => sub(f, STUB22, '  - "2.1"', '  - "2.2"'), /lists itself/));
});

describe("cut order", () => {
  test("numbers are unique", () => expectError((f) => sub(f, "lessons/2.4-batched-inference/lesson.md", "cut: 2", "cut: 1"), /Cut order 1 is used by/));
  test("numbers run without gaps", () => expectError((f) => sub(f, "lessons/5.1-vllm-vs-llama-cpp/lesson.md", "cut: 4", "cut: 6"), /without gaps/));
  test("protected lessons are never cut", () => expectError((f) => sub(f, L32, "cut: null", "cut: 5"), /Protected lessons are never cut/));
  test("lessons in a protected phase must be protected", () => expectError((f) => sub(f, L32, "protected: true", "protected: false"), /must be protected: true/));
  test("a never-cut lesson needs a fallback for a cuttable prerequisite", () =>
    expectError((f) => sub(f, L31, /fallbacks:\n {2}"1\.4": .*\n/, ""), /3\.1 needs 1\.4, which is cut #3.*fallbacks/));
  test("fallback keys must be prerequisites", () =>
    expectError((f) => sub(f, L31, 'fallbacks:\n', 'fallbacks:\n  "2.2": not a prerequisite\n'), /fallbacks\.2\.2: 2\.2 isn't a prerequisite/));
});

describe("frontmatter", () => {
  test("file name matches id", () => expectError((f) => { f["lessons/9.9-oops.md"] = f[STUB22]; }, /should start with "2\.2-"|Duplicate lesson id/));
  test("phase must exist", () => expectError((f) => sub(f, STUB22, 'phase: "2"', 'phase: "7"'), /Unknown phase "7"/));
  test("unknown keys are rejected", () => expectError((f) => sub(f, STUB22, "hours: 5", "hours: 5\nhourz: 5"), /Unrecognized key/));
  test("type is one of build | run | research | sidequest", () => expectError((f) => sub(f, STUB22, "type: build", "type: lecture"), /type/));
  test("stubs need no quiz, ready lessons need 2–3", () => {
    expectError((f) => sub(f, L31, /quiz:\n[\s\S]*?\n---\n/, "---\n"), /needs 2–3 quiz questions/);
  });
});

describe("quiz", () => {
  test("answer must be in range", () => expectError((f) => sub(f, L31, "    answer: 1\n", "    answer: 3\n"), /answer 3 is out of range/));
  test("every option needs an explanation", () => expectError((f) => sub(f, L31, /why: Primality has nothing.*\n/, 'why: ""\n'), /why|explanation/));
  test("quiz cards must reference a frontmatter question", () => expectError((f) => sub(f, L31, "ref: why_127", "ref: why_128"), /no quiz question with id "why_128"/));
});

describe("Learn cards", () => {
  test("unknown block types are rejected", () => expectError((f) => sub(f, L31, "type: calc", "type: abacus"), /card:/));
  test("missing required fields are rejected", () => expectError((f) => sub(f, L31, /why_wrong: Each int8 weight.*\n/, ""), /why_wrong/));
  test("fill_blank answers must be tokens", () => expectError((f) => sub(f, L31, "answer: [w.abs().max(), \"127\"]", "answer: [w.abs().max(), \"126\"]"), /answer "126" isn't one of the tokens/));
  test("fill_blank tokens must be listed as often as the answer uses them", () =>
    expectError((f) => { sub(f, L31, "scale = [[0]] / [[1]]", "scale = [[0]] / [[1]] / [[2]]"); sub(f, L31, "answer: [w.abs().max(), \"127\"]", "answer: [w.abs().max(), \"127\", \"127\"]"); }, /uses "127" 2 times but tokens list it 1 time/));
  test("fill_blank slot count must match answers", () => expectError((f) => sub(f, L31, "scale = [[0]] / [[1]]", "scale = [[0]] / 127"), /1 slot\(s\) but 2 answer/));
  test("predict needs exactly one correct option", () => expectError((f) => sub(f, L31, '  - text: "[0, -1, 0]"\n', '  - text: "[0, -1, 0]"\n    correct: true\n'), /exactly one option must be correct/));
  test("slider answer is recomputed", () => expectError((f) => sub(f, L31, "answer: 5\n", "answer: 4\n"), /fewest bits under 0\.05 is 5/));
  test("reorder shuffle must actually shuffle", () => expectError((f) => sub(f, L31, "shuffle: [3, 0, 4, 2, 1]", "shuffle: [0, 1, 2, 3, 4]"), /already in the right order/));
  test("reorder shuffle must be a permutation", () => expectError((f) => sub(f, L31, "shuffle: [3, 0, 4, 2, 1]", "shuffle: [3, 0, 4, 2, 2]"), /permutation/));
  test("decks hold 5–8 cards", () =>
    expectError((f) => {
      for (const id of ["absmax_scale", "predict_round", "round_trip_order", "llama_int8_gb"]) {
        sub(f, L31, new RegExp("```card\\ntype: \\w+\\nid: " + id + "\\n[\\s\\S]*?```\\n\\n"), "");
      }
    }, /Learn needs 5–8 cards including the quiz, found 4/));
  test("prose outside cards is rejected", () => expectError((f) => sub(f, L31, "## Learn\n", "## Learn\n\nSome stray text.\n"), /holds ```card blocks only/));
});

describe("Build challenges", () => {
  test("the setup never types a solution line", () =>
    expectError((f) => sub(f, L31, 'type: "w = torch.tensor([0.42, -1.30, 0.07, 0.88, -0.25, 2.10, -0.61, 0.33])\\n\\n"', 'type: "scale = w.abs().max() / 127\\n"'), /types a solution line: "scale = w\.abs\(\)\.max\(\) \/ 127"/));
  test("a solution line with a trailing comment or odd spacing still counts as typed", () =>
    expectError((f) => { f["lessons/3.1-int8-absmax/dequantize/stub.py"] = f["lessons/3.1-int8-absmax/dequantize/stub.py"].replace("    return ...", "    return q.float()*scale  # ..."); }, /types a solution line/));
  test("hints.solution must cover the whole answer", () =>
    expectError((f) => sub(f, L31, "    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)\non_solve", "on_solve"), /neither in the setup nor in hints\.solution/));
  test("the setup must end on a stub with ... placeholders", () =>
    expectError((f) => { f["lessons/3.1-int8-absmax/dequantize/stub.py"] = "\ndef dequantize(q, scale):\n    return q\n"; }, /\.\.\. placeholders/));
  test("the skeleton needs ____ blanks", () => expectError((f) => sub(f, L31, "return q.____() * _____", "return q.float() * scale"), /skeleton needs ____ blanks/));
  test("hint solution must match the reference file", () => expectError((f) => sub(f, L31, "    return q.float() * scale\non_solve", "    return scale * q.float()\non_solve"), /isn't in dequantize\/solution\.py/));
  test("referenced files must exist", () => expectError((f) => { delete f["lessons/3.1-int8-absmax/quantize_absmax/tests.py"]; }, /file quantize_absmax\/tests\.py not found/));
  test("mistakes must name a real test", () => expectError((f) => sub(f, L31, "fails: round_trip, says: \"chops", "fails: round_tripp, says: \"chops"), /doesn't define/));
  test("start.from must name an earlier challenge", () => expectError((f) => sub(f, L31, "start: { from: quantize_absmax }", "start: { from: dequantize }"), /must name an earlier challenge/));
  test("replay logs must be valid JSONL", () => expectError((f) => { f["lessons/3.1-int8-absmax/replays/quantize-30m.jsonl"] += "{not json\n"; }, /not valid JSON/));
});

describe("Done when", () => {
  test("auto targets must exist", () => expectError((f) => sub(f, L31, "auto=challenge:dequantize", "auto=challenge:dequantise"), /doesn't match anything/));
  test("needs 2–4 checkboxes", () => expectError((f) => sub(f, L31, /- \[ \] Every nn\.Linear.*\n- \[ \] Perplexity.*\n/, ""), /2–4 checkboxes/));
  test("each item needs an id", () => expectError((f) => sub(f, L31, " {#model_int8}", ""), /needs an id/));
});

describe("ready-lesson metrics", () => {
  test("need reference and pass limits", () => expectError((f) => sub(f, L31, "  reference: 18.9\n", ""), /systems_metric: a ready lesson needs/));
  test("compute takes the right number of inputs", () => expectError((f) => sub(f, L31, "compute: pct_change", "compute: value"), /takes 1 input/));
});
