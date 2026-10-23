// Fold reader data: paper-notes parsing and the lesson fold derived from a ready lesson.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";
import { lessonFold, parsePaperNotes, parsePrimer } from "../fold.ts";
import type { Lesson } from "../schema.ts";

const PAPERS: Lesson["papers"] = [{ title: "LLM.int8()", authors: "Dettmers et al.", year: 2022, url: "https://arxiv.org/abs/2208.07339" }];
const NOTES = `---
paper: LLM.int8()
gist: One sentence about it.
---

## The problem {#problem ref="§1"}
> Key sentence one.

Body one.

## Outliers {#outliers}
> Key sentence two
> continues here.

Body two.
`;

describe("paper notes", () => {
  test("parse into a fold document", () => {
    const { doc, issues } = parsePaperNotes(NOTES, "llm-int8", PAPERS);
    assert.deepEqual(issues, []);
    assert.equal(doc!.id, "paper-llm-int8");
    assert.equal(doc!.gist, "One sentence about it.");
    assert.deepEqual(doc!.source, { citation: "Dettmers et al., 2022", url: "https://arxiv.org/abs/2208.07339" });
    assert.deepEqual(doc!.sections.map((s) => [s.id, s.title, s.ref, s.key, s.body]), [
      ["problem", "The problem", "§1", "Key sentence one.", "Body one."],
      ["outliers", "Outliers", undefined, "Key sentence two continues here.", "Body two."],
    ]);
  });
  test("must name one of the lesson's papers", () => {
    const { issues } = parsePaperNotes(NOTES.replace("paper: LLM.int8()", "paper: GPTQ"), "x", PAPERS);
    assert.match(issues[0].message, /must match the title of one of the lesson's papers/);
  });
  test("each section needs an id, a key sentence and notes", () => {
    const bad = NOTES.replace(" {#outliers}", "").replace("> Key sentence one.\n", "").replace("Body two.\n", "");
    const msgs = parsePaperNotes(bad, "x", PAPERS).issues.map((i) => i.message).join("\n");
    assert.match(msgs, /needs an id/);
    assert.match(msgs, /needs a key sentence/);
  });
  test("gist is required and unknown keys are rejected", () => {
    const msgs = parsePaperNotes(NOTES.replace("gist: One sentence about it.", "summary: nope"), "x", PAPERS).issues.map((i) => i.message).join("\n");
    assert.match(msgs, /gist: one sentence is required/);
    assert.match(msgs, /Unknown key "summary"/);
  });
  test("3.1's LLM.int8() notes parse cleanly", () => {
    const src = readFileSync(fileURLToPath(new URL("../../content/lessons/3.1-int8-absmax/papers/llm-int8.md", import.meta.url)), "utf8");
    const { doc, issues } = parsePaperNotes(src, "llm-int8", PAPERS);
    assert.deepEqual(issues, []);
    assert.ok(doc!.sections.length >= 5);
  });
});

describe("primers", () => {
  const PRIMER = `---
gist: What this lesson is about.
minutes: 12
---

## Why {#why}
> Key sentence.

The explanation.
`;
  test("parse into a primer document", () => {
    const { doc, issues } = parsePrimer(PRIMER, "Lesson title");
    assert.deepEqual(issues, []);
    assert.equal(doc!.kind, "primer");
    assert.equal(doc!.id, "primer");
    assert.equal(doc!.title, "Lesson title");
    assert.equal(doc!.minutes, 12);
    assert.deepEqual(doc!.sections.map((s) => [s.id, s.key, s.body]), [["why", "Key sentence.", "The explanation."]]);
  });
  test("need a gist, whole minutes and known keys", () => {
    const msgs = parsePrimer(PRIMER.replace("gist: What this lesson is about.", "paper: X").replace("minutes: 12", "minutes: 1.5"), "T").issues.map((i) => i.message).join("\n");
    assert.match(msgs, /gist: one sentence is required/);
    assert.match(msgs, /minutes: a whole number/);
    assert.match(msgs, /Unknown key "paper"/);
  });
  test("0.0's primer parses cleanly", () => {
    const src = readFileSync(fileURLToPath(new URL("../../content/lessons/0.0-python-bridge-and-environment/primer.md", import.meta.url)), "utf8");
    const { doc, issues } = parsePrimer(src, "Python bridge and environment");
    assert.deepEqual(issues, []);
    assert.ok(doc!.sections.length >= 8);
  });
});

describe("lesson fold", () => {
  const lesson = {
    id: "9.9", title: "T", summary: "The summary.", goal: "Do the thing. Then more.", hints: "- Try X first.", reading: [],
    learn: [
      { type: "concept", id: "c", title: "Concept", body: "First idea. Second idea." },
      { type: "fill_blank", id: "f", title: "Fill it", code: "", tokens: [], answer: [], tips: {}, whyOk: "THE ANSWER", whyWrong: "" },
      { type: "quiz", id: "q", ref: "q", q: "Why?", answer: 1, options: [{ text: "no", why: "a" }, { text: "yes", why: "because" }] },
    ],
    build: [
      { kind: "challenge", id: "ch", title: "Write it", intro: "Write the function.", setup: [{ say: "Here is the stub.", type: "", pause: 1 }],
        hints: { logic: "SECRET LOGIC", tools: "SECRET TOOLS", nudge: "SECRET NUDGE", concept: "SECRET CONCEPT", example: "SECRET EXAMPLE", skeleton: "SECRET ____", solution: "SECRET SOLUTION" }, solution: "SECRET REFERENCE", tests: "", testNames: [], file: "a.py", start: { code: "" }, onSolve: "Done." },
      { kind: "replay", id: "r", title: "Run it", intro: "Watch the run.", recorded: { hardware: "GPU", illustrative: true }, log: [], summary: [], speed: 40 },
      { kind: "metrics", id: "m", title: "Record", intro: "Write numbers down." },
    ],
    doneWhen: [{ id: "a", text: "Did A" }, { id: "b", text: "Did B" }],
  } as unknown as Lesson;
  const fold = lessonFold(lesson);
  test("uses the summary as the gist and keeps the lesson's order", () => {
    assert.equal(fold.gist, "The summary.");
    assert.deepEqual(fold.sections.map((s) => s.title), ["Goal", "Concept", "Fill it", "Check yourself", "Build: Write it", "Build: Run it", "Build: Record", "When you're stuck", "Done when"]);
  });
  test("keys are first sentences", () => {
    assert.equal(fold.sections[0].key, "Do the thing.");
    assert.equal(fold.sections[1].key, "First idea.");
  });
  test("challenge hint ladders and solutions never appear", () => {
    assert.doesNotMatch(JSON.stringify(fold), /SECRET/);
  });
  test("replays are labelled illustrative", () => {
    assert.match(fold.sections.find((s) => s.title === "Build: Run it")!.body!, /^Illustrative recording, not a real run./);
  });
  test("answers wait for Learn", () => {
    const ex = fold.sections.find((s) => s.title === "Fill it")!;
    assert.equal(ex.locked, "learn");
    assert.equal(ex.key, undefined);
    const quiz = fold.sections.find((s) => s.title === "Check yourself")!;
    assert.equal(quiz.locked, "learn");
    assert.match(quiz.body!, /yes.*because/);
    assert.equal(fold.sections.find((s) => s.title === "Concept")!.locked, undefined);
  });
});
