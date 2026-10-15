import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { checkChart, extractCharts } from "../chart.ts";
import { readContent, runtime } from "../cli.ts";
import { compileCourse } from "../compile.ts";
import { CourseOut } from "../schema.ts";

const LINE = `kind: line
title: Surprise against chance
x: { label: chance p, values: [0.1, 0.5, 1] }
y: { label: surprise, min: 0, max: 3 }
series:
  - { name: surprise, values: [2.3, 0.69, 0] }
caption: Rare things surprise you more.`;

function problems(yaml: string) {
  const { chart, issues } = checkChart(yaml);
  assert.equal(chart, undefined, "expected the chart to be rejected");
  return issues.join("\n");
}

describe("checkChart", () => {
  test("accepts a valid chart and defaults illustrative to false", () => {
    const { chart, issues } = checkChart(LINE);
    assert.deepEqual(issues, []);
    assert.equal(chart!.kind, "line");
    assert.equal(chart!.illustrative, false);
  });
  test("rejects an unknown kind", () => assert.match(problems(LINE.replace("kind: line", "kind: pie")), /kind must be one of bar, line, scatter/));
  test("series length must match x", () => assert.match(problems(LINE.replace("[2.3, 0.69, 0]", "[2.3, 0.69]")), /series "surprise" has 2 values but x has 3/));
  test("values must be finite numbers", () => {
    assert.match(problems(LINE.replace("[2.3, 0.69, 0]", "[.inf, 0.69, 0]")), /series\.0\.values\.0/);
    assert.match(problems(LINE.replace("[2.3, 0.69, 0]", "[.nan, 0.69, 0]")), /series\.0\.values\.0/);
    assert.match(problems(LINE.replace("[2.3, 0.69, 0]", "[high, 0.69, 0]")), /series\.0\.values\.0/);
  });
  test("x can't mix numbers and labels", () => assert.match(problems(LINE.replace("[0.1, 0.5, 1]", "[0.1, half, 1]")), /not a mix/));
  test("a line's numeric x goes up", () => assert.match(problems(LINE.replace("[0.1, 0.5, 1]", "[0.1, 1, 0.5]")), /go up from left to right/));
  test("a scatter needs numbers on x", () => assert.match(problems(LINE.replace("kind: line", "kind: scatter").replace("[0.1, 0.5, 1]", "[a, b, c]")), /scatter chart needs numbers/));
  test("values must fit the y range", () => assert.match(problems(LINE.replace("max: 3", "max: 2")), /y\.max is 2 but a value is 2\.3/));
  test("y.min must be below y.max", () => assert.match(problems(LINE.replace("min: 0, max: 3", "min: 3, max: 3")), /min \(3\) must be below max/));
  test("bars start at zero", () => assert.match(problems(LINE.replace("kind: line", "kind: bar").replace("min: 0", "min: 0.5").replace("0.69, 0]", "0.69, 0.6]")), /must include 0/));
  test("at most three series, with unique names", () => {
    const s = "  - { name: surprise, values: [2.3, 0.69, 0] }";
    assert.match(problems(LINE.replace(s, `${s}\n${s}`)), /appears twice/);
    assert.match(problems(LINE.replace(s, [1, 2, 3, 4].map((i) => `  - { name: s${i}, values: [1, 1, 1] }`).join("\n"))), /series/);
  });
  test("highlight must point at an x and a series", () => {
    assert.deepEqual(checkChart(`${LINE}\nhighlight: { x: 0.5, label: a coin flip }`).issues, []);
    assert.match(problems(`${LINE}\nhighlight: { x: 0.3, label: nope }`), /highlight\.x 0\.3 isn't one of x\.values/);
    assert.match(problems(`${LINE}\nhighlight: { x: 0.5, series: other, label: nope }`), /highlight\.series "other"/);
  });
  test("threshold must sit inside the y range", () => {
    assert.deepEqual(checkChart(`${LINE}\nthreshold: { y: 1, label: pass limit }`).issues, []);
    assert.match(problems(`${LINE}\nthreshold: { y: 4, label: pass limit }`), /threshold\.y \(4\) is above y\.max/);
  });
  test("caption is one sentence", () => {
    assert.match(problems(LINE.replace("Rare things surprise you more.", "Rare things surprise you more")), /ending in \. ! or \?/);
    assert.match(problems(LINE.replace("Rare things surprise you more.", "Rare things surprise. Common ones don't.")), /one sentence/);
  });
  test("unknown keys and missing captions are errors", () => {
    assert.match(problems(`${LINE}\ncolour: red`), /colour/);
    assert.match(problems(LINE.replace(/\ncaption:.*/, "")), /caption/);
  });
});

describe("extractCharts", () => {
  test("replaces the fence with a marker and reports lines", () => {
    const md = `Before.\n\n\`\`\`chart\n${LINE}\n\`\`\`\n\nAfter.`;
    const { body, charts, issues } = extractCharts(md, 10);
    assert.deepEqual(issues, []);
    assert.equal(charts.length, 1);
    assert.equal(body, "Before.\n\n<!-- chart:0 -->\n\nAfter.");
    const bad = extractCharts(md.replace("kind: line", "kind: pie"), 10);
    assert.equal(bad.issues[0].line, 12);
    assert.match(bad.issues[0].message, /^chart: kind/);
  });
  test("leaves chart syntax inside another fence alone", () => {
    const md = `\`\`\`\`md\n\`\`\`chart\nkind: pie\n\`\`\`\n\`\`\`\``;
    const { body, charts, issues } = extractCharts(md);
    assert.deepEqual(issues, []);
    assert.equal(charts.length, 0);
    assert.equal(body, md);
  });
  test("an unclosed chart fence is an error", () => assert.match(extractCharts("```chart\nkind: bar").issues[0].message, /unclosed/));
});

describe("charts in content", () => {
  const BASE = readContent();
  const RT = runtime();
  const PRIMER = "lessons/3.1-int8-absmax/primer.md";
  const LESSON = "lessons/3.1-int8-absmax/lesson.md";
  const fence = (indent = "") => ["```chart", ...LINE.split("\n"), "```"].map((l) => indent + l).join("\n");
  const compile = (edit: (f: Record<string, string>) => void) => {
    const files = { ...BASE };
    edit(files);
    return compileCourse(files, { runtime: RT, now: "2026-01-01T00:00:00Z" });
  };

  test("a primer chart lands in its fold section as data", () => {
    const { issues, course } = compile((f) => { f[PRIMER] = f[PRIMER].replace("Phase 3 builds this up step by step:", `${fence()}\n\nPhase 3 builds this up step by step:`); });
    assert.deepEqual(issues.filter((i) => i.level === "error"), []);
    assert.ok(CourseOut.safeParse(course).success);
    const primer = course!.lessons.find((l) => l.id === "3.1")!.reading.find((d) => d.kind === "primer")!;
    const why = primer.sections.find((s) => s.id === "why")!;
    assert.equal(why.charts?.length, 1);
    assert.match(why.body!, /<!-- chart:0 -->/);
    assert.doesNotMatch(why.body!, /```chart/);
  });
  test("a broken primer chart is reported at its file line", () => {
    const { issues, course } = compile((f) => { f[PRIMER] = f[PRIMER].replace("Phase 3 builds this up step by step:", `${fence().replace("kind: line", "kind: pie")}\n\nPhase 3 builds this up step by step:`); });
    assert.equal(course, undefined);
    const e = issues.find((i) => i.file === PRIMER && /chart: kind/.test(i.message))!;
    assert.ok(e, "expected a chart error in the primer");
    const at = BASE[PRIMER].split("\n").findIndex((l) => l.startsWith("Phase 3 builds")) + 1;
    assert.equal(e.line, at);
  });
  test("a concept card chart is checked and copied to the card and the lesson fold", () => {
    const { issues, course } = compile((f) => { f[LESSON] = f[LESSON].replace("body: |\n  int8 stores each weight", `body: |\n${fence("  ")}\n\n  int8 stores each weight`); });
    assert.deepEqual(issues.filter((i) => i.level === "error"), []);
    const l = course!.lessons.find((x) => x.id === "3.1")!;
    const card = l.learn.find((c) => c.id === "why_bother")!;
    assert.ok(card.type === "concept" && card.charts?.length === 1 && card.body.startsWith("<!-- chart:0 -->"));
    const sec = l.reading.find((d) => d.kind === "lesson")!.sections.find((s) => s.id === "card-why_bother")!;
    assert.equal(sec.charts?.length, 1);
    assert.match(sec.key!, /^int8 stores each weight/);
  });
  test("a broken concept card chart is an error", () => {
    const { issues, course } = compile((f) => { f[LESSON] = f[LESSON].replace("body: |\n  int8 stores each weight", `body: |\n${fence("  ").replace("[2.3, 0.69, 0]", "[2.3]")}\n\n  int8 stores each weight`); });
    assert.equal(course, undefined);
    assert.ok(issues.some((i) => i.file === LESSON && /card "why_bother": chart: series "surprise" has 1 values/.test(i.message)));
  });
});
