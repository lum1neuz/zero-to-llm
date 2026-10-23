// Validate every lesson and assemble course.json. Pure: takes a map of content files
// (POSIX paths relative to content/) so tests can feed it fixtures without touching disk.
import { parse as parseYaml } from "yaml";
import type { z } from "zod";
import { fewestBitsUnder } from "../src/lib/quant.ts";
import { extractCharts } from "./chart.ts";
import { lessonFold, parsePaperNotes, parsePrimer } from "./fold.ts";
import { extractFences, parseDoneWhen, splitFrontmatter, splitHeadings, testNamesFrom } from "./parse.ts";
import {
  CardSrc, ChallengeSrc, CourseOut, CourseSrc, Frontmatter, MetricsStepSrc, ReplayLine, ReplaySrc,
  type Card, type Course, type FrontmatterT, type Lesson, type Step,
} from "./schema.ts";

export interface Issue { level: "error" | "warn"; file: string; line?: number; message: string }
export interface CompileOptions { runtime: { pyodide: string; torch: string; harness: string; hash: string }; now?: string }

const BODY_SECTIONS = ["Goal", "Learn", "Build steps", "Hints", "Done when"];
const STEP_FENCES = ["challenge", "replay", "metrics"];

interface Parsed {
  file: string;
  dir: string;
  fm: FrontmatterT;
  body: string;
  bodyLine: number;
}

export function compileCourse(files: Record<string, string>, opts: CompileOptions) {
  const issues: Issue[] = [];
  const err = (file: string, message: string, line?: number) => issues.push({ level: "error", file, line, message });
  const warn = (file: string, message: string, line?: number) => issues.push({ level: "warn", file, line, message });
  const zodIssues = (file: string, prefix: string, e: z.ZodError, line?: number) =>
    e.issues.forEach((i) => err(file, `${prefix}${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`, line));

  // ---------- course.yaml ----------
  const courseRaw = files["course.yaml"];
  if (!courseRaw) return { issues: [{ level: "error", file: "course.yaml", message: "Missing course.yaml" } as Issue] };
  const courseParsed = CourseSrc.safeParse(safeYaml(courseRaw, "course.yaml", err));
  if (!courseParsed.success) { zodIssues("course.yaml", "", courseParsed.error); return { issues }; }
  const courseSrc = courseParsed.data;
  const phaseIds = courseSrc.phases.map((p) => p.id);
  const phaseById = new Map(courseSrc.phases.map((p) => [p.id, p]));

  // ---------- frontmatter ----------
  const parsed: Parsed[] = [];
  for (const file of Object.keys(files).sort()) {
    const m = /^lessons\/(?:([^/]+)\.md|([^/]+)\/lesson\.md)$/.exec(file);
    if (!m) continue;
    const base = m[1] ?? m[2];
    const dir = m[2] ? `lessons/${m[2]}/` : "lessons/";
    const split = splitFrontmatter(files[file]);
    if (!split) { err(file, "Lesson files start with a --- frontmatter block"); continue; }
    const fmRes = Frontmatter.safeParse(safeYaml(split.fm, file, err));
    if (!fmRes.success) { zodIssues(file, "frontmatter ", fmRes.error, 1); continue; }
    const fm = fmRes.data;
    if (!base.startsWith(fm.id + "-")) err(file, `File name should start with "${fm.id}-" to match the lesson id`);
    parsed.push({ file, dir, fm, body: split.body, bodyLine: split.bodyLine });
  }

  const byId = new Map<string, Parsed>();
  for (const p of parsed) {
    if (byId.has(p.fm.id)) err(p.file, `Duplicate lesson id ${p.fm.id} (also in ${byId.get(p.fm.id)!.file})`);
    else byId.set(p.fm.id, p);
  }

  // ---------- cross-lesson rules ----------
  for (const p of parsed) {
    const { fm, file } = p;
    const phase = phaseById.get(fm.phase);
    if (!phase) { err(file, `Unknown phase "${fm.phase}". Phases: ${phaseIds.join(", ")}`); continue; }
    const expectedPhase = /^[SW]/.test(fm.id) ? fm.id[0] : fm.id.split(".")[0];
    if (expectedPhase !== fm.phase) err(file, `Lesson ${fm.id} belongs in phase ${expectedPhase}, not ${fm.phase}`);
    if (phase.protected && !fm.protected) err(file, `Phase ${phase.id} is protected, so lesson ${fm.id} must be protected: true`);
    if (fm.protected && fm.cut !== null) err(file, `Protected lessons are never cut, but ${fm.id} has cut: ${fm.cut}`);
    if ((phase.optional || fm.type === "sidequest") && fm.cut !== null) err(file, `Optional lessons aren't in the cut order (${fm.id} has cut: ${fm.cut})`);
    if (fm.type === "sidequest" !== phase.optional) err(file, `type: sidequest goes with optional phases only`);
    for (const pre of fm.prerequisites) {
      if (pre === fm.id) err(file, `${fm.id} lists itself as a prerequisite`);
      else if (!byId.has(pre)) err(file, `Prerequisite ${pre} doesn't exist`);
    }
    for (const key of Object.keys(fm.fallbacks)) {
      if (!fm.prerequisites.includes(key)) err(file, `fallbacks.${key}: ${key} isn't a prerequisite of ${fm.id}`);
    }
    // A lesson that stays when you fall behind can't hard-depend on one that gets cut.
    if (fm.cut === null) {
      for (const pre of fm.prerequisites) {
        const dep = byId.get(pre);
        if (dep && dep.fm.cut !== null && !fm.fallbacks[pre]) {
          err(file, `${fm.id} needs ${pre}, which is cut #${dep.fm.cut} when you're behind. Add fallbacks: { "${pre}": "what to use instead" }`);
        }
      }
    }
  }
  // cycles
  const state = new Map<string, 1 | 2>();
  const visit = (id: string, path: string[]) => {
    if (state.get(id) === 2) return;
    if (state.get(id) === 1) { err(byId.get(id)!.file, `Prerequisite cycle: ${[...path.slice(path.indexOf(id)), id].join(" → ")}`); return; }
    state.set(id, 1);
    for (const pre of byId.get(id)?.fm.prerequisites ?? []) if (byId.has(pre)) visit(pre, [...path, id]);
    state.set(id, 2);
  };
  for (const id of byId.keys()) visit(id, []);
  // cut order
  const cut = parsed.filter((p) => p.fm.cut !== null).sort((a, b) => a.fm.cut! - b.fm.cut!);
  cut.forEach((p, i) => {
    if (cut.filter((q) => q.fm.cut === p.fm.cut).length > 1 && cut.findIndex((q) => q.fm.cut === p.fm.cut) === i) {
      err(p.file, `Cut order ${p.fm.cut} is used by ${cut.filter((q) => q.fm.cut === p.fm.cut).map((q) => q.fm.id).join(" and ")}. Each cut number must be unique`);
    }
  });
  const nums = [...new Set(cut.map((p) => p.fm.cut!))];
  nums.forEach((n, i) => { if (n !== i + 1) err(cut.find((p) => p.fm.cut === n)!.file, `Cut order must run 1, 2, 3… without gaps; found ${nums.join(", ")}`); });

  // ---------- per-lesson bodies ----------
  const lessons: Lesson[] = [];
  for (const p of parsed) {
    const out = compileLesson(p, files, { err, warn });
    if (out) lessons.push(out);
  }
  if (issues.some((i) => i.level === "error")) return { issues };

  // order lessons: phase order, then numeric
  const key = (id: string) => (/^[SW]/.test(id) ? Number(id.slice(1)) : Number(id.split(".")[1]));
  lessons.sort((a, b) => phaseIds.indexOf(a.phase) - phaseIds.indexOf(b.phase) || key(a.id) - key(b.id));
  lessons.forEach((l, i) => (l.order = i));

  const core = lessons.filter((l) => !phaseById.get(l.phase)!.optional);
  const course: Course = {
    schemaVersion: 1,
    generatedAt: opts.now ?? new Date().toISOString(),
    title: courseSrc.title,
    subtitle: courseSrc.subtitle,
    weeks: courseSrc.weeks,
    phases: courseSrc.phases.map((ph) => ({ ...ph })),
    cutOrder: cut.map((p) => p.fm.id),
    totals: {
      coreLessons: core.length,
      coreHours: core.reduce((a, l) => a + l.hours, 0),
      sidequestHours: lessons.filter((l) => !core.includes(l)).reduce((a, l) => a + l.hours, 0),
    },
    lessons,
    runtime: opts.runtime,
  };
  const check = CourseOut.safeParse(course);
  if (!check.success) zodIssues("course.json", "internal: emitted course.json doesn't match its schema: ", check.error);
  return { issues, course: check.success ? course : undefined };
}

function safeYaml(src: string, file: string, err: (f: string, m: string, l?: number) => void) {
  try {
    return parseYaml(src);
  } catch (e) {
    err(file, `YAML: ${(e as Error).message.split("\n")[0]}`);
    return undefined;
  }
}

type Report = { err: (file: string, msg: string, line?: number) => void; warn: (file: string, msg: string, line?: number) => void };

function metricOut(m: FrontmatterT["quality_metric"]) {
  return { id: m.id, label: m.label, unit: m.unit, note: m.note, inputs: m.inputs, compute: m.compute, reference: m.reference, pass: m.pass };
}

function compileLesson(p: Parsed, files: Record<string, string>, { err, warn }: Report): Lesson | null {
  const { fm, file } = p;
  const base: Lesson = {
    id: fm.id, title: fm.title, summary: fm.summary, phase: fm.phase, order: 0, hours: fm.hours, type: fm.type, status: fm.status,
    protected: fm.protected, cut: fm.cut, prerequisites: fm.prerequisites, fallbacks: fm.fallbacks,
    papers: fm.papers, qualityMetric: metricOut(fm.quality_metric), systemsMetric: metricOut(fm.systems_metric),
    quiz: fm.quiz ?? [], learn: [], build: [], doneWhen: [], reading: [],
  };

  // quiz (validated whenever present; required when ready)
  const quiz = fm.quiz ?? [];
  if (fm.status === "ready" && (quiz.length < 2 || quiz.length > 3)) err(file, `A ready lesson needs 2–3 quiz questions, found ${quiz.length}`, 1);
  const qids = new Set<string>();
  quiz.forEach((q, i) => {
    if (qids.has(q.id)) err(file, `quiz[${i}]: duplicate quiz id "${q.id}"`, 1);
    qids.add(q.id);
    if (q.options.length < 2 || q.options.length > 5) err(file, `quiz "${q.id}": needs 2–5 options, found ${q.options.length}`, 1);
    if (q.answer < 0 || q.answer >= q.options.length) err(file, `quiz "${q.id}": answer ${q.answer} is out of range (options 0–${q.options.length - 1})`, 1);
    q.options.forEach((o, j) => { if (!o.why.trim()) err(file, `quiz "${q.id}" option ${j}: every option needs an explanation`, 1); });
  });

  if (fm.status === "ready" && fm.type !== "intro") {
    for (const [name, m] of [["quality_metric", fm.quality_metric], ["systems_metric", fm.systems_metric]] as const) {
      if (!m.inputs || !m.compute || m.reference === undefined || !m.pass || (m.pass.max === undefined && m.pass.min === undefined)) {
        err(file, `${name}: a ready lesson needs inputs, compute, reference and pass (max or min)`, 1);
        continue;
      }
      const need = m.compute === "value" ? 1 : 2;
      if (m.inputs.length !== need) err(file, `${name}: compute "${m.compute}" takes ${need} input${need > 1 ? "s" : ""}, found ${m.inputs.length}`, 1);
    }
  }

  const bodyLines = p.body.split("\n");
  const { preamble, sections } = splitHeadings(bodyLines, p.bodyLine, 2);
  if (fm.status === "stub") {
    if (p.body.trim()) warn(file, "This stub has body content. Set status: ready to publish it", p.bodyLine);
    return base;
  }

  if (preamble.join("").trim()) err(file, "Body text must sit under a ## section", p.bodyLine);
  const sec = new Map<string, (typeof sections)[number]>();
  for (const s of sections) {
    if (!BODY_SECTIONS.includes(s.title)) err(file, `Unknown section "## ${s.title}". Sections: ${BODY_SECTIONS.join(", ")}`, s.line);
    else if (sec.has(s.title)) err(file, `Section "## ${s.title}" appears twice`, s.line);
    else sec.set(s.title, s);
  }
  for (const need of fm.type === "intro" ? ["Goal", "Learn", "Done when"] : ["Goal", "Learn", "Build steps", "Done when"]) if (!sec.has(need)) err(file, `A ready lesson needs a "## ${need}" section`);
  const md = (t: string) => (sec.get(t)?.lines.join("\n").trim() || undefined);
  base.goal = md("Goal");
  base.hints = md("Hints");

  // ---------- Learn ----------
  const learnSec = sec.get("Learn");
  if (learnSec) {
    const { fences, prose, unclosedAt } = extractFences(learnSec.lines, learnSec.startLine);
    if (unclosedAt) err(file, "Unclosed ``` fence", unclosedAt);
    if (prose) err(file, "## Learn holds ```card blocks only. Move prose into a card's body", learnSec.line);
    const used = new Set<string>();
    const cardIds = new Set<string>();
    for (const f of fences) {
      if (f.info !== "card") { err(file, `## Learn holds \`\`\`card blocks only, found \`\`\`${f.info}`, f.line); continue; }
      const res = CardSrc.safeParse(safeYaml(f.body, file, (fl, m) => err(fl, m, f.line)));
      if (!res.success) { res.error.issues.forEach((i) => err(file, `card: ${i.path.join(".")}${i.path.length ? ": " : ""}${i.message}`, f.line)); continue; }
      const c = res.data;
      const at = (m: string) => err(file, `card "${"id" in c ? c.id : c.ref}": ${m}`, f.line);
      if (c.type === "quiz") {
        const q = quiz.find((x) => x.id === c.ref);
        if (!q) { at(`no quiz question with id "${c.ref}" in the frontmatter`); continue; }
        if (used.has(c.ref)) at(`quiz "${c.ref}" is placed twice`);
        used.add(c.ref);
        base.learn.push({ type: "quiz", id: q.id, ref: q.id, q: q.q, answer: q.answer, options: q.options });
        continue;
      }
      if (cardIds.has(c.id) || qids.has(c.id)) at("duplicate card id");
      cardIds.add(c.id);
      const card = checkCard(c, at);
      if (card) base.learn.push(card);
    }
    for (const q of quiz) if (!used.has(q.id)) base.learn.push({ type: "quiz", id: q.id, ref: q.id, q: q.q, answer: q.answer, options: q.options });
    if (base.learn.length < 5 || base.learn.length > 8) err(file, `Learn needs 5–8 cards including the quiz, found ${base.learn.length}`, learnSec.line);
  }

  // ---------- Build steps ----------
  const buildSec = sec.get("Build steps");
  if (buildSec) {
    const { preamble: pre, sections: steps } = splitHeadings(buildSec.lines, buildSec.startLine, 3);
    if (pre.join("").trim()) err(file, "Put each build step under a ### heading", buildSec.line);
    if (!steps.length) err(file, "## Build steps needs at least one ### step", buildSec.line);
    const stepIds = new Set<string>();
    for (const s of steps) {
      const { fences, prose, unclosedAt } = extractFences(s.lines, s.startLine);
      if (unclosedAt) err(file, "Unclosed ``` fence", unclosedAt);
      const kinds = fences.filter((f) => STEP_FENCES.includes(f.info));
      if (kinds.length !== 1) { err(file, `Step "${s.title}" needs exactly one \`\`\`challenge, \`\`\`replay or \`\`\`metrics block`, s.line); continue; }
      const step = compileStep(kinds[0], s.title, prose || undefined, p, files, base, err);
      if (!step) continue;
      if (stepIds.has(step.id)) err(file, `Duplicate step id "${step.id}"`, kinds[0].line);
      stepIds.add(step.id);
      base.build.push(step);
    }
  }

  // ---------- Done when ----------
  const doneSec = sec.get("Done when");
  if (doneSec) {
    const { items, errors } = parseDoneWhen(doneSec.lines, doneSec.startLine);
    errors.forEach((e) => err(file, e.message, e.line));
    const minDone = fm.type === "intro" ? 1 : 2;
    if (items.length < minDone || items.length > 4) err(file, `Done when needs ${minDone}–4 checkboxes, found ${items.length}`, doneSec.line);
    const seen = new Set<string>();
    for (const it of items) {
      if (seen.has(it.id)) err(file, `Duplicate Done-when id "${it.id}"`, it.line);
      seen.add(it.id);
      if (it.auto) {
        const [kind, target] = it.auto.split(":");
        const ok = kind === "learn" ? !target
          : kind === "metrics" ? !target && base.build.some((s) => s.kind === "metrics")
          : kind === "challenge" ? base.build.some((s) => s.kind === "challenge" && s.id === target)
          : false;
        if (!ok) err(file, `auto=${it.auto} doesn't match anything. Use learn, metrics, or challenge:<id> of a challenge step`, it.line);
      }
      base.doneWhen.push({ id: it.id, text: it.text, auto: it.auto });
    }
  }
  // ---------- reading: the primer, the lesson fold, then authored paper notes ----------
  const primerFile = p.dir + "primer.md";
  if (files[primerFile] !== undefined) {
    const { doc, issues } = parsePrimer(files[primerFile], base.title);
    issues.forEach((x) => err(primerFile, x.message, x.line));
    if (doc) base.reading.push(doc);
  }
  base.reading.push(lessonFold(base));
  for (const nf of Object.keys(files).filter((k) => k.startsWith(p.dir + "papers/") && k.endsWith(".md")).sort()) {
    const name = nf.slice((p.dir + "papers/").length, -3);
    if (!/^[a-z0-9][a-z0-9-]*$/.test(name)) { err(nf, "Paper notes file names use lower-case-kebab, e.g. llm-int8.md"); continue; }
    const { doc, issues } = parsePaperNotes(files[nf], name, base.papers);
    issues.forEach((x) => err(nf, x.message, x.line));
    if (doc) base.reading.push(doc);
  }
  return base;
}

function checkCard(c: Exclude<z.infer<typeof CardSrc>, { type: "quiz" }>, at: (m: string) => void): Card | null {
  const common = { id: c.id, eyebrow: c.eyebrow, title: c.title };
  switch (c.type) {
    case "concept": {
      const { body, charts, issues } = extractCharts(c.body);
      issues.forEach((x) => at(x.message));
      return { ...common, type: "concept", body, ...(charts.length ? { charts } : {}), visual: c.visual };
    }
    case "slider_viz": {
      const p = c.params;
      if (!(p.min_bits <= p.start_bits && p.start_bits <= p.max_bits)) at("params need min_bits ≤ start_bits ≤ max_bits");
      const best = fewestBitsUnder(p.error_limit, p.weights, p.min_bits, p.max_bits);
      if (best === null) at(`no bit width in ${p.min_bits}–${p.max_bits} gets under ${p.error_limit}`);
      else if (best !== c.answer) at(`answer is ${c.answer}, but the fewest bits under ${p.error_limit} is ${best}`);
      return { ...common, hint: c.hint, type: "slider_viz", viz: c.viz, goal: c.goal, answer: c.answer, body: c.body,
        params: { weights: p.weights, minBits: p.min_bits, maxBits: p.max_bits, startBits: p.start_bits, errorLimit: p.error_limit } };
    }
    case "fill_blank": {
      const slots = [...c.code.matchAll(/\[\[(\d+)\]\]/g)].map((m) => Number(m[1]));
      const sorted = [...slots].sort((a, b) => a - b);
      if (!sorted.every((v, i) => v === i)) at(`slots must be [[0]], [[1]], … each used once; found ${slots.map((s) => `[[${s}]]`).join(", ") || "none"}`);
      if (slots.length !== c.answer.length) at(`${slots.length} slot(s) but ${c.answer.length} answer(s)`);
      for (const a of new Set(c.answer)) {
        const need = c.answer.filter((x) => x === a).length, have = c.tokens.filter((x) => x === a).length;
        if (!have) at(`answer "${a}" isn't one of the tokens`);
        else if (have < need) at(`answer uses "${a}" ${need} times but tokens list it ${have} time${have > 1 ? "s" : ""}; each chip fills one blank`);
      }
      for (const k of Object.keys(c.tips)) {
        if (!c.tokens.includes(k)) at(`tip for "${k}", which isn't a token`);
        if (c.answer.includes(k)) at(`tip for "${k}", which is a correct answer`);
      }
      return { ...common, hint: c.hint, type: "fill_blank", code: c.code, tokens: c.tokens, answer: c.answer, tips: c.tips, whyOk: c.why_ok, whyWrong: c.why_wrong };
    }
    case "predict": {
      const n = c.options.filter((o) => o.correct).length;
      if (n !== 1) at(`exactly one option must be correct: true, found ${n}`);
      return { ...common, hint: c.hint, type: "predict", code: c.code, options: c.options.map((o) => ({ text: o.text, correct: !!o.correct, why: o.why })) };
    }
    case "reorder": {
      const n = c.lines.length;
      const perm = [...c.shuffle].sort((a, b) => a - b);
      if (c.shuffle.length !== n || !perm.every((v, i) => v === i)) at(`shuffle must be a permutation of 0–${n - 1}`);
      else if (c.shuffle.every((v, i) => v === i)) at("shuffle is already in the right order");
      return { ...common, hint: c.hint, type: "reorder", lines: c.lines, shuffle: c.shuffle, whyOk: c.why_ok, whyWrong: c.why_wrong };
    }
    case "calc":
      if (c.answer.min > c.answer.max) at("answer.min is larger than answer.max");
      return { ...common, hint: c.hint, type: "calc", given: c.given, unit: c.unit, answer: c.answer, placeholder: c.placeholder, whyOk: c.why_ok, whyWrong: c.why_wrong };
  }
}

const codeLines = (s: string) => s.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#") && l !== "...");
/** Compare code lines without trailing comments or whitespace. */
const norm = (line: string) => line.replace(/\s+#.*$/, "").replace(/\s+/g, "");

function compileStep(
  f: { info: string; body: string; line: number }, title: string, intro: string | undefined,
  p: Parsed, files: Record<string, string>, lesson: Lesson, err: Report["err"],
): Step | null {
  const { file, dir } = p;
  const at = (m: string) => err(file, `step "${title}": ${m}`, f.line);
  const read = (rel: string) => {
    const src = files[dir + rel];
    if (src === undefined) at(`file ${rel} not found (looked in ${dir})`);
    return src?.replace(/\r\n?/g, "\n");
  };
  const raw = safeYaml(f.body, file, (fl, m) => err(fl, m, f.line)) ?? {};

  if (f.info === "metrics") {
    const r = MetricsStepSrc.safeParse(raw);
    if (!r.success) { r.error.issues.forEach((i) => at(`${i.path.join(".")}: ${i.message}`)); return null; }
    return { kind: "metrics", id: r.data.id, title, intro: r.data.intro ?? intro };
  }

  if (f.info === "replay") {
    const r = ReplaySrc.safeParse(raw);
    if (!r.success) { r.error.issues.forEach((i) => at(`${i.path.join(".")}: ${i.message}`)); return null; }
    const d = r.data;
    const logSrc = read(d.log);
    const log: Extract<Step, { kind: "replay" }>["log"] = [];
    (logSrc ?? "").split("\n").forEach((line, i) => {
      if (!line.trim()) return;
      try {
        const lr = ReplayLine.safeParse(JSON.parse(line));
        if (lr.success) log.push(lr.data);
        else at(`${d.log}:${i + 1}: ${lr.error.issues[0].message}`);
      } catch { at(`${d.log}:${i + 1}: not valid JSON`); }
    });
    if (logSrc !== undefined && !log.length) at(`${d.log} is empty`);
    return {
      kind: "replay", id: d.id, title, intro: d.intro, speed: d.speed, recorded: d.recorded, log,
      chart: d.chart && { title: d.chart.title, flagAbove: d.chart.flag_above, caption: d.chart.caption, xLabels: d.chart.x_labels },
      summary: d.summary, runYourself: d.run_yourself,
    };
  }

  const r = ChallengeSrc.safeParse(raw);
  if (!r.success) { r.error.issues.forEach((i) => at(`${i.path.join(".")}: ${i.message}`)); return null; }
  const c = r.data;
  const tests = read(c.tests) ?? "";
  const solution = read(c.solution) ?? "";
  const testNames = testNamesFrom(tests);
  if (tests && !testNames.length) at(`${c.tests} defines no @test("id", "name") tests`);
  const ids = new Set(testNames.map((t) => t.id));
  if (ids.size !== testNames.length) at(`${c.tests} has duplicate test ids`);

  if ("from" in c.start && !lesson.build.some((s) => s.kind === "challenge" && s.id === (c.start as { from: string }).from)) {
    at(`start.from "${c.start.from}" must name an earlier challenge step`);
  }
  const setup = c.setup.map((s) => {
    if (s.type !== undefined && s.type_file !== undefined) at("a setup segment takes type or type_file, not both");
    const text = s.type_file !== undefined ? read(s.type_file) ?? "" : s.type ?? "";
    return { say: s.say, type: text, pause: s.pause };
  });
  const typed = ("code" in c.start ? c.start.code : "") + setup.map((s) => s.type).join("");
  if (!typed.includes("...")) at("the setup must end on a stub with ... placeholders for the learner to fill");
  // Never type the answer for the learner. Lines compare with comments and whitespace removed.
  // hints.solution must cover every line of the reference solution that the setup (and, for a
  // chained step, the previous step's solution) doesn't already contain, so it is the whole answer.
  const answer = new Map(codeLines(c.hints.solution).map((l) => [norm(l), l]));
  const typedSet = new Set(codeLines(typed).map(norm));
  for (const [n, line] of answer) if (typedSet.has(n)) at(`the narrated setup types a solution line: "${line}"`);
  const prev = "from" in c.start ? lesson.build.find((s) => s.kind === "challenge" && s.id === (c.start as { from: string }).from) : undefined;
  const base = new Set([...typedSet, ...(prev?.kind === "challenge" ? codeLines(prev.solution).map(norm) : [])]);
  const solSet = new Set(codeLines(solution).map(norm));
  for (const [n, line] of answer) if (solution && !solSet.has(n)) at(`hints.solution line "${line}" isn't in ${c.solution}`);
  for (const line of codeLines(solution)) {
    const n = norm(line);
    if (!base.has(n) && !answer.has(n)) at(`${c.solution} line "${line}" is neither in the setup nor in hints.solution`);
  }
  if (!/_{3,}/.test(c.hints.skeleton)) at("hints.skeleton needs ____ blanks");
  // Logic, tools and example sit below the skeleton on the ladder, so none may quote a whole answer
  // line, even in prose maths (× ÷ − ²). Very short lines ("return q") are skipped: they can appear
  // in prose without giving anything away.
  const maths = (s: string) => s.replace(/×/g, "*").replace(/÷/g, "/").replace(/[−–]/g, "-").replace(/²/g, "**2").replace(/'/g, '"').replace(/\s+/g, "");
  for (const k of ["logic", "tools", "example"] as const) {
    const flat = maths(c.hints[k]);
    for (const [n, line] of answer) if (n.length >= 12 && flat.includes(maths(n))) at(`hints.${k} gives away the answer line "${line}"`);
  }
  for (const alt of c.alternatives) read(alt);
  for (const m of c.mistakes) {
    read(m.file);
    if (testNames.length && !ids.has(m.fails)) at(`mistake ${m.file} expects test "${m.fails}", which ${c.tests} doesn't define`);
  }
  return {
    kind: "challenge", id: c.id, title, intro, file: c.file, minutes: c.minutes, start: c.start, setup,
    tests, testNames, solution, hints: c.hints, onSolve: c.on_solve,
  };
}

/** For check-challenges: the raw challenge blocks with their file paths. */
export function challengeSources(files: Record<string, string>) {
  const out: { lesson: string; dir: string; id: string; file: string; tests: string; solution: string; alternatives: string[]; mistakes: { file: string; fails: string; says: string }[] }[] = [];
  for (const file of Object.keys(files)) {
    const m = /^lessons\/([^/]+)\/lesson\.md$/.exec(file);
    if (!m) continue;
    const split = splitFrontmatter(files[file]);
    if (!split) continue;
    const { fences } = extractFences(split.body.split("\n"), split.bodyLine);
    for (const f of fences.filter((x) => x.info === "challenge")) {
      const c = ChallengeSrc.safeParse(parseYaml(f.body));
      if (c.success) out.push({ lesson: m[1], dir: `lessons/${m[1]}/`, id: c.data.id, file: c.data.file, tests: c.data.tests, solution: c.data.solution, alternatives: c.data.alternatives, mistakes: c.data.mistakes });
    }
  }
  return out;
}
