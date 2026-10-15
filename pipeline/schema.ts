// Schemas for lesson source (frontmatter + fenced blocks) and for the emitted course.json.
// Source keys are snake_case (what authors type); course.json keys are camelCase (what clients read).
import { z } from "zod";

export const LESSON_ID = /^(?:[0-5]|4A|4B)\.\d+$|^[SW]\d+$/;
export const LessonId = z.string().regex(LESSON_ID, "Lesson ids look like 3.1, 4A.2, W1 or S1");
const Slug = z.string().regex(/^[a-z][a-z0-9_]*$/, "Use lower_snake_case");
const Md = z.string().min(1);

/** intro: a gentle starter lesson. Build steps and metric targets are optional. */
export const LESSON_TYPES = ["intro", "build", "run", "research", "sidequest"] as const;

// ---------- course.yaml ----------
export const PhaseSrc = z.strictObject({
  id: z.string(),
  title: z.string(),
  weeks: z.tuple([z.int(), z.int()]).nullable(),
  protected: z.boolean().default(false),
  optional: z.boolean().default(false),
  blurb: z.string(),
});
export const CourseSrc = z.strictObject({
  title: z.string(),
  subtitle: z.string(),
  weeks: z.int().positive(),
  phases: z.array(PhaseSrc).min(1),
});

// ---------- frontmatter ----------
const Paper = z.strictObject({
  title: z.string(),
  authors: z.string().optional(),
  year: z.int().optional(),
  url: z.url().optional(),
});
const MetricInput = z.strictObject({ id: Slug, label: z.string(), hint: z.string().optional() });
export const Metric = z.strictObject({
  id: Slug,
  label: z.string(),
  unit: z.string(),
  note: z.string().optional(),
  inputs: z.array(MetricInput).min(1).optional(),
  compute: z.enum(["value", "pct_change", "ratio"]).optional(),
  reference: z.number().optional(),
  pass: z.strictObject({ max: z.number().optional(), min: z.number().optional() }).optional(),
});
export const QuizItem = z.strictObject({
  id: Slug,
  q: z.string(),
  answer: z.int(),
  options: z.array(z.strictObject({ text: z.string(), why: Md })),
});
export const Frontmatter = z.strictObject({
  id: LessonId,
  title: z.string().min(1),
  summary: z.string().optional(),
  phase: z.string(),
  hours: z.number().positive(),
  type: z.enum(LESSON_TYPES),
  status: z.enum(["stub", "ready"]),
  protected: z.boolean(),
  cut: z.int().min(1).nullable(),
  prerequisites: z.array(LessonId),
  fallbacks: z.record(z.string(), z.string()).default({}),
  papers: z.array(Paper).default([]),
  quality_metric: Metric,
  systems_metric: Metric,
  quiz: z.array(QuizItem).optional(),
});
export type FrontmatterT = z.infer<typeof Frontmatter>;

// ---------- Learn cards (```card fences) ----------
const cardBase = { id: Slug, eyebrow: z.string().optional(), title: z.string().optional() };
const exercise = { hint: z.string().optional() };

const Bars = z.strictObject({
  kind: z.literal("bars"),
  max: z.number().positive(),
  bars: z.array(z.strictObject({ label: z.string(), value: z.number().nonnegative(), text: z.string().optional() })).min(1),
  limit: z.strictObject({ value: z.number().positive(), label: z.string() }).optional(),
});

export const CardSrc = z.discriminatedUnion("type", [
  z.strictObject({ ...cardBase, type: z.literal("concept"), body: Md, visual: Bars.optional() }),
  z.strictObject({
    ...cardBase, ...exercise, type: z.literal("slider_viz"),
    viz: z.literal("quant_grid"),
    params: z.strictObject({
      weights: z.array(z.number()).min(2),
      min_bits: z.int().min(2), max_bits: z.int().max(16), start_bits: z.int(),
      error_limit: z.number().positive(),
    }),
    goal: z.literal("fewest_bits_under_limit"),
    answer: z.int(),
    body: Md.optional(),
  }),
  z.strictObject({
    ...cardBase, ...exercise, type: z.literal("fill_blank"),
    code: z.string(), tokens: z.array(z.string()).min(2), answer: z.array(z.string()).min(1),
    tips: z.record(z.string(), Md).default({}), why_ok: Md, why_wrong: Md,
  }),
  z.strictObject({
    ...cardBase, ...exercise, type: z.literal("predict"),
    code: z.string(),
    options: z.array(z.strictObject({ text: z.string(), correct: z.boolean().optional(), why: Md })).min(2),
  }),
  z.strictObject({
    ...cardBase, ...exercise, type: z.literal("reorder"),
    lines: z.array(z.string()).min(2), shuffle: z.array(z.int()), why_ok: Md, why_wrong: Md,
  }),
  z.strictObject({
    ...cardBase, ...exercise, type: z.literal("calc"),
    given: z.array(z.strictObject({ label: z.string(), value: z.string() })).default([]),
    unit: z.string(), answer: z.strictObject({ min: z.number(), max: z.number() }),
    placeholder: z.string().optional(), why_ok: Md, why_wrong: Md,
  }),
  z.strictObject({ type: z.literal("quiz"), ref: Slug }),
]);
export type CardSrcT = z.infer<typeof CardSrc>;

// ---------- Build steps (```challenge / ```replay / ```metrics fences) ----------
const Segment = z.strictObject({
  say: z.string(),
  type: z.string().optional(),
  type_file: z.string().optional(),
  pause: z.number().nonnegative().default(1.5),
});
export const ChallengeSrc = z.strictObject({
  id: Slug,
  file: z.string().regex(/^[\w.-]+\.py$/),
  minutes: z.int().positive().optional(),
  start: z.union([z.strictObject({ code: z.string() }), z.strictObject({ from: Slug })]),
  setup: z.array(Segment).min(1),
  tests: z.string(),
  solution: z.string(),
  alternatives: z.array(z.string()).default([]),
  hints: z.strictObject({ nudge: Md, concept: Md, skeleton: z.string(), solution: z.string() }),
  on_solve: Md,
  mistakes: z.array(z.strictObject({ file: z.string(), fails: Slug, says: z.string() })).default([]),
});
export const ReplaySrc = z.strictObject({
  id: Slug,
  intro: Md,
  log: z.string(),
  speed: z.number().positive().default(40),
  recorded: z.strictObject({
    hardware: z.string(), date: z.string().optional(), illustrative: z.boolean(), note: z.string().optional(),
  }),
  chart: z.strictObject({ title: z.string(), flag_above: z.number(), caption: Md.optional(), x_labels: z.tuple([z.string(), z.string()]).optional() }).optional(),
  summary: z.array(z.strictObject({ label: z.string(), value: z.string() })).default([]),
  run_yourself: z.string().optional(),
});
export const MetricsStepSrc = z.strictObject({ id: Slug, intro: Md.optional() });

export const ReplayLine = z.strictObject({
  t: z.string(),
  c: z.enum(["", "dim", "hi", "ok", "bad"]).default(""),
  layer: z.int().nonnegative().optional(),
  label: z.string().optional(),
  value: z.number().optional(),
  delay: z.int().nonnegative().optional(),
});

// ---------- course.json (output) ----------
const MetricOut = z.object({
  id: z.string(), label: z.string(), unit: z.string(), note: z.string().optional(),
  inputs: z.array(z.object({ id: z.string(), label: z.string(), hint: z.string().optional() })).optional(),
  compute: z.enum(["value", "pct_change", "ratio"]).optional(),
  reference: z.number().optional(),
  pass: z.object({ max: z.number().optional(), min: z.number().optional() }).optional(),
});
const QuizOut = z.object({
  id: z.string(), q: z.string(), answer: z.int(),
  options: z.array(z.object({ text: z.string(), why: z.string() })),
});
/** A chart from a ```chart fence (see pipeline/chart.ts). The Markdown keeps a `<!-- chart:N -->` line where it goes. */
export const ChartOut = z.object({
  kind: z.enum(["bar", "line", "scatter"]), title: z.string(),
  x: z.object({ label: z.string(), values: z.union([z.array(z.number()), z.array(z.string())]) }),
  y: z.object({ label: z.string().optional(), min: z.number().optional(), max: z.number().optional() }).optional(),
  series: z.array(z.object({ name: z.string(), values: z.array(z.number()) })),
  highlight: z.object({ x: z.union([z.number(), z.string()]), series: z.string().optional(), label: z.string() }).optional(),
  threshold: z.object({ y: z.number(), label: z.string() }).optional(),
  illustrative: z.boolean(), caption: z.string(),
});
export type Chart = z.infer<typeof ChartOut>;
const CardCommon = { id: z.string(), eyebrow: z.string().optional(), title: z.string().optional(), hint: z.string().optional() };
export const CardOut = z.discriminatedUnion("type", [
  z.object({ ...CardCommon, type: z.literal("concept"), body: z.string(), charts: z.array(ChartOut).optional(), visual: z.object({
    kind: z.literal("bars"), max: z.number(), bars: z.array(z.object({ label: z.string(), value: z.number(), text: z.string().optional() })),
    limit: z.object({ value: z.number(), label: z.string() }).optional() }).optional() }),
  z.object({ ...CardCommon, type: z.literal("slider_viz"), viz: z.literal("quant_grid"), params: z.object({
    weights: z.array(z.number()), minBits: z.int(), maxBits: z.int(), startBits: z.int(), errorLimit: z.number() }),
    goal: z.literal("fewest_bits_under_limit"), answer: z.int(), body: z.string().optional() }),
  z.object({ ...CardCommon, type: z.literal("fill_blank"), code: z.string(), tokens: z.array(z.string()), answer: z.array(z.string()),
    tips: z.record(z.string(), z.string()), whyOk: z.string(), whyWrong: z.string() }),
  z.object({ ...CardCommon, type: z.literal("predict"), code: z.string(),
    options: z.array(z.object({ text: z.string(), correct: z.boolean(), why: z.string() })) }),
  z.object({ ...CardCommon, type: z.literal("reorder"), lines: z.array(z.string()), shuffle: z.array(z.int()), whyOk: z.string(), whyWrong: z.string() }),
  z.object({ ...CardCommon, type: z.literal("calc"), given: z.array(z.object({ label: z.string(), value: z.string() })), unit: z.string(),
    answer: z.object({ min: z.number(), max: z.number() }), placeholder: z.string().optional(), whyOk: z.string(), whyWrong: z.string() }),
  z.object({ ...CardCommon, type: z.literal("quiz"), ref: z.string(), q: z.string(), answer: z.int(),
    options: z.array(z.object({ text: z.string(), why: z.string() })) }),
]);
export const StepOut = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("challenge"), id: z.string(), title: z.string(), intro: z.string().optional(), file: z.string(), minutes: z.int().optional(),
    start: z.union([z.object({ code: z.string() }), z.object({ from: z.string() })]),
    setup: z.array(z.object({ say: z.string(), type: z.string(), pause: z.number() })),
    tests: z.string(), testNames: z.array(z.object({ id: z.string(), name: z.string() })), solution: z.string(),
    hints: z.object({ nudge: z.string(), concept: z.string(), skeleton: z.string(), solution: z.string() }),
    onSolve: z.string(),
  }),
  z.object({
    kind: z.literal("replay"), id: z.string(), title: z.string(), intro: z.string(), speed: z.number(),
    recorded: z.object({ hardware: z.string(), date: z.string().optional(), illustrative: z.boolean(), note: z.string().optional() }),
    log: z.array(z.object({ t: z.string(), c: z.string(), layer: z.int().optional(), label: z.string().optional(), value: z.number().optional(), delay: z.int().optional() })),
    chart: z.object({ title: z.string(), flagAbove: z.number(), caption: z.string().optional(), xLabels: z.tuple([z.string(), z.string()]).optional() }).optional(),
    summary: z.array(z.object({ label: z.string(), value: z.string() })),
    runYourself: z.string().optional(),
  }),
  z.object({ kind: z.literal("metrics"), id: z.string(), title: z.string(), intro: z.string().optional() }),
]);
export const FoldSectionOut = z.object({
  id: z.string(), title: z.string(), ref: z.string().optional(), key: z.string().optional(), body: z.string().optional(),
  charts: z.array(ChartOut).optional(),
  /** "learn": the body gives answers away, so readers show it only after Learn is done. */
  locked: z.literal("learn").optional(),
});
export const FoldDocOut = z.object({
  id: z.string(), kind: z.enum(["primer", "lesson", "paper"]), title: z.string(), gist: z.string(), minutes: z.int().optional(),
  source: z.object({ citation: z.string(), url: z.string().optional() }).optional(),
  sections: z.array(FoldSectionOut),
});
export type FoldDoc = z.infer<typeof FoldDocOut>;
export type FoldSection = z.infer<typeof FoldSectionOut>;

export const LessonOut = z.object({
  id: z.string(), title: z.string(), summary: z.string().optional(), phase: z.string(), order: z.int(),
  hours: z.number(), type: z.enum(LESSON_TYPES), status: z.enum(["stub", "ready"]), protected: z.boolean(), cut: z.int().nullable(),
  prerequisites: z.array(z.string()), fallbacks: z.record(z.string(), z.string()),
  papers: z.array(z.object({ title: z.string(), authors: z.string().optional(), year: z.int().optional(), url: z.string().optional() })),
  qualityMetric: MetricOut, systemsMetric: MetricOut, quiz: z.array(QuizOut),
  goal: z.string().optional(), learn: z.array(CardOut), build: z.array(StepOut), hints: z.string().optional(),
  doneWhen: z.array(z.object({ id: z.string(), text: z.string(), auto: z.string().optional() })),
  /** Fold reader documents: the lesson itself (ready lessons), then authored paper notes. */
  reading: z.array(FoldDocOut),
});
export const CourseOut = z.object({
  schemaVersion: z.literal(1),
  generatedAt: z.string(),
  title: z.string(), subtitle: z.string(), weeks: z.int(),
  phases: z.array(z.object({ id: z.string(), title: z.string(), weeks: z.tuple([z.int(), z.int()]).nullable(), protected: z.boolean(), optional: z.boolean(), blurb: z.string() })),
  cutOrder: z.array(z.string()),
  totals: z.object({ coreLessons: z.int(), coreHours: z.number(), sidequestHours: z.number() }),
  lessons: z.array(LessonOut),
  runtime: z.object({ pyodide: z.string(), torch: z.string(), harness: z.string(), hash: z.string() }),
});
export type Course = z.infer<typeof CourseOut>;
export type Lesson = z.infer<typeof LessonOut>;
export type Card = z.infer<typeof CardOut>;
export type Step = z.infer<typeof StepOut>;
export type ChallengeStep = Extract<Step, { kind: "challenge" }>;
export type ReplayStep = Extract<Step, { kind: "replay" }>;
export type MetricDef = z.infer<typeof MetricOut>;
