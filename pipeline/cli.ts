// npm run validate  -> check content, print issues
// npm run content   -> check content and write generated/course.json (+ schema, + public copy)
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { compileCourse, type Issue } from "./compile.ts";
import { CourseOut } from "./schema.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const TEXT = /\.(md|ya?ml|py|jsonl)$/;

export function readContent(dir = join(root, "content")) {
  const files: Record<string, string> = {};
  const walk = (d: string) => {
    for (const name of readdirSync(d)) {
      const p = join(d, name);
      if (statSync(p).isDirectory()) walk(p);
      else if (TEXT.test(name)) files[relative(dir, p).split(sep).join("/")] = readFileSync(p, "utf8");
    }
  };
  walk(dir);
  return files;
}

export function runtime() {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  const torch = readFileSync(join(root, "python/torch.py"), "utf8");
  const harness = readFileSync(join(root, "python/harness.py"), "utf8");
  // Served at /runtime/<hash>/: a page only ever loads the mini-torch its tests were written against.
  const hash = createHash("sha256").update(torch).update("\0").update(harness).digest("hex").slice(0, 12);
  return { pyodide: pkg.devDependencies.pyodide as string, torch, harness, hash };
}

export function formatIssue(i: Issue) {
  return `${i.level === "error" ? "✗" : "!"} content/${i.file}${i.line ? `:${i.line}` : ""}  ${i.message}`;
}

/** `--only 1.2,1.3`: lesson ids to report on, so parallel authors only see their own lessons' issues. */
export function onlyIds(argv = process.argv) {
  const i = argv.indexOf("--only");
  return i >= 0 && argv[i + 1] ? argv[i + 1].split(",").map((s) => s.trim()).filter(Boolean) : null;
}
export const inLessons = (file: string, ids: string[]) => ids.some((id) => file.startsWith(`lessons/${id}-`));

function main(cmd: string) {
  const only = onlyIds();
  if (only) {
    // Report issues for the named lessons only (plus course.yaml). Nothing is written.
    const { issues } = compileCourse(readContent(), { runtime: runtime() });
    // Cross-lesson errors land on one lesson's file but can be caused by another (a cut-order clash,
    // a cycle, a fallback rule), so also keep any issue whose message names one of these lessons,
    // plus the global cut-order and course.json checks.
    const named = (msg: string) => only.some((id) => new RegExp(`(^|[^\\w.])${id.replace(/\./g, "\\.")}([^\\w.]|$)`).test(msg));
    const mine = issues.filter((i) => i.file === "course.yaml" || i.file === "course.json" || inLessons(i.file, only)
      || named(i.message) || /^Cut order/.test(i.message));
    for (const i of mine) console.log(formatIssue(i));
    const errors = mine.filter((i) => i.level === "error").length;
    console.log(errors ? `\n${errors} error(s) in ${only.join(", ")}.` : `✓ ${only.join(", ")}: no errors`);
    process.exit(errors ? 1 : 0);
  }
  const { issues, course } = compileCourse(readContent(), { runtime: runtime() });
  for (const i of issues) console.log(formatIssue(i));
  const errors = issues.filter((i) => i.level === "error").length;
  if (errors || !course) {
    console.log(`\n${errors} error(s). course.json not written.`);
    process.exit(1);
  }
  const ready = course.lessons.filter((l) => l.status === "ready").length;
  console.log(`✓ ${course.lessons.length} lessons valid (${ready} ready, ${course.lessons.length - ready} stubs) · ${course.totals.coreLessons} core lessons · ${course.totals.coreHours} h core + ${course.totals.sidequestHours} h sidequests`);
  if (cmd !== "build") return;
  const json = JSON.stringify(course, null, 1);
  mkdirSync(join(root, "generated"), { recursive: true });
  writeFileSync(join(root, "generated/course.json"), json);
  writeFileSync(join(root, "generated/course.schema.json"), JSON.stringify(z.toJSONSchema(CourseOut, { target: "draft-2020-12" }), null, 2));
  mkdirSync(join(root, "public"), { recursive: true });
  writeFileSync(join(root, "public/course.json"), json);
  const rtDir = join(root, "public/runtime", course.runtime.hash);
  rmSync(join(root, "public/runtime"), { recursive: true, force: true });
  mkdirSync(rtDir, { recursive: true });
  writeFileSync(join(rtDir, "torch.py"), course.runtime.torch);
  writeFileSync(join(rtDir, "harness.py"), course.runtime.harness);
  console.log(`✓ wrote generated/course.json (${(json.length / 1024).toFixed(0)} KB), generated/course.schema.json, public/course.json`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main(process.argv[2] ?? "validate");
