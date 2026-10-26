// npm run check:challenges
// Runs every challenge's reference solution, alternative solutions and known mistakes through
// the real harness in Pyodide. Solutions must pass every test; each mistake must fail first on
// the test it names, with a diagnostic containing the expected words.
import { fileURLToPath } from "node:url";
import { challengeSources } from "./compile.ts";
import { inLessons, onlyIds, readContent } from "./cli.ts";
import { runChallenge, trySnippet, type RunResult } from "./pyrun.ts";

export async function checkChallenges(files: Record<string, string>, only: string[] | null = null) {
  const failures: string[] = [];
  let checked = 0;
  for (const ch of challengeSources(files).filter((c) => !only || inLessons(c.dir, only))) {
    const tests = files[ch.dir + ch.tests];
    const run = (rel: string) => runChallenge(files[ch.dir + rel] ?? "", tests, ch.file);
    const tag = `${ch.lesson} › ${ch.id}`;
    const allPass = (r: RunResult) => r.tests.every((t) => t.status === "pass");
    const summary = (r: RunResult) => r.tests.map((t) => `${t.status === "pass" ? "✓" : "✗"} ${t.id}${t.message ? `: ${t.message}` : ""}`).join("\n      ");

    // Start here: the snippet runs on the starting editor code and prints exactly what the hint says.
    const t = await trySnippet(ch.editor, ch.start.try, ch.file);
    checked++;
    if (t.scriptError) failures.push(`${tag}: hints.start.try raised ${t.scriptError.type}: ${t.scriptError.message}`);
    else if (t.stdout.trimEnd() !== ch.start.see.trimEnd()) failures.push(`${tag}: hints.start.see doesn't match what hints.start.try prints\n      expected: ${JSON.stringify(ch.start.see.trimEnd())}\n      printed:  ${JSON.stringify(t.stdout.trimEnd())}`);

    for (const rel of [ch.solution, ...ch.alternatives]) {
      const r = await run(rel);
      checked++;
      if (!allPass(r)) failures.push(`${tag}: ${rel} should pass every test\n      ${summary(r)}`);
      if (r.scriptError) failures.push(`${tag}: ${rel} raised ${r.scriptError.type}: ${r.scriptError.message}`);
    }
    for (const m of ch.mistakes) {
      const r = await run(m.file);
      checked++;
      const first = r.tests.find((t) => t.status !== "pass");
      if (!first) failures.push(`${tag}: ${m.file} passes every test, but it's listed as a mistake`);
      else if (first.id !== m.fails) failures.push(`${tag}: ${m.file} should fail first on "${m.fails}", but fails first on "${first.id}": ${first.message}`);
      else if (!first.message.includes(m.says)) failures.push(`${tag}: ${m.file} diagnostic should contain "${m.says}"\n      got: ${first.message}`);
      if (r.tests.some((t) => t.status === "error")) failures.push(`${tag}: ${m.file} crashed a test: ${r.tests.find((t) => t.status === "error")!.message}`);
    }
  }
  return { checked, failures };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const { checked, failures } = await checkChallenges(readContent(), onlyIds());
  for (const f of failures) console.log(`✗ ${f}`);
  console.log(failures.length ? `\n${failures.length} problem(s) in ${checked} runs.` : `✓ ${checked} challenge runs behaved as expected (solutions pass, mistakes fail with the right diagnostic).`);
  process.exit(failures.length ? 1 : 0);
}
