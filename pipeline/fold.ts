// Fold documents: one text at four depths. Gist (one line) → Map (section titles) →
// Keys (one key sentence per section) → Full (the section's text). Every key belongs to the
// section it summarises, so the reader can unfold exactly the part a key points at.
//
// Two sources:
//  - lesson folds, derived from what a ready lesson already contains (no extra authoring). The lesson-level
//    ## Hints (general tips) are included; challenge hint ladders and reference solutions never are;
//  - paper notes, authored as content/lessons/<id>-<slug>/papers/<name>.md.
import { parse as parseYaml } from "yaml";
import { splitFrontmatter, splitHeadings } from "./parse.ts";
import type { FoldDoc, FoldSection, Lesson } from "./schema.ts";

const firstSentence = (md: string) => {
  const flat = md.replace(/```[\s\S]*?```/g, " ").replace(/\s+/g, " ").trim();
  const m = /^(.+?[.!?])(\s|$)/.exec(flat);
  return (m ? m[1] : flat).trim();
};

/** Build the lesson's own fold from its Goal, cards, steps and Done-when. */
export function lessonFold(l: Lesson): FoldDoc {
  const sections: FoldSection[] = [];
  if (l.goal) sections.push({ id: "goal", title: "Goal", key: firstSentence(l.goal), body: l.goal });
  for (const c of l.learn) {
    if (c.type === "quiz") {
      // An out-of-range answer is reported by the validator; don't crash on it here.
      const right = c.options[c.answer];
      sections.push({ id: `card-${c.id}`, title: "Check yourself", key: c.q, body: right ? `**${right.text}.** ${right.why}` : undefined, locked: "learn" });
    } else if (c.type === "concept") {
      sections.push({ id: `card-${c.id}`, title: c.title ?? c.eyebrow ?? "Concept", key: firstSentence(c.body), body: c.body });
    } else {
      // Exercises: the question is the key; the explanation gives the answer away, so it waits for Learn.
      sections.push({ id: `card-${c.id}`, title: c.title ?? c.eyebrow ?? "Exercise", body: "whyOk" in c ? c.whyOk : undefined, locked: "learn" });
    }
  }
  for (const s of l.build) {
    const narration = s.kind === "challenge" ? s.setup.map((x) => x.say).join(" ") : undefined;
    const text = s.kind === "challenge" ? [s.intro, narration].filter(Boolean).join("\n\n")
      : s.kind === "replay" ? `${s.recorded.illustrative ? "Illustrative recording, not a real run. " : "Recording. "}${s.intro}` : s.intro;
    sections.push({ id: `step-${s.id}`, title: `Build: ${s.title}`, key: text ? firstSentence(text) : undefined, body: text || undefined });
  }
  if (l.hints) sections.push({ id: "hints", title: "When you're stuck", key: firstSentence(l.hints.replace(/^[-*]\s+/gm, "")), body: l.hints });
  if (l.doneWhen.length) sections.push({ id: "done", title: "Done when", key: `${l.doneWhen.length} checks finish the lesson.`, body: l.doneWhen.map((d) => `- ${d.text}`).join("\n") });
  return { id: "lesson", kind: "lesson", title: l.title, gist: l.summary ?? (l.goal ? firstSentence(l.goal) : l.title), sections };
}

export interface NoteIssue { line?: number; message: string }

/**
 * Paper notes format:
 *   ---
 *   paper: LLM.int8()          # title of one of the lesson's frontmatter papers
 *   gist: One sentence.
 *   ---
 *   ## Section title {#id ref="§3.2"}
 *   > The key sentence for this section.
 *
 *   Detailed notes in Markdown, in your own words.
 */
export function parsePaperNotes(src: string, name: string, papers: Lesson["papers"]): { doc?: FoldDoc; issues: NoteIssue[] } {
  const issues: NoteIssue[] = [];
  const split = splitFrontmatter(src);
  if (!split) return { issues: [{ line: 1, message: "Paper notes start with a --- frontmatter block (paper, gist)" }] };
  let fm: Record<string, unknown> = {};
  try { fm = (parseYaml(split.fm) ?? {}) as Record<string, unknown>; } catch (e) { issues.push({ line: 1, message: `YAML: ${(e as Error).message.split("\n")[0]}` }); }
  for (const k of Object.keys(fm)) if (!["paper", "gist"].includes(k)) issues.push({ line: 1, message: `Unknown key "${k}" (paper notes take paper and gist)` });
  const paper = papers.find((p) => p.title === fm.paper);
  if (!paper) issues.push({ line: 1, message: `paper: "${String(fm.paper ?? "")}" must match the title of one of the lesson's papers (${papers.map((p) => p.title).join("; ") || "none listed"})` });
  if (typeof fm.gist !== "string" || !fm.gist.trim()) issues.push({ line: 1, message: "gist: one sentence is required" });

  const { preamble, sections } = splitHeadings(split.body.split("\n"), split.bodyLine, 2);
  if (preamble.join("").trim()) issues.push({ line: split.bodyLine, message: "Put notes under ## section headings" });
  if (!sections.length) issues.push({ line: split.bodyLine, message: "Paper notes need at least one ## section" });
  const out: FoldSection[] = [];
  const ids = new Set<string>();
  for (const s of sections) {
    const m = /^(.*?)\s*(?:\{([^}]*)\})?$/.exec(s.title)!;
    const title = m[1].trim();
    const attrs = m[2] ?? "";
    const id = /#([a-z][a-z0-9_-]*)/.exec(attrs)?.[1];
    const ref = /ref="([^"]+)"/.exec(attrs)?.[1];
    if (!id) { issues.push({ line: s.line, message: `Section "${title}" needs an id: ## ${title} {#some_id}` }); continue; }
    if (ids.has(id)) issues.push({ line: s.line, message: `Duplicate section id "${id}"` });
    ids.add(id);
    const lines = [...s.lines];
    while (lines.length && !lines[0].trim()) lines.shift();
    const keyLines: string[] = [];
    while (lines.length && lines[0].startsWith(">")) keyLines.push(lines.shift()!.replace(/^>\s?/, ""));
    const key = keyLines.join(" ").trim();
    const body = lines.join("\n").trim();
    if (!key) issues.push({ line: s.line, message: `Section "${title}" needs a key sentence: a "> " line right under the heading` });
    if (!body) issues.push({ line: s.line, message: `Section "${title}" needs notes under its key sentence` });
    out.push({ id, title, ref, key: key || undefined, body: body || undefined });
  }
  if (issues.length) return { issues };
  return {
    doc: {
      id: `paper-${name}`, kind: "paper", title: paper!.title, gist: String(fm.gist).trim(), sections: out,
      source: { citation: [paper!.authors, paper!.year].filter(Boolean).join(", "), url: paper!.url },
    },
    issues,
  };
}
