// Markdown structure for lesson files: frontmatter, `## sections`, `### subsections`,
// fenced blocks and the Done-when checklist. Line numbers are 1-based and file-relative.

export interface Fence { info: string; body: string; line: number }
export interface Section { title: string; line: number; lines: string[]; startLine: number }

const FENCE = /^(`{3,}|~{3,})\s*([\w-]*)\s*$/;

export function splitFrontmatter(src: string) {
  const text = src.replace(/^﻿/, "").replace(/\r\n?/g, "\n");
  if (!text.startsWith("---\n")) return null;
  const end = text.indexOf("\n---", 4);
  if (end < 0) return null;
  const after = text.indexOf("\n", end + 4);
  const fm = text.slice(4, end + 1);
  const body = after < 0 ? "" : text.slice(after + 1);
  const bodyLine = fm.split("\n").length + 2;
  return { fm, body, bodyLine };
}

/** Split lines into heading-delimited sections, ignoring headings inside fences. */
export function splitHeadings(lines: string[], startLine: number, level: 2 | 3): { preamble: string[]; sections: Section[] } {
  const marker = "#".repeat(level) + " ";
  const sections: Section[] = [];
  const preamble: string[] = [];
  let inFence: string | null = null;
  let cur: Section | null = null;
  lines.forEach((line, i) => {
    const fm = FENCE.exec(line);
    if (fm) {
      if (inFence === null) inFence = fm[1][0];
      else if (line.trim().startsWith(inFence.repeat(3)) && fm[2] === "") inFence = null;
    }
    if (inFence === null && line.startsWith(marker) && !fm) {
      cur = { title: line.slice(marker.length).trim(), line: startLine + i, lines: [], startLine: startLine + i + 1 };
      sections.push(cur);
      return;
    }
    (cur ? cur.lines : preamble).push(line);
  });
  return { preamble, sections };
}

/** Pull fenced blocks out of a run of lines. Returns the fences and the remaining prose. */
export function extractFences(lines: string[], startLine: number) {
  const fences: Fence[] = [];
  const prose: string[] = [];
  let open: { char: string; len: number; info: string; line: number; body: string[] } | null = null;
  lines.forEach((line, i) => {
    const m = FENCE.exec(line);
    if (!open) {
      if (m) open = { char: m[1][0], len: m[1].length, info: m[2], line: startLine + i, body: [] };
      else prose.push(line);
      return;
    }
    if (m && m[1][0] === open.char && m[1].length >= open.len && m[2] === "") {
      fences.push({ info: open.info, body: open.body.join("\n"), line: open.line });
      open = null;
    } else open.body.push(line);
  });
  const unclosed = open as { line: number } | null;
  return { fences, prose: prose.join("\n").trim(), unclosedAt: unclosed ? unclosed.line : null };
}

export interface DoneItem { id: string; text: string; auto?: string; line: number }

/** `- [ ] Text {#id auto=challenge:x}` */
export function parseDoneWhen(lines: string[], startLine: number) {
  const items: DoneItem[] = [];
  const errors: { line: number; message: string }[] = [];
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line) return;
    const m = /^[-*] \[[ xX]\] (.+?)(?:\s*\{([^}]*)\})?$/.exec(line);
    if (!m) { errors.push({ line: startLine + i, message: `Done-when lines look like "- [ ] Text {#id}", got: ${line}` }); return; }
    const attrs = (m[2] || "").trim().split(/\s+/).filter(Boolean);
    let id = "";
    let auto: string | undefined;
    for (const a of attrs) {
      if (a.startsWith("#")) id = a.slice(1);
      else if (a.startsWith("auto=")) auto = a.slice(5);
      else errors.push({ line: startLine + i, message: `Unknown Done-when attribute "${a}"` });
    }
    if (!id) errors.push({ line: startLine + i, message: `Done-when item needs an id: add {#some_id} at the end` });
    items.push({ id, text: m[1].trim(), auto, line: startLine + i });
  });
  return { items, errors };
}

export function testNamesFrom(src: string) {
  const out: { id: string; name: string }[] = [];
  const re = /@test\(\s*"([^"]+)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)/g;
  for (let m; (m = re.exec(src)); ) out.push({ id: m[1], name: m[2].replace(/\\"/g, '"') });
  return out;
}
