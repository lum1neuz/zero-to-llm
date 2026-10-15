// Text rendering: a tiny inline Markdown (code, bold, em) that escapes everything else,
// used for card feedback, hints and test diagnostics (which can echo learner code);
// full Markdown via marked for authored blocks; and the editor's Python highlighter.
import { marked } from "marked";

export const esc = (s: string) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

export function inlineMd(s: string) {
  return s.split(/(`[^`]+`)/g).map((part) => {
    if (part.length > 1 && part.startsWith("`") && part.endsWith("`")) return `<code>${esc(part.slice(1, -1))}</code>`;
    return esc(part)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w*])\*([^*\s][^*]*)\*(?!\w)/g, "$1<em>$2</em>");
  }).join("");
}

/** Authored Markdown blocks (Goal, Hints, card bodies). Content comes from the repo, not learners. */
export function blockMd(s: string) {
  return marked.parse(s, { async: false, gfm: true }) as string;
}

const KW = new Set(["def", "return", "import", "from", "for", "in", "if", "else", "elif", "None", "True", "False", "and", "or", "not", "with", "as", "while", "lambda", "class", "pass", "raise", "try", "except", "assert"]);
export function hl(src: string) {
  const re = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"?|'(?:[^'\\\n]|\\.)*'?)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
  let out = "";
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    if (m[1]) out += `<span class="t-c">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="t-s">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="t-n">${m[3]}</span>`;
    else if (KW.has(m[4])) out += `<span class="t-k">${m[4]}</span>`;
    else if (src[re.lastIndex] === "(") out += `<span class="t-f">${m[4]}</span>`;
    else out += m[4];
    last = re.lastIndex;
  }
  return out + esc(src.slice(last));
}

export const fmtTime = (ms: number) => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
export const plural = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;
